from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, Tuple, Any

import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
from torch import device as torch_device, cuda

from src.data.model_inputs.factory import create_model_input_builder
from src.data.processed import create_processed_loader
from src.data.splitting.splitter_factory import create_splitter
from src.evaluation.evaluator_factory import create_evaluator
from src.models.model_factory import create_model
from src.training.trainer_factory import create_trainer

try:  # optional dependency
    import optuna
except ImportError:  # pragma: no cover
    optuna = None


def _clone_cfg(cfg_section):
    return OmegaConf.create(OmegaConf.to_container(cfg_section, resolve=True))


def _format_combo(combo: Dict[str, object]) -> str:
    return "_".join(f"{k}-{v}" for k, v in combo.items()) if combo else "default"


def _to_plain(value):
    if isinstance(value, ListConfig):
        return list(value)
    return value


def _coerce_numeric(value):
    if isinstance(value, str):
        try:
            if "." in value or "e" in value.lower():
                return float(value)
            return int(value)
        except ValueError:
            return value
    return value


def _parse_range_spec(value):
    if isinstance(value, ListConfig):
        plain = list(value)
        if len(plain) == 1:
            return _parse_range_spec(plain[0])
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1].strip()
        if ":" in text:
            parts = [p.strip() for p in text.split(":")]
            if len(parts) < 2 or len(parts) > 3:
                raise ValueError(f"Invalid range specification: {value}")
            start = float(parts[0])
            end = float(parts[1])
            step = float(parts[2]) if len(parts) == 3 else None
            dtype = int if float(start).is_integer() and float(end).is_integer() and (step is None or float(step).is_integer()) else float
            return {"type": "range", "start": start, "end": end, "step": step, "dtype": dtype}
    return None


def _expand_for_grid(value):
    spec = _parse_range_spec(value)
    if spec is None:
        plain = _to_plain(value)
        if isinstance(plain, list):
            return [_coerce_numeric(v) for v in plain]
        return [_coerce_numeric(plain)]
    if spec["step"] is None:
        raise ValueError("Range specification requires a step when using grid search. Provide [start:end:step].")
    start = spec["start"]
    end = spec["end"]
    step = spec["step"]
    if spec["dtype"] == int:
        start = int(start)
        end = int(end)
        step = int(step)
        if step == 0:
            raise ValueError("Step cannot be zero for range specification.")
        return list(range(start, end + (1 if step > 0 else -1), step))
    else:
        values = []
        current = start
        if step == 0:
            raise ValueError("Step cannot be zero for range specification.")
        compare = (lambda a, b: a <= b) if step > 0 else (lambda a, b: a >= b)
        while compare(current, end):
            values.append(current)
            current += step
        return values


def _suggest_optuna(trial, name: str, value):
    spec = _parse_range_spec(value)
    if spec is None:
        plain = _to_plain(value)
        if not isinstance(plain, (list, tuple)):
            plain = [plain]
        choices = [_coerce_numeric(v) for v in plain]
        return trial.suggest_categorical(name, choices)
    if spec["dtype"] == int:
        return trial.suggest_int(name, int(spec["start"]), int(spec["end"]))
    return trial.suggest_float(name, spec["start"], spec["end"])


def _prepare_data(cfg: DictConfig):
    dataset_context = create_processed_loader(cfg.preprocessor).load()
    splitter = create_splitter(cfg.splitter)
    train_split, val_split, test_split = splitter.split(dataset_context)

    model_input_builder = create_model_input_builder(
        cfg.model_inputs,
        log_level=cfg.logging.level,
        run_mode=cfg.run.mode,
    )
    voc_size = dataset_context.vocab_sizes()
    train_data, val_data, test_data = model_input_builder.run(
        source=dataset_context.name,
        context=dataset_context,
        train_data=train_split,
        val_data=val_split,
        test_data=test_split,
        voc_size=voc_size,
    )
    return dataset_context, (train_data, val_data, test_data)


def _run_single(
    combo: Dict[str, object],
    cfg: DictConfig,
    dataset_context,
    data_splits: Tuple,
    run_dir: Path,
    tuning_cfg: DictConfig,
    device,
):
    train_data, val_data, _ = data_splits
    combo = {k: _coerce_numeric(v) for k, v in combo.items()}
    combo_name = _format_combo(combo)

    model_cfg = _clone_cfg(cfg.model)
    for key, value in combo.items():
        model_cfg[key] = value
    model = create_model(model_cfg, device=device)

    run_dir.mkdir(parents=True, exist_ok=True)

    trainer_cfg = _clone_cfg(cfg.trainer)
    trainer_cfg.save_dir = str(run_dir / "checkpoints")
    trainer = create_trainer(trainer_cfg)
    trainer_metrics = trainer.train(model, train_data, val_data, context=dataset_context)

    if tuning_cfg.metric_source == "trainer":
        metrics = trainer_metrics or {}
    else:
        evaluator_cfg = _clone_cfg(cfg.evaluator)
        evaluator_cfg.save_dir = str(run_dir / "eval")
        evaluator = create_evaluator(evaluator_cfg)
        metric_names = tuning_cfg.get("evaluator_metric_names", [])
        if not metric_names:
            raise ValueError("evaluator_metric_names must be specified when metric_source='evaluator'")
        eval_values = evaluator.evaluate(model, val_data, context=dataset_context)
        metrics = dict(zip(metric_names, eval_values))

    metric_name = tuning_cfg.metric
    if metric_name not in metrics:
        raise ValueError(f"Metric '{metric_name}' not found for combo {combo_name}")

    score = metrics[metric_name]
    return score, metrics, model


def _grid_search(cfg, tuning_cfg, dataset_context, data_splits, work_dir, device):
    param_grid = tuning_cfg.get("params", {})
    names = list(param_grid.keys())
    values = [_expand_for_grid(param_grid[name]) for name in names]
    combinations = list(itertools.product(*values)) if names else [()]

    best_score = None
    best_combo = None
    maximize = tuning_cfg.get("maximize", True)

    for idx, vals in enumerate(combinations):
        combo = dict(zip(names, vals))
        run_dir = work_dir / f"grid_{idx}_{_format_combo(combo)}"
        score, metrics, _ = _run_single(combo, cfg, dataset_context, data_splits, run_dir, tuning_cfg, device)

        is_better = (best_score is None) or (score > best_score if maximize else score < best_score)
        if is_better:
            best_score = score
            best_combo = combo

        print(f"Grid combo {_format_combo(combo)} -> {metrics}")

    return best_combo, best_score


def _optuna_search(cfg, tuning_cfg, dataset_context, data_splits, work_dir, device):
    if optuna is None:
        raise ImportError("Optuna is not installed. Please pip install optuna to use this feature.")

    param_grid = tuning_cfg.get("params", {})
    print(param_grid)
    maximize = tuning_cfg.get("maximize", True)
    metric_name = tuning_cfg.metric

    def objective(trial: "optuna.trial.Trial"):
        combo = {}
        for name, values in param_grid.items():
            combo[name] = _coerce_numeric(_suggest_optuna(trial, name, values))
        run_dir = work_dir / f"optuna_trial_{trial.number}"
        score, metrics, _ = _run_single(combo, cfg, dataset_context, data_splits, run_dir, tuning_cfg, device)
        trial.set_user_attr("params", combo)
        trial.set_user_attr("metrics", metrics)
        return score

    direction = "maximize" if maximize else "minimize"
    study = optuna.create_study(direction=direction)
    n_trials = tuning_cfg.get("optuna", {}).get("n_trials", 20)
    study.optimize(objective, n_trials=n_trials)

    best_trial = study.best_trial
    best_combo = best_trial.user_attrs.get("params", best_trial.params)
    best_score = best_trial.value
    print(f"Optuna best trial #{best_trial.number}: params={best_combo}, {metric_name}={best_score}")
    return best_combo, best_score


def _evaluate_best(cfg, best_combo, dataset_context, data_splits, tuning_cfg, device):
    train_data, val_data, test_data = data_splits

    best_model_cfg = _clone_cfg(cfg.model)
    for key, value in best_combo.items():
        best_model_cfg[key] = value
    best_model = create_model(best_model_cfg, device=device)

    trainer_cfg = _clone_cfg(cfg.trainer)
    trainer_cfg.save_dir = tuning_cfg.get("log_best_dir", "tuning_runs/best")
    trainer = create_trainer(trainer_cfg)
    trainer.train(best_model, train_data, val_data, context=dataset_context)

    evaluator = create_evaluator(cfg.evaluator)
    results = evaluator.evaluate(best_model, test_data, context=dataset_context)
    metric_names = tuning_cfg.get("evaluator_metric_names", [])
    if metric_names:
        results = dict(zip(metric_names, results))
    print(f"Best combo test metrics: {results}")


@hydra.main(config_path="config", config_name="config", version_base=None)
def tune(cfg: DictConfig):
    tuning_cfg = cfg.get("tuning")
    if tuning_cfg is None:
        raise ValueError("tuning section missing from config")

    device = torch_device("cuda" if cuda.is_available() else "cpu")
    dataset_context, data_splits = _prepare_data(cfg)

    work_dir = Path(tuning_cfg.get("work_dir", "tuning_runs"))
    work_dir.mkdir(parents=True, exist_ok=True)

    method = tuning_cfg.get("method", "grid").lower()
    if method == "grid":
        best_combo, best_score = _grid_search(cfg, tuning_cfg, dataset_context, data_splits, work_dir, device)
    elif method == "optuna":
        best_combo, best_score = _optuna_search(cfg, tuning_cfg, dataset_context, data_splits, work_dir, device)
    else:
        raise ValueError(f"Unknown tuning method: {method}")

    if best_combo is None:
        print("No valid combinations evaluated.")
        return

    print(f"Selected best combo {best_combo} with {tuning_cfg.metric}={best_score}")

    if tuning_cfg.get("evaluate_best", True):
        _evaluate_best(cfg, best_combo, dataset_context, data_splits, tuning_cfg, device)


if __name__ == "__main__":
    tune()

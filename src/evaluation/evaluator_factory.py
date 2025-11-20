from src.evaluation.GAMENetEvaluator import GAMENetEvaluator
from src.evaluation.KNN_Evaluator import KNNEvaluator


def create_evaluator(cfg):
    """
    Factory that returns the correct trainer object based on cfg.evaluator.name.
    """
    name = cfg.name.lower()

    if name == "gamenet":
        return GAMENetEvaluator(
            save_dir=cfg.save_dir,
            log_level=cfg.log_level,
            show_progress=cfg.show_progress,
        )
    if name == "knn":
        return KNNEvaluator(
            save_dir=cfg.save_dir,
            log_level=cfg.log_level,
            show_progress=cfg.show_progress,
        )

    raise ValueError(f"Unknown evaluator: {cfg.name}")

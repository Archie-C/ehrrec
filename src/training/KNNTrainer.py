from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np

from src.core.interfaces.trainer import Trainer
from src.utils.metrics import ddi_rate_score, multi_label_metric


class KNNTrainer(Trainer):
    """
    Trainer for the non-parametric KNN model.
    Fits the memory bank (diagnosis/procedure vectors) and
    optionally evaluates on a validation split.
    """

    def __init__(
        self,
        save_dir: str | Path = "saved/KNN",
        log_level: int = logging.INFO,
        show_progress: bool = True,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(handler)
            self.logger.propagate = False
        self.show_progress = show_progress

    def train(self, model, train_data, val_data, context=None) -> Dict[str, float]:
        self.logger.info("Fitting KNN memory with %d admissions", len(train_data["features"]))
        model.fit(train_data["features"], train_data["labels"])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.save_dir / "knn_model.pt"
        if hasattr(model, "save_memory"):
            model.save_memory(model_path)
        else:
            model.save(model_path)
        self.logger.info("Saved KNN memory to %s", model_path)
        metrics: Dict[str, float] = {}
        ddi_adj = context.get_adj("ddi") if context else None
        if val_data and ddi_adj is not None:
            metrics = self._evaluate_split(model, val_data, ddi_adj)
            self.logger.info(
                "Validation metrics | JA=%.4f | PRAUC=%.4f | AvgP=%.4f | AvgR=%.4f | AvgF1=%.4f | DDI=%.4f",
                metrics["ja"],
                metrics["prauc"],
                metrics["avg_p"],
                metrics["avg_r"],
                metrics["avg_f1"],
                metrics["ddi_rate"],
            )
        return metrics

    # ------------------------------------------------------------------
    def _evaluate_split(self, model, split_data, ddi_adj):
        features = split_data["features"]
        labels = split_data["labels"]
        if features.size(0) == 0:
            return {"ja": 0.0, "prauc": 0.0, "avg_p": 0.0, "avg_r": 0.0, "avg_f1": 0.0, "ddi_rate": 0.0}
        preds, scores = model.predict_with_scores(features)

        y_true = labels.detach().cpu().numpy()
        y_pred = preds.detach().cpu().numpy()
        y_prob = scores.detach().cpu().numpy()

        ja, prauc, avg_p, avg_r, avg_f1 = multi_label_metric(y_true, y_pred, y_prob)
        ddi_rate = 0.0
        if ddi_adj is not None:
            ddi_rate = self._compute_ddi_rate(split_data["patient_counts"], y_pred, ddi_adj)

        return {
            "ja": ja,
            "prauc": prauc,
            "avg_p": avg_p,
            "avg_r": avg_r,
            "avg_f1": avg_f1,
            "ddi_rate": ddi_rate,
        }

    def _compute_ddi_rate(self, patient_counts, y_pred, ddi_adj):
        idx = 0
        smm_record = []
        for count in patient_counts:
            adm_preds = []
            for _ in range(count):
                labels = np.nonzero(y_pred[idx])[0].tolist()
                adm_preds.append(labels)
                idx += 1
            smm_record.append(adm_preds)
        return ddi_rate_score(smm_record, ddi_adj)

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import dill

from src.core.interfaces.evaluator import Evaluator
from src.utils.metrics import ddi_rate_score, multi_label_metric


class KNNEvaluator(Evaluator):
    def __init__(self, save_dir: str | Path = "results/KNN", log_level=logging.INFO, show_progress: bool = True):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(handler)
            self.logger.propagate = False
        self.show_progress = show_progress

    def evaluate(self, model, data_eval, context, epoch: int | None = None):
        features = data_eval["features"]
        if features.size(0) == 0:
            self.logger.warning("Evaluation split is empty; skipping metrics computation.")
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        preds, scores = model.predict_with_scores(features)

        y_true = data_eval["labels"].detach().cpu().numpy()
        y_pred = preds.detach().cpu().numpy()
        y_prob = scores.detach().cpu().numpy()

        ja, prauc, avg_p, avg_r, avg_f1 = multi_label_metric(y_true, y_pred, y_prob)
        ddi_adj = context.get_adj("ddi") if context else None
        ddi_rate = self._ddi_rate(data_eval["patient_counts"], y_pred, ddi_adj) if ddi_adj is not None else 0.0

        self.logger.info(
            "Evaluation metrics | JA=%.4f | PRAUC=%.4f | AvgP=%.4f | AvgR=%.4f | AvgF1=%.4f | DDI=%.4f",
            ja,
            prauc,
            avg_p,
            avg_r,
            avg_f1,
            ddi_rate,
        )

        self._save_predictions(y_pred, data_eval["patient_counts"])

        return ddi_rate, ja, prauc, avg_p, avg_r, avg_f1

    def _ddi_rate(self, patient_counts, y_pred, ddi_adj):
        idx = 0
        smm_record = []
        for count in patient_counts:
            patient_preds = []
            for _ in range(count):
                labels = np.nonzero(y_pred[idx])[0].tolist()
                patient_preds.append(labels)
                idx += 1
            smm_record.append(patient_preds)
        return ddi_rate_score(smm_record, ddi_adj)

    def _save_predictions(self, y_pred, patient_counts):
        idx = 0
        smm_record = []
        for count in patient_counts:
            patient_preds = []
            for _ in range(count):
                labels = np.nonzero(y_pred[idx])[0].tolist()
                patient_preds.append(labels)
                idx += 1
            smm_record.append(patient_preds)
        with open(self.save_dir / "knn_predictions.pkl", "wb") as f:
            dill.dump(smm_record, f)

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.core.interfaces.trainer import Trainer
from src.utils.metrics import multi_label_metric, ddi_rate_score


class MostPopularTrainer(Trainer):
    def __init__(self, save_dir: str | Path = "saved/MostPopular", log_level=logging.INFO):
        self.save_dir = Path(save_dir)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(handler)
            self.logger.propagate = False

    def train(self, model, train_data, val_data, context=None) -> Dict[str, float]:
        self.logger.info("Training MostPopular baseline on %d patients", len(train_data))
        model.fit(train_data)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.save_dir / "most_popular.pt"
        model.save(model_path)
        self.logger.info("Saved MostPopular baseline to %s", model_path)

        metrics = {}
        ddi_adj = context.get_adj("ddi") if context else None
        if val_data and ddi_adj is not None:
            metrics = self._evaluate_split(model, val_data, ddi_adj, context.vocab_sizes()[2])
        return metrics

    def _evaluate_split(self, model, patients: List, ddi_adj, med_vocab_size: int) -> Dict[str, float]:
        y_true = []
        y_pred = []
        smm_record = []

        for patient in patients:
            patient_preds = []
            for admission in patient:
                preds = model.predict(admission)
                patient_preds.append(preds)
                y_true.append(self._to_multihot(admission[2], med_vocab_size))
                y_pred.append(self._to_multihot(preds, med_vocab_size))
            smm_record.append(patient_preds)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        ja, avg_p, avg_r, avg_f1 = multi_label_metric(y_true, y_pred)
        ddi_rate = ddi_rate_score(smm_record, ddi_adj)

        return {"ja": ja, "avg_p": avg_p, "avg_r": avg_r, "avg_f1": avg_f1, "ddi_rate": ddi_rate}

    def _to_multihot(self, meds, size):
        vec = np.zeros(size)
        for m in meds:
            vec[m] = 1
        return vec

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import dill
import numpy as np

from src.core.interfaces.evaluator import Evaluator
from src.utils.metrics import multi_label_metric, ddi_rate_score


class MostPopularEvaluator(Evaluator):
    def __init__(self, save_dir: str | Path = "results/MostPopular", log_level=logging.INFO):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(handler)
            self.logger.propagate = False

    def evaluate(self, model, data_eval: List, context, **kwargs):
        med_vocab_size = context.vocab_sizes()[2]
        ddi_adj = context.get_adj("ddi")

        y_true = []
        y_pred = []
        smm_record = []

        for patient in data_eval:
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
        ddi_rate = ddi_rate_score(smm_record, ddi_adj) if ddi_adj is not None else 0.0

        self.logger.info(
            "MostPopular evaluation: DDI=%.4f JA=%.4f AvgP=%.4f AvgR=%.4f AvgF1=%.4f",
            ddi_rate,
            ja,
            avg_p,
            avg_r,
            avg_f1,
        )

        with open(self.save_dir / "most_popular_predictions.pkl", "wb") as f:
            dill.dump(smm_record, f)

        return ddi_rate, ja, avg_p, avg_r, avg_f1

    def _to_multihot(self, meds, size):
        vec = np.zeros(size)
        for m in meds:
            vec[m] = 1
        return vec

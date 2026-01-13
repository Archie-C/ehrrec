from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Sequence

import dill
import numpy as np
import torch

from src.core.interfaces.evaluator import Evaluator
from src.utils.metrics import ddi_rate_score, multi_label_metric


class FourSDrugEvaluator(Evaluator):
    def __init__(self, save_dir: str | Path = "results/4SDrug", log_level: int = logging.INFO):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)

            file_handler = logging.FileHandler(self.save_dir / "evaluation.log")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger.propagate = False

    def evaluate(self, model, data_eval, context, **kwargs) -> Sequence[float]:
        start_time = time.time()
        model.eval()
        device = next(model.parameters()).device
        n_drugs = getattr(model, "n_drug", context.metadata.get("n_drug"))
        if n_drugs is None:
            raise ValueError("Unable to determine number of drugs for evaluation.")

        y_true = []
        y_pred = []
        y_prob = []
        smm_record = []

        num_records = len(data_eval) if hasattr(data_eval, "__len__") else None
        if num_records == 0:
            self.logger.warning("Evaluation dataset is empty; returning zeros.")
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        sym_sequences = data_eval["sym"]
        drug_targets = data_eval["drug"]

        for syms_raw, meds_raw in zip(sym_sequences, drug_targets):
            syms = torch.tensor(syms_raw).to(device)
            meds = meds_raw
            scores = model.evaluate(syms, device=device)
            prob = torch.sigmoid(scores).detach().cpu().numpy()
            pred = (prob >= 0.5).astype(int)

            gt = np.zeros(n_drugs)
            gt[meds] = 1

            y_true.append(gt)
            y_pred.append(pred)
            y_prob.append(prob)
            smm_record.append([list(np.where(pred == 1)[0])])

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)
        ja, prauc, avg_p, avg_r, avg_f1 = multi_label_metric(y_true, y_pred, y_prob)
        ddi_adj = context.get_adj("ddi")
        ddi_rate = ddi_rate_score(smm_record, ddi_adj) if ddi_adj is not None else 0.0

        metrics = {
            "ddi_rate": float(ddi_rate),
            "ja": float(ja),
            "prauc": float(prauc),
            "avg_p": float(avg_p),
            "avg_r": float(avg_r),
            "avg_f1": float(avg_f1),
        }
        self.logger.info(
            "4SDrug evaluation | JA=%.4f | PRAUC=%.4f | AvgP=%.4f | AvgR=%.4f | AvgF1=%.4f | DDI=%.4f",
            ja,
            prauc,
            avg_p,
            avg_r,
            avg_f1,
            ddi_rate,
        )
        self.logger.debug("Evaluated %s admissions in %.2fs", num_records, time.time() - start_time)
        with open(self.save_dir / "predictions.pkl", "wb") as f:
            dill.dump(smm_record, f)
        self.logger.info("Saved prediction records to %s", self.save_dir / "predictions.pkl")

        order = ["ddi_rate", "ja", "prauc", "avg_p", "avg_r", "avg_f1"]
        return tuple(metrics[key] for key in order)

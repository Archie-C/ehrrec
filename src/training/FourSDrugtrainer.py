from __future__ import annotations

import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam

from src.core.interfaces.trainer import Trainer
from src.utils.metrics import ddi_rate_score, multi_label_metric


class FourSDrugTrainer(Trainer):
    """
    Trainer implementing the published 4SDrug optimization loop.

    Expects ``train_data`` to provide parallel sequences of symptom sets,
    medication multi-hot targets, and similar-set indices. Validation data
    is a list of admissions, each shaped like ``(symptoms, procedures, meds)``.
    """

    def __init__(
        self,
        epochs: int = 200,
        lr: float = 2e-4,
        alpha: float = 0.5,
        beta: float = 1.0,
        eval_every: int = 5,
        save_dir: Path | str = "saved/4SDrug",
        log_level: int = logging.INFO,
        show_progress: bool = True,
    ) -> None:
        self.epochs = epochs
        self.lr = lr
        self.alpha = alpha
        self.beta = beta
        self.eval_every = max(1, eval_every)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)

            file_handler = logging.FileHandler(self.save_dir / "training.log")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger.propagate = False

        self.show_progress = show_progress

    def train(self, model, train_data, val_data, context) -> Dict[str, Sequence[float]]:
        self.logger.info("Starting 4SDrug training")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        device = next(model.parameters()).device

        optimizer = Adam(model.parameters(), lr=self.lr)
        total_params = sum(np.prod(p.size()) for p in model.parameters())
        train_batches = len(train_data["sym"]) if train_data and "sym" in train_data else 0
        val_items = len(val_data) if val_data is not None and hasattr(val_data, "__len__") else 0

        self.logger.info(
            "Config | epochs=%d | eval_every=%d | lr=%.5f | alpha=%.3f | beta=%.3f",
            self.epochs,
            self.eval_every,
            self.lr,
            self.alpha,
            self.beta,
        )
        self.logger.info("Dataset summary | train_batches=%d | val_items=%d | parameters=%d", train_batches, val_items, total_params)

        best_ja = float("-inf")
        history: Dict[str, list] = defaultdict(list)

        for epoch in range(self.epochs):
            model.train()
            start = time.time()
            epoch_loss = self._train_epoch(model, train_data, optimizer, device)
            elapsed = time.time() - start

            metrics = {}
            if (epoch + 1) % self.eval_every == 0 and val_data:
                self.logger.debug("Running validation at epoch %d", epoch + 1)
                metrics = self._evaluate_dataset(model, val_data, context)
                history["ja"].append(metrics["ja"])
                history["prauc"].append(metrics["prauc"])
                history["avg_p"].append(metrics["avg_p"])
                history["avg_r"].append(metrics["avg_r"])
                history["avg_f1"].append(metrics["avg_f1"])
                history["ddi_rate"].append(metrics["ddi_rate"])
                self.logger.info(
                    "Epoch %d | loss=%.4f | JA=%.4f | PRAUC=%.4f | AvgP=%.4f | AvgR=%.4f | AvgF1=%.4f | DDI=%.4f | time=%.2fm",
                    epoch + 1,
                    epoch_loss,
                    metrics["ja"],
                    metrics["prauc"],
                    metrics["avg_p"],
                    metrics["avg_r"],
                    metrics["avg_f1"],
                    metrics["ddi_rate"],
                    elapsed / 60,
                )

                if metrics["ja"] > best_ja:
                    best_ja = metrics["ja"]
                    best_path = self.save_dir / "best_model.pt"
                    model.save(best_path)
                    self.logger.info("New best model saved at epoch %d (JA=%.4f)", epoch + 1, best_ja)
            else:
                self.logger.info("Epoch %d | loss=%.4f | time=%.2fm", epoch + 1, epoch_loss, elapsed / 60)

        self.logger.info("Training finished | best_JA=%.4f", best_ja if best_ja != float('-inf') else float('nan'))
        return history

    def _train_epoch(self, model, train_data, optimizer, device):
        losses = []
        sym_sequences = train_data["sym"]
        drug_targets = train_data["drug"]
        similar_indices = train_data["similar_idx"]

        iterator = zip(sym_sequences, drug_targets, similar_indices)
        if self.show_progress:
            iterator = enumerate(iterator)
        self.logger.debug("Training epoch batches=%d", len(sym_sequences))
        for item in iterator:
            if self.show_progress:
                _, batch = item
            else:
                batch = item
            syms_raw, drugs_raw, similar_idx_raw = batch
            syms = torch.tensor(syms_raw).to(device)
            drugs = torch.tensor(drugs_raw).to(device)
            similar_idx = torch.tensor(similar_idx_raw).to(device)
            model.zero_grad()
            optimizer.zero_grad()
            scores, bpr_loss, ddi_loss = model(syms, drugs, similar_idx)

            sig_scores = torch.sigmoid(scores)
            safe_sig_scores = torch.clamp(sig_scores, min=1e-8)
            bce_loss = F.binary_cross_entropy_with_logits(scores, drugs)
            entropy = -torch.mean(sig_scores * (torch.log(safe_sig_scores) - 1))
            loss = bce_loss + 0.5 * entropy + self.alpha * bpr_loss + self.beta * ddi_loss

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if not losses:
            self.logger.warning("No losses recorded this epoch; check training data.")
        return float(np.mean(losses)) if losses else 0.0

    def _evaluate_dataset(self, model, dataset, context):
        eval_start = time.time()
        model.eval()
        y_true = []
        y_pred = []
        y_prob = []
        smm_record = []

        n_drugs = getattr(model, "n_drug", context.metadata.get("n_drug"))
        if n_drugs is None:
            raise ValueError("Unable to determine number of drugs for evaluation.")
        
        sym_sequences = dataset["sym"]
        drug_targets = dataset["drug"]

        for syms_raw, meds_raw in zip(sym_sequences, drug_targets):

            # Convert symptoms to tensor of indices
            syms = torch.tensor(syms_raw, dtype=torch.long).to(model.device)

            if syms.dim() == 1:
                syms = syms.unsqueeze(0)
            # Forward pass
            scores = model.evaluate(syms)

            prob = torch.sigmoid(scores).detach().cpu().numpy()
            pred = (prob >= 0.5).astype(int)

            # Build ground truth vector
            gt = np.array(meds_raw)

            y_true.append(gt)
            y_pred.append(pred)
            y_prob.append(prob)

            smm_record.append([list(np.where(pred == 1)[0])])

        if not y_true:
            self.logger.warning("Validation set empty; skipping metrics computation.")
            return {"ja": 0.0, "prauc": 0.0, "avg_p": 0.0, "avg_r": 0.0, "avg_f1": 0.0, "ddi_rate": 0.0}

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)
        ja, prauc, avg_p, avg_r, avg_f1 = multi_label_metric(y_true, y_pred, y_prob)
        ddi_adj = context.get_adj("ddi")
        ddi_rate = ddi_rate_score(smm_record, ddi_adj) if ddi_adj is not None else 0.0
        self.logger.debug("Validation computed on %d admissions in %.2fs", len(y_true), time.time() - eval_start)

        return {
            "ja": float(ja),
            "prauc": float(prauc),
            "avg_p": float(avg_p),
            "avg_r": float(avg_r),
            "avg_f1": float(avg_f1),
            "ddi_rate": float(ddi_rate),
        }

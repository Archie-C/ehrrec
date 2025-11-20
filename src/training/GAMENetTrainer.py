from pathlib import Path
from src.core.interfaces.trainer import Trainer

import torch
import numpy as np
import logging
import time

from torch.optim import Adam
import torch.nn.functional as F
from collections import defaultdict

from src.utils.metrics import ddi_rate_score, multi_label_metric

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


class GAMENetTrainer(Trainer):
    """
    Clean, modular, fully logged GAMENet trainer.
    Mirrors the style of GAMENetPreprocessor.
    """

    def __init__(
        self,
        epochs: int,
        lr: float,
        ddi_loss: bool = False,
        target_ddi: float = 0.05,
        ddi_decay: float = 0.85,
        save_dir: Path | str = "saved/GAMENet/",
        log_level: int = logging.INFO,
        show_progress: bool = True
    ):
        self.epochs = epochs
        self.lr = lr
        self.ddi_loss = ddi_loss
        self.target_ddi = target_ddi
        self.ddi_decay = ddi_decay
        self.save_dir = Path(save_dir)

        # logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ))
            self.logger.addHandler(handler)
            self.logger.propagate = False

        # tqdm toggle
        self.show_progress = show_progress and (tqdm is not None)
        if show_progress and tqdm is None:
            self.logger.warning("tqdm not available; progress bars disabled.")

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------
    def train(self, model, train_data, val_data, context):
        self.logger.info("Starting GAMENet training")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        ddi_adj = context.get_adj("ddi")
        if ddi_adj is None:
            raise ValueError("DDI adjacency matrix not available in context.")

        optimizer = Adam(model.parameters(), lr=self.lr)
        voc_size = model.vocab_size
        device = next(model.parameters()).device

        history = defaultdict(list)
        best_ja = -1
        best_epoch = 0
        T = 0.5  # DDI annealing temperature

        for epoch in range(self.epochs):
            model.train()
            start_time = time.time()

            loss_epoch = self._train_epoch(
                model,
                train_data,
                optimizer,
                voc_size,
                device,
                epoch,
                T,
                ddi_adj
            )

            # update T
            T *= self.ddi_decay

            # evaluation
            metrics = self._evaluate(model, val_data, ddi_adj, voc_size, epoch)
            ddi, ja, prauc, avg_p, avg_r, avg_f1 = metrics

            # log metrics
            history["ja"].append(ja)
            history["ddi_rate"].append(ddi)
            history["prauc"].append(prauc)
            history["avg_p"].append(avg_p)
            history["avg_r"].append(avg_r)
            history["avg_f1"].append(avg_f1)

            elapsed = (time.time() - start_time) / 60
            self.logger.info(
                f"Epoch {epoch} | loss={loss_epoch:.4f} | JA={ja:.4f} | "
                f"DDI={ddi:.4f} | time={elapsed:.2f} min"
            )

            # save checkpoint
            ckpt_path = self.save_dir / f"Epoch_{epoch}_JA_{ja:.4f}_DDI_{ddi:.4f}.pt"
            extra = getattr(model, "get_config", lambda: None)()
            model.save(ckpt_path, extra=extra)
            self.logger.info(f"Saved checkpoint: {ckpt_path}")

            # best model tracking
            if ja > best_ja:
                best_ja = ja
                best_epoch = epoch
                best_path = self.save_dir / "best_model.pt"
                model.save(best_path, extra=extra)
                self.logger.info(f"New best model saved (Epoch {epoch})")

        self.logger.info(f"Training complete. Best epoch = {best_epoch}")
        return history

    # ------------------------------------------------------------------
    # TRAINING UTILITIES
    # ------------------------------------------------------------------
    def _train_epoch(self, model, train_data, optimizer, voc_size, device, epoch, T, ddi_adj):
        """One full training epoch."""
        loss_record = []
        prediction_cnt = 0
        neg_cnt = 0

        iterator = enumerate(train_data)
        if self.show_progress:
            iterator = tqdm(iterator, total=len(train_data), desc=f"Train epoch {epoch}")

        for step, patient in iterator:
            for adm_idx, adm in enumerate(patient):
                seq_input = patient[:adm_idx + 1]

                y, y_margin = self._prepare_targets(adm, voc_size, device)

                out, batch_neg = model(seq_input)

                loss1 = F.binary_cross_entropy_with_logits(out, y)
                loss3 = F.multilabel_margin_loss(torch.sigmoid(out), y_margin)

                if self.ddi_loss:
                    loss, pc_inc, nc_inc = self._apply_ddi_loss(
                        out, batch_neg, loss1, loss3, T, ddi_adj
                    )
                    prediction_cnt += pc_inc
                    neg_cnt += nc_inc
                else:
                    loss = 0.9 * loss1 + 0.01 * loss3
                    prediction_cnt += 1

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

                loss_record.append(loss.item())

            if self.show_progress:
                iterator.set_postfix({"loss": np.mean(loss_record)})

        self.logger.debug(
            f"Epoch {epoch}: L_p={prediction_cnt}, L_neg={neg_cnt}"
        )

        return np.mean(loss_record)

    def _prepare_targets(self, adm, voc_size, device):
        """Prepare BCE and margin loss targets."""
        y = np.zeros((1, voc_size[2]))
        y[:, adm[2]] = 1
        y = torch.FloatTensor(y).to(device)

        y_margin = np.full((1, voc_size[2]), -1)
        for i, item in enumerate(adm[2]):
            y_margin[0][i] = item
        y_margin = torch.LongTensor(y_margin).to(device)

        return y, y_margin

    def _apply_ddi_loss(self, out, batch_neg, loss1, loss3, T, ddi_adj):
        """Implements the original DDI-based loss switching logic."""
        pred = torch.sigmoid(out).detach().cpu().numpy()[0]
        pred = (pred >= 0.5).astype(int)
        labels = np.where(pred == 1)[0]

        ddi_now = ddi_rate_score([[labels]], ddi_adj)

        if ddi_now <= self.target_ddi:
            return 0.9 * loss1 + 0.01 * loss3, 1, 0
        else:
            rnd = np.exp((self.target_ddi - ddi_now) / T)
            if np.random.rand(1) < rnd:
                return batch_neg, 0, 1
            else:
                return 0.9 * loss1 + 0.01 * loss3, 1, 0

    # ------------------------------------------------------------------
    # EVALUATION
    # ------------------------------------------------------------------
    def _evaluate(self, model, data_eval, ddi_adj, voc_size, epoch):
        model.eval()

        smm_record = []
        ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]

        # ---------- ABSOLUTELY CLEAN TQDM ------------
        if self.show_progress:
            iterator = tqdm(
                range(len(data_eval)),
                desc=f"Eval epoch {epoch}",
                leave=False,
                dynamic_ncols=True,
            )
        else:
            iterator = range(len(data_eval))

        # ---------- LOOP -----------------------------
        for step in iterator:
            patient = data_eval[step]     # <— IMPORTANT: remove enumerate()

            y_gt = []
            y_pred = []
            y_prob = []
            y_label = []

            for adm_idx, adm in enumerate(patient):
                out = model(patient[:adm_idx+1])
                out = torch.sigmoid(out).detach().cpu().numpy()[0]

                gt = np.zeros(voc_size[2])
                gt[adm[2]] = 1

                pred = (out >= 0.5).astype(int)
                labels = np.where(pred == 1)[0]

                y_gt.append(gt)
                y_pred.append(pred)
                y_prob.append(out)
                y_label.append(labels.tolist())

            smm_record.append(y_label)

            m = multi_label_metric(np.array(y_gt), np.array(y_pred), np.array(y_prob))
            ja_i, prauc_i, avg_p_i, avg_r_i, avg_f1_i = m

            ja.append(ja_i)
            prauc.append(prauc_i)
            avg_p.append(avg_p_i)
            avg_r.append(avg_r_i)
            avg_f1.append(avg_f1_i)

        ddi = ddi_rate_score(smm_record, ddi_adj)

        return (
            ddi,
            np.mean(ja),
            np.mean(prauc),
            np.mean(avg_p),
            np.mean(avg_r),
            np.mean(avg_f1),
        )

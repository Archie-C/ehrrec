import torch
import os
import numpy as np
import dill
import logging
from collections import defaultdict

from src.core.interfaces.evaluator import Evaluator
from src.utils.metrics import ddi_rate_score, multi_label_metric

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


class GAMENetEvaluator(Evaluator):
    """
    Evaluator with logging identical to GAMENetTrainer.
    """

    def __init__(self, save_dir: str, log_level=logging.INFO, show_progress=True):
        self.save_dir = save_dir

        # -----------------------
        # Logger (identical style)
        # -----------------------
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
    # MAIN EVALUATION
    # ------------------------------------------------------------------
    def evaluate(self, model, data_eval, context, epoch: int = None):
        self.logger.info(f"Starting evaluation for epoch {epoch}")
        ddi_adj = context.get_adj("ddi")
        if ddi_adj is None:
            raise ValueError("DDI adjacency matrix not available in context.")
        voc_size = model.vocab_size
        model.eval()
        smm_record = []
        ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
        case_study = defaultdict(dict)

        # Clean tqdm
        if self.show_progress:
            iterator = tqdm(
                range(len(data_eval)),
                desc=f"Eval epoch {epoch}",
                leave=False,
                dynamic_ncols=True,
            )
        else:
            iterator = range(len(data_eval))

        # --------------------------------------------------------------
        # Evaluation loop
        # --------------------------------------------------------------
        for step in iterator:
            patient = data_eval[step]

            y_gt = []
            y_pred = []
            y_prob = []
            y_label = []

            # Infer each admission
            for adm_idx, adm in enumerate(patient):
                out = model(patient[:adm_idx + 1])
                out = torch.sigmoid(out).detach().cpu().numpy()[0]

                # ground truth
                gt = np.zeros(voc_size[2])
                gt[adm[2]] = 1
                y_gt.append(gt)

                # prediction
                prob = out
                pred = (prob >= 0.5).astype(int)
                labels = np.where(pred == 1)[0]

                y_prob.append(prob)
                y_pred.append(pred)
                y_label.append(sorted(labels))

            smm_record.append(y_label)

            m = multi_label_metric(
                np.array(y_gt), np.array(y_pred), np.array(y_prob)
            )
            adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = m

            # save case study entry
            case_study[adm_ja] = {
                'ja': adm_ja,
                'patient': patient,
                'y_label': y_label
            }

            ja.append(adm_ja)
            prauc.append(adm_prauc)
            avg_p.append(adm_avg_p)
            avg_r.append(adm_avg_r)
            avg_f1.append(adm_avg_f1)

        # final metrics
        ddi_rate = ddi_rate_score(smm_record, ddi_adj)

        self.logger.info(
            f"Epoch {epoch} Evaluation Complete | "
            f"DDI={ddi_rate:.4f} | JA={np.mean(ja):.4f} | PRAUC={np.mean(prauc):.4f} | "
            f"AvgP={np.mean(avg_p):.4f} | AvgR={np.mean(avg_r):.4f} | AvgF1={np.mean(avg_f1):.4f}"
        )

        # save results
        os.makedirs(self.save_dir, exist_ok=True)
        dill.dump(smm_record, open(os.path.join(self.save_dir, "predicted_records.pkl"), "wb"))
        dill.dump(case_study, open(os.path.join(self.save_dir, "case_study.pkl"), "wb"))

        self.logger.info(f"Saved predicted_records.pkl and case_study.pkl to {self.save_dir}")

        return (
            ddi_rate,
            np.mean(ja),
            np.mean(prauc),
            np.mean(avg_p),
            np.mean(avg_r),
            np.mean(avg_f1),
        )

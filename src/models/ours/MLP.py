import time
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any
from src.models.BaseModel import BaseModel
from src.utils.logging import get_logger

logger = get_logger("MLP")

@dataclass
class MLPTrainingConfig:
    seed: int = 42
    lr: float = 2e-4
    save_dir: str = "saved/MLP"
    ckpt_name: str = "best_model" + str(time.time())
    batch_size: int = 128
    log_every: int = 50
    epochs: int = 40
    max_grad_norm: int = 5
    weight_decay: float = 1e-5


class MLPModel(BaseModel, nn.Module):
    def __init__(
        self, 
        vocab_size, 
        device,
        num_layers: int = 2, 
        hidden_size: int = 64, 
        dropout: float = 0.4, 
        emb_dim: int = 128,
        cfg: MLPTrainingConfig = MLPTrainingConfig()
    ):
        super().__init__()
        nn.Module.__init__(self)
        n_diag, n_proc, n_out = vocab_size
        self.n_out = n_out
        self.n_diag = n_diag
        self.n_proc = n_proc

        self.diag_proj = nn.Linear(n_diag, emb_dim, bias=False)
        self.proc_proj = nn.Linear(n_proc, emb_dim, bias=False)
        self.norm = nn.LayerNorm(2 * emb_dim)
        self.mlp = self._build_mlp(2 * emb_dim, vocab_size, num_layers, hidden_size, dropout)
        self.cfg = cfg
        self.device = device
        
    def forward(self, X):
        diag_x = X[:, :self.n_diag]
        proc_x = X[:, self.n_diag:self.n_diag + self.n_proc]
        diag_emb = self.diag_proj(diag_x)
        proc_emb = self.proc_proj(proc_x)
        diag_emb = diag_emb / diag_x.sum(dim=1, keepdim=True).clamp_min(1.0)
        proc_emb = proc_emb / proc_x.sum(dim=1, keepdim=True).clamp_min(1.0)
        x = torch.cat([diag_emb, proc_emb], dim=1)
        x = self.norm(x)
        return self.mlp(x)    
    def fit(self, train_loader, val_loader):
        training_start_time = time.time()
        
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        logger.info(f"Set random seed to {self.cfg.seed}")
        self.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        
        
        logger.info(f"Initialized optimizer with lr={self.cfg.lr}")
        
        history: Dict[str, Any] = {
            "train_loss": [],
            "val_metrics": [],
            "best_epoch": None,
            "best_score": None,
            "best_ckpt_path": None,
            "training_time": None,
        }
        global_step = 0
        
        threshold = getattr(self.cfg, "threshold", 0.2)
        ignore_ids = getattr(self.cfg, "ignore_ids", [0, 1])
        pos_weight = self.compute_pos_weight_from_loader(
            train_loader,
            num_labels=self.n_out-2,
            ignore_ids=ignore_ids,
            device=self.device,
            cap=3.0
        )
        
        bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        save_dir = Path(self.cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = save_dir / self.cfg.ckpt_name
        logger.info(f"Checkpoints will be saved to: {ckpt_path}")

        best_score = -float("inf")
        best_epoch = -1
        best_state = None
        
        logger.info(f"Starting training for {self.cfg.epochs} epochs on {len(train_loader) * self.cfg.batch_size} samples, on device {self.device}")
        for epoch in range(1, self.cfg.epochs + 1):
            epoch_start_time = time.time()
            self.train()

            epoch_loss_sum = 0.0
            num_batches = 0
            
            for batch in train_loader:
                
                X, y = batch
                
                mask = torch.ones(y.size(1), dtype=torch.bool, device=y.device)
                mask[ignore_ids] = False
                
                preds = self(X)
                
                y_masked = y[:, mask]
                pred_masked = preds[:, mask]
                
                bce_loss = bce_loss_fn(pred_masked, y_masked)
                optimizer.zero_grad(set_to_none=True)
                bce_loss.backward()
                
                if self.cfg.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.cfg.max_grad_norm)
                
                optimizer.step()

                # Accumulate metrics
                epoch_loss_sum += float(bce_loss.item())
                num_batches += 1
                global_step += 1
                
                if self.cfg.log_every and global_step % self.cfg.log_every == 0:
                    logger.debug(
                        f"[Epoch {epoch}/{self.cfg.epochs}] Step {global_step} | "
                        f"(BCE={bce_loss.item():.4f},"
                    )
            avg_train_loss = epoch_loss_sum / max(num_batches, 1)
            epoch_time = time.time() - epoch_start_time
            history["train_loss"].append(avg_train_loss)
            
            logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s | Train Loss={avg_train_loss:.4f}")
            
            if val_loader is not None:
                val_start_time = time.time()
                self.eval()
                mask = torch.ones(y.size(1), dtype=torch.bool, device=y.device)
                mask[ignore_ids] = False

                jaccard_scores = []
                prec_scores = []
                rec_scores = []
                f1_scores = []

                with torch.no_grad():
                    for X, y in val_loader:
                        X = X.to(self.device)
                        y = y.to(self.device)

                        logits = self(X)                 # (B, C)
                        logits = logits[:, mask]
                        y_val = y[:, mask].bool()
                        
                        probs = torch.sigmoid(logits)          # (B, C)
                        k = 11  # replace with your avg meds
                        topk = probs.topk(k, dim=1).indices    # (B, k)
                        pred = torch.zeros_like(probs, dtype=torch.bool)
                        pred.scatter_(1, topk, True)

                        # probs = torch.sigmoid(logits)    # (B, C)
                        # pred = probs >= threshold        # (B, C) bool

                        TP = (pred & y_val).sum(dim=1)
                        FP = (pred & ~y_val).sum(dim=1)
                        FN = (~pred & y_val).sum(dim=1)

                        precision = TP / (TP + FP + 1e-8)
                        recall    = TP / (TP + FN + 1e-8)
                        f1        = 2 * precision * recall / (precision + recall + 1e-8)

                        union = (pred | y_val).sum(dim=1)
                        jacc  = TP / (union + 1e-8)

                        prec_scores.append(precision.mean().item())
                        rec_scores.append(recall.mean().item())
                        f1_scores.append(f1.mean().item())
                        jaccard_scores.append(jacc.mean().item())

                avg_precision = sum(prec_scores) / len(prec_scores)
                avg_recall    = sum(rec_scores) / len(rec_scores)
                avg_f1        = sum(f1_scores) / len(f1_scores)
                avg_jaccard   = sum(jaccard_scores) / len(jaccard_scores)

                val_time = time.time() - val_start_time

                logger.info(
                    f"Epoch {epoch} Validation ({val_time:.2f}s) | "
                    f"Jaccard={avg_jaccard:.4f} | "
                    f"Precision={avg_precision:.4f} | "
                    f"Recall={avg_recall:.4f} | "
                    f"F1={avg_f1:.4f}"
                )
                    
        total_training_time = time.time() - training_start_time
        history["training_time"] = total_training_time
        
        logger.info(f"Training completed in {total_training_time:.2f}s ({total_training_time/60:.2f} minutes)")

        return history
    
    def predict(self, X):
        return super().predict(X)
    
    def _build_mlp(self, in_dim, vocab_size, num_layers: int, hidden_size: int = 256, dropout: float = 0.1):
        in_size = in_dim
        out_size = vocab_size[2]
        
        assert num_layers >= 1
        
        layers = []

        if num_layers == 1:
            layers.append(nn.Linear(in_size, out_size))
            return nn.Sequential(*layers)
        
        layers += [nn.Linear(in_size, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(), nn.Dropout(dropout)]

        # middle layers: hidden -> hidden
        for i in range(num_layers - 2):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)]

        # last layer: hidden -> out
        layers.append(nn.Linear(hidden_size, out_size))

        return nn.Sequential(*layers)
    
    def compute_pos_weight_from_loader(self, train_loader, num_labels: int, ignore_ids=(0, 1), device="cpu", cap=50.0):
        pos = torch.zeros(num_labels, dtype=torch.float64)
        n_samples = 0

        for _, y in train_loader:
            # y: (B, C)
            y = y.detach()[:, 2:]
            n_samples += y.size(0)
            pos += y.sum(dim=0).to(torch.float64)

        neg = n_samples - pos
        pos_weight = (neg / (pos + 1e-8)).to(torch.float32)

        # ignore PAD/UNK output columns
        if ignore_ids is not None:
            pos_weight[list(ignore_ids)] = 0.0  # won't be used if you mask anyway

        # cap for stability
        pos_weight = pos_weight.clamp(max=cap)

        return pos_weight.to(device)
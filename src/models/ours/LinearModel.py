from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.BaseModel import BaseModel

@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 0.0
    use_pos_weight: bool = True
    clip_grad_norm: Optional[float] = 1.0
    threshold: float = 0.5
    early_stop_patience: Optional[int] = None
    val_metric: Literal["loss", "jaccard"] = "loss"

class LinearModel(BaseModel, nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        cfg: Optional[TrainConfig] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        nn.Module.__init__(self)   # must be first for buffers/params
        BaseModel.__init__(self)

        self.name = "linear multi-label"
        self.linear = nn.Linear(in_dim, out_dim)
        self.cfg = cfg or TrainConfig()
        self.threshold = float(self.cfg.threshold)
        self.dtype = dtype
        self._trained = False

        if device is not None:
            self.to(device)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.linear(X.to(dtype=self.dtype))  # logits [B,M]

    @staticmethod
    def _pos_weight(y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        y = y.float()
        pos = y.sum(dim=0)
        neg = y.size(0) - pos
        return (neg / (pos + eps)).clamp(min=1.0)

    @staticmethod
    def _jaccard(pred01: torch.Tensor, y01: torch.Tensor) -> torch.Tensor:
        pred = pred01.bool()
        y = y01.bool()
        inter = (pred & y).sum(dim=1).float()
        union = (pred | y).sum(dim=1).float().clamp_min(1.0)
        return (inter / union).mean()

    def fit(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        cfg = self.cfg
        device = next(self.parameters()).device

        X_train = X_train.to(device)
        y_train = y_train.to(device)

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val = X_val.to(device)
            y_val = y_val.to(device)

        self.threshold = float(cfg.threshold)

        pos_weight = self._pos_weight(y_train).to(device) if cfg.use_pos_weight else None
        opt = torch.optim.AdamW(self.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        n = X_train.size(0)
        history: Dict[str, list] = {"train_loss": []}
        if has_val:
            history["val_loss"] = []
            history["val_jaccard"] = []

        best_score = None
        best_state = None
        patience_left = cfg.early_stop_patience

        for epoch in range(cfg.epochs):
            # training mode (do NOT call self.train(True) here or you'll recurse)
            nn.Module.train(self, True)

            perm = torch.randperm(n, device=device)
            total = 0.0

            for i in range(0, n, cfg.batch_size):
                idx = perm[i : i + cfg.batch_size]
                xb = X_train[idx]
                yb = y_train[idx].float()

                opt.zero_grad(set_to_none=True)
                logits = self(xb)
                loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight)
                loss.backward()

                if cfg.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), cfg.clip_grad_norm)

                opt.step()
                total += loss.item() * xb.size(0)

            history["train_loss"].append(total / n)

            if has_val:
                nn.Module.eval(self)
                with torch.no_grad():
                    v_logits = self(X_val)
                    v_loss = F.binary_cross_entropy_with_logits(
                        v_logits, y_val.float(), pos_weight=pos_weight
                    ).item()
                    v_prob = torch.sigmoid(v_logits)
                    v_pred = (v_prob >= self.threshold).to(torch.uint8)
                    v_j = float(self._jaccard(v_pred, y_val))

                history["val_loss"].append(v_loss)
                history["val_jaccard"].append(v_j)

                # early stopping
                score = v_loss if cfg.val_metric == "loss" else (-v_j)
                if best_score is None or score < best_score:
                    best_score = score
                    best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
                    if patience_left is not None:
                        patience_left = cfg.early_stop_patience
                elif patience_left is not None:
                    patience_left -= 1
                    if patience_left <= 0:
                        break

        if best_state is not None:
            self.load_state_dict(best_state)

        self._trained = True
        return history

    @torch.no_grad()
    def predict(self, X: torch.Tensor):
        nn.Module.eval(self)
        device = next(self.parameters()).device
        X = X.to(device)
        logits = self(X)
        prob = torch.sigmoid(logits)
        pred = (prob >= self.threshold).to(torch.uint8)
        return pred, prob
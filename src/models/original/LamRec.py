import sys
import copy
import math
import time
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from typing import Any, Dict
from dataclasses import dataclass
from torch.utils.data import DataLoader

from src.utils.logging import get_logger
from src.models.BaseModel import BaseModel

logger = get_logger("LamRec Original Model")

@dataclass
class LamRecOriginalConfig:
    seed: int = 42
    
    embedding_dim: int = 256
    # transformer encoder
    n_heads: int = 4
    n_layers: int = 2
    # Multi-view Constrastive Loss
    temperature: float = 0.5
    
    lr: float = 5e-4
    weight_decay: float = 1e-5
    mcvl_lambda: float = 0.1
    epochs: int = 50
    max_grad_norm: float = 5.0
    log_every: int = 200
    
    save_dir: str = "saved/lamrec_original"
    ckpt_name: str = "best_model.pt"

class LabelAttention(nn.Module):
    def __init__(self, input_size: int, projection_size: int, num_classes: int):
        super().__init__()
        self.first_linear = nn.Linear(input_size, projection_size, bias=False)
        self.second_linear = nn.Linear(projection_size, num_classes, bias=False)
        self.third_linear = nn.Linear(input_size, num_classes)
        self._init_weights(mean=0.0, std=0.03)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """LAAT attention mechanism

        Args:
            x (torch.Tensor): [batch_size, seq_len, input_size]

        Returns:
            torch.Tensor: [batch_size, num_classes]
        """
        weights = torch.tanh(self.first_linear(x))  # [batch_size, seq_len, projection_size]
        att_weights = self.second_linear(weights)  # [batch_size, seq_len, num_classes]
        att_weights = torch.nn.functional.softmax(att_weights, dim=1).transpose(1,
                                                                                2)  # [batch_size,num_classes, seq_len]
        weighted_output = att_weights @ x  # [batch_size,num_classes, input_size]
        return (
            self.third_linear.weight.mul(weighted_output)
            .sum(dim=2)
            .add(self.third_linear.bias)
        )

    def _init_weights(self, mean: float = 0.0, std: float = 0.03) -> None:
        """
        Initialise the weights

        Args:
            mean (float, optional): Mean of the normal distribution. Defaults to 0.0.
            std (float, optional): Standard deviation of the normal distribution. Defaults to 0.03.
        """

        torch.nn.init.normal_(self.first_linear.weight, mean, std)
        torch.nn.init.normal_(self.second_linear.weight, mean, std)
        torch.nn.init.normal_(self.third_linear.weight, mean, std)

class MultiViewContrastiveLoss(nn.Module):
    def __init__(self, temperature=10):
        super(MultiViewContrastiveLoss, self).__init__()
        self.temperature = temperature

    def compute_joint(self, x_out, x_tf_out):
        # produces variable that requires grad (since args require grad)

        bn, k = x_out.size()
        assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)

        p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
        p_i_j = p_i_j.sum(dim=0)  # k, k
        p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
        p_i_j = p_i_j / p_i_j.sum()  # normalise

        return p_i_j

    def forward(self, x_out, x_tf_out, EPS=sys.float_info.epsilon):
        """Contrastive loss for maximizng the consistency"""
        if len(x_out.size()) == 3:
            x_out = x_out.mean(dim=1)  # (batch_size, hidden_dim)
            x_tf_out = x_tf_out.mean(dim=1)  # (batch_size, hidden_dim)

        x_out, x_tf_out = F.softmax(x_out, dim=-1), F.softmax(x_tf_out, dim=-1)
        _, k = x_out.size()

        p_i_j = self.compute_joint(x_out, x_tf_out)
        assert (p_i_j.size() == (k, k))

        p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
        p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)

        p_i_j = torch.where(p_i_j < EPS, torch.tensor([EPS], device=p_i_j.device), p_i_j)
        p_j = torch.where(p_j < EPS, torch.tensor([EPS], device=p_j.device), p_j)
        p_i = torch.where(p_i < EPS, torch.tensor([EPS], device=p_i.device), p_i)

        loss = - p_i_j * (torch.log(p_i_j) \
                          - self.temperature * torch.log(p_j) \
                          - self.temperature * torch.log(p_i))

        loss = loss.sum()

        return loss

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Project and reshape query, key, and value
        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # Broadcast mask to (batch_size, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))  # Fill masked positions with -inf

        # Compute attention probabilities
        attn_probs = nn.functional.softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Compute attended values
        attn_output = torch.matmul(attn_probs, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Project attended values
        output = self.out_proj(attn_output)

        return output

class CrossAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query, key, value, mask=None):
        attn_output = self.multihead_attn(query, key, value, mask=mask)
        output = query + self.dropout(attn_output)
        output = self.norm(output)
        return output

class TransformerCrossAttn(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super(TransformerCrossAttn, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        self.cross_attn_layers = nn.ModuleList([CrossAttention(d_model, nhead, dropout) for _ in range(num_layers)])
        self.feed_forward_layers = nn.ModuleList([nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        ) for _ in range(num_layers)])
        self.norm_layers = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

    def forward(self, x1, x2, mask=None):
        x1_pos = self.pos_encoder(x1)
        x2_pos = self.pos_encoder(x2)

        for i in range(len(self.cross_attn_layers)):
            # x1 attend to x2
            x1_pos = self.cross_attn_layers[i](query=x2_pos, key=x1_pos, value=x1_pos, mask=mask)
            x1_pos = x1_pos + self.feed_forward_layers[i](x1_pos)
            x1_pos = self.norm_layers[i](x1_pos)

            # x2 attend to x1
            x2_pos = self.cross_attn_layers[i](query=x1_pos, key=x2_pos, value=x2_pos, mask=mask)
            x2_pos = x2_pos + self.feed_forward_layers[i](x2_pos)
            x2_pos = self.norm_layers[i](x2_pos)

        return x1_pos, x2_pos

class LamRecOriginal(BaseModel, nn.Module):
    def __init__(
        self, 
        vocab_size, 
        ddi_adj, 
        device, 
        cfg: LamRecOriginalConfig = LamRecOriginalConfig()
    ):
        super().__init__()
        nn.Module.__init__(self)
        
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.ddi_adj = ddi_adj
        self.device = device
        
        self.diag_embeddings = nn.Embedding(vocab_size[0], cfg.embedding_dim, padding_idx=0)
        self.proc_embeddings = nn.Embedding(vocab_size[1], cfg.embedding_dim, padding_idx=0)
        
        self.seq_encoder = TransformerCrossAttn(
            d_model=cfg.embedding_dim,
            nhead=cfg.n_heads,
            num_layers=cfg.n_layers,
            dim_feedforward=cfg.embedding_dim,
        )
        self.mutli_view_cl = MultiViewContrastiveLoss(temperature=cfg.temperature)
        self.label_wise_attention = LabelAttention(
            cfg.embedding_dim * 2, 
            cfg.embedding_dim,
            vocab_size[2]
        )
    
    def forward(self, x):
        diag_history, proc_history, _ = x
        diag_history = diag_history.long()
        proc_history = proc_history.long()
        
        diags = self.diag_embeddings(diag_history)
        diags = torch.sum(diags, dim=2)
        mask = torch.any(diag_history != 0, dim=2)
        
        procs = self.proc_embeddings(proc_history)
        procs = torch.sum(procs, dim=2)
        
        diag_out, proc_out = self.seq_encoder(diags, procs, mask)
        
        mvcl = self.mutli_view_cl(diag_out, proc_out)
        logits = self.label_wise_attention(torch.cat((diag_out, proc_out), dim=-1))
        
        return {
            "logits": logits,
            "mvcl": mvcl
        }
    
    def fit(self, train_loader: DataLoader, val_loader:DataLoader=None):
        training_start_time = time.time()
        batch_size = train_loader.batch_size
        
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        logger.info(f"Set random seed to {self.cfg.seed}")
        
        self.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        bce_loss_fn = nn.BCEWithLogitsLoss()
        
        logger.info(f"Initialized optimizer with lr={self.cfg.lr}, weight_decay={self.cfg.weight_decay}")
        
        history: Dict[str, Any] = {
            "train_loss": [],
            "val_metrics": [],
            "best_epoch": None,
            "best_score": None,
            "best_ckpt_path": None,
            "training_time": None,
        }
        global_step = 0

        # Get evaluation settings
        threshold = getattr(self.cfg, "threshold", 0.1)
        ignore_ids = getattr(self.cfg, "ignore_ids", [0, 1])
        
        def compute_val_score(metrics: Dict[str, float]) -> float:
            """
            Compute validation score for model selection.
            Score = Jaccard - 0.1 * DDI_rate (higher is better)
            """
            return float(metrics["jaccard"] - 0.1 * metrics["ddi_rate_pred"])

        save_dir = Path(self.cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = save_dir / self.cfg.ckpt_name
        logger.info(f"Checkpoints will be saved to: {ckpt_path}")

        best_score = -float("inf")
        best_epoch = -1
        best_state = None
        
        logger.info(f"Starting training for {self.cfg.epochs} epochs on {len(train_loader) * batch_size} samples, on device {self.device}")
        
        for epoch in range(1, self.cfg.epochs + 1):
            epoch_start_time = time.time()
            self.train()

            epoch_loss_sum = 0.0
            num_batches = 0
            
            for batch in train_loader:
                out = self(batch)
                _, _, meds_history = batch
                target = meds_history[:, -1:, :]
                target = target.squeeze(1)
                medication_logits = out["logits"]
                bce_loss = bce_loss_fn(medication_logits, target)
                total_loss = 0.9 * bce_loss + self.cfg.mcvl_lambda * out["mvcl"]
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                
                if self.cfg.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.cfg.max_grad_norm)
                
                optimizer.step()

                # Accumulate metrics
                epoch_loss_sum += float(total_loss.item())
                num_batches += 1
                global_step += 1
                
                if self.cfg.log_every and global_step % self.cfg.log_every == 0:
                    logger.debug(
                        f"[Epoch {epoch}/{self.cfg.epochs}] Step {global_step} | "
                        f"Loss={total_loss.item():.4f} (BCE={bce_loss.item():.4f}, "
                        f"MVCL={out['mvcl'].item():.4f})"
                    )
            avg_train_loss = epoch_loss_sum / max(num_batches, 1)
            epoch_time = time.time() - epoch_start_time
            history["train_loss"].append(avg_train_loss)
            
            logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s | Train Loss={avg_train_loss:.4f}")
            
            if val_loader is not None:
                val_start_time = time.time()
                self.eval()
                jaccard = []
                batch_precisions = []
                batch_recalls = []
                batch_f1_scores = []
                for batch in val_loader:
                    out = self(batch)
                    medication_logits = out["logits"]
                    _, _, meds_history = batch
                    target = meds_history[:, -1:, :]
                    target = target.squeeze(1)
                    medication_probs = torch.sigmoid(medication_logits).squeeze(1)
                    # print(f"Medication_probs (first item): {medication_probs[0]}")
                    predicted_indices = (medication_probs >= threshold)
                    intersection = torch.sum(predicted_indices * target, dim=1)
                    union = torch.sum((predicted_indices + target) > 0, dim=1)
                    jaccard_index = intersection.float() / union.float()
                    jaccard.append(jaccard_index.mean())
                    
                    TP = torch.sum(predicted_indices * target)  # True positives
                    FP = torch.sum(predicted_indices * (1 - target))  # False positives
                    FN = torch.sum((~predicted_indices) * target) 
                    
                    precision = TP.float() / (TP + FP).float() if TP + FP > 0 else torch.tensor(0.0)
                    recall = TP.float() / (TP + FN).float() if TP + FN > 0 else torch.tensor(0.0)
                    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else torch.tensor(0.0)

                    # Store metrics for averaging later
                    batch_precisions.append(precision)
                    batch_recalls.append(recall)
                    batch_f1_scores.append(f1)
                
                avg_precision = torch.mean(torch.tensor(batch_precisions))
                avg_recall = torch.mean(torch.tensor(batch_recalls))
                avg_f1 = torch.mean(torch.tensor(batch_f1_scores))

                # Average Jaccard index
                jaccard = torch.mean(torch.tensor(jaccard))

                val_time = time.time() - val_start_time

                logger.info(
                    f"Epoch {epoch} Validation ({val_time:.2f}s) | "
                    f"Jaccard={jaccard:.4f} | "
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
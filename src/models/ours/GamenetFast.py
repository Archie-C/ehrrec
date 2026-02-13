from dataclasses import dataclass
import math
import time
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter

from src.models.BaseModel import BaseModel
from src.utils.logging import get_logger
from src.utils.metrics import evaluate_multilabel_sets

logger = get_logger("Fast Gamenet")

@dataclass
class GameNetTrainConfig:
    epochs: int = 40
    lr: float = 5e-4
    weight_decay: float = 1e-4
    ddi_lambda: float = 0.1
    max_grad_norm: float = 5.0
    log_every: int = 40
    seed: int = 42
    batch_size: int = 8
    ct_path: str = "saved/GAMENETOG/"
    
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device='cpu'):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        
        # Learnable weight matrix for feature transformation
        self.weight = Parameter(torch.FloatTensor(in_features, out_features).to(device))
        
        # Optional learnable bias vector
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features).to(device))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """
        Forward pass of the graph convolution layer for batched inputs using sparse matrices.
        
        Args:
            input: Node feature matrix of shape (batch_size, num_nodes, in_features).
            adj: Sparse adjacency matrix of shape (num_nodes, num_nodes).
        
        Returns:
            Transformed node features of shape (batch_size, num_nodes, out_features).
        """
        # Transform input features: X * W
        support = torch.mm(input, self.weight)
        
        # Aggregate features from neighbors: A * (X * W)
        output = torch.sparse.mm(adj, support)
        
        # Add bias if present
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, adj, device='cpu'):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        # # Add self-loops and normalize adjacency matrix
        # adj = self.normalize(torch.eye(adj.shape[0]).to(device) + adj)
        
        # # Convert adjacency matrix to a sparse tensor
        # adj_sparse = torch.sparse.FloatTensor(torch.LongTensor(adj.nonzero()), 
        #                                       torch.FloatTensor(adj[adj != 0]), 
        #                                       torch.Size(adj.shape)).to(device)
        
        

        # Register adjacency matrix and identity features as buffers
        self.register_buffer('adj', adj.to(dtype=torch.float32))
        self.register_buffer('x', torch.eye(voc_size).to(device))

        # Two-layer GCN architecture
        self.gcn1 = GraphConvolution(voc_size, emb_dim, device=device)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim, device=device)

    def forward(self):
        """
        Compute node embeddings through two GCN layers.
        
        Args:
            input: Node feature matrix of shape (batch_size, num_nodes, in_features).
        
        Returns:
            Node embeddings of shape (batch_size, num_nodes, emb_dim).
        """

        # First GCN layer: voc_size -> emb_dim
        node_embedding = self.gcn1(self.x, self.adj)
        
        # Non-linearity (ReLU is in-place)
        node_embedding = F.relu(node_embedding)
        
        # Regularization to prevent overfitting (Dropout is in-place)
        node_embedding = self.dropout(node_embedding)
        
        # Second GCN layer: emb_dim -> emb_dim
        node_embedding = self.gcn2(node_embedding, self.adj)
        
        return node_embedding

    def normalize(self, mx):
        """
        Row-normalize a matrix so each row sums to 1.
        
        Args:
            mx: Input matrix as tensor.
        
        Returns:
            Row-normalized matrix as tensor.
        """
        rowsum = mx.sum(1)
        r_inv = torch.pow(rowsum, -1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv.matmul(mx)
        return mx



    
class GamenetFast(BaseModel, nn.Module):
    def __init__(self, cfg: GameNetTrainConfig, vocab_size, ehr_adj, ddi_adj, device, emb_dim=64, ddi_in_memory=True):
        super().__init__()
        nn.Module.__init__(self)
        self.name: str = "gamenet"
        
        
        self.cfg = cfg
        K = len(vocab_size)
        self.K = K
        self.vocab_size = vocab_size
        self.device = device
        
        self.ehr_adj = ehr_adj.to(device=device)
        self.ddi_adj = ddi_adj.to(dtype=torch.float32, device=device)
        
        
        self.ddi_in_memory = ddi_in_memory
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim).to(device) for i in range(K-1)])
        self.dropout = nn.Dropout(p=0.4).to(device)

        self.encoders = nn.ModuleList([nn.GRU(emb_dim, emb_dim*2, batch_first=True) for _ in range(K-1)])

        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )

        self.ehr_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ehr_adj, device=device)
        self.ddi_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ddi_adj, device=device)
        self.inter = nn.Parameter(torch.FloatTensor(1))

        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 3, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, vocab_size[2])
        )

        self.init_weights()
        
    def forward(self, batch, batch_size):
        diag_history, proc_history, meds_history = batch
        batch_size = diag_history.shape[0]
        diag_history = diag_history.long()  # Cast to LongTensor
        proc_history = proc_history.long()
        

        # Ensure diag_history and proc_history are batched correctly
        diag_embedding = self.dropout(self.embeddings[0](diag_history))  # (batch_size, seq_len, emb_dim)
        proc_embedding = self.dropout(self.embeddings[1](proc_history))  # (batch_size, seq_len, emb_dim)
        
        # Mean embeddings (batch_size, emb_dim)
        diag_mean_embedding = diag_embedding.mean(dim=2)
        proc_mean_embedding = proc_embedding.mean(dim=2)
        
        diag_output, diag_hidden = self.encoders[0](diag_mean_embedding)  # (batch_size, seq_len_diag, emb_dim*2)
        proc_output, proc_hidden = self.encoders[1](proc_mean_embedding)

        # Concatenate (batch_size, 2*emb_dim)
        patient_representation = torch.cat([diag_output, proc_output], dim=-1)
        queries = self.query(patient_representation)  # (batch_size, dim)
        

        query = queries[:, -1, :]  # (batch_size, dim)

        # Apply the same adjacency matrices to all samples in the batch
        if self.ddi_in_memory:
            drug_memory = self.ehr_gcn() - self.ddi_gcn() * self.inter  # (batch_size, dim)
        else:
            drug_memory = self.ehr_gcn()
            
        drug_memory = drug_memory.unsqueeze(0)
        drug_memory = drug_memory.repeat(batch_size, 1, 1)

        # Handle history (batch_size, seq_len-1, dim)
        
        history_keys = queries[:, :-1, :]
        history_values = meds_history[:, :-1, :]
        

        # Memory interaction (batch_size, dim)
        key_weights1 = F.softmax(torch.bmm(query.unsqueeze(1), drug_memory.transpose(1, 2)), dim=-1)
        fact1 = torch.bmm(key_weights1, drug_memory)

        if len(diag_history) > 1:
            visit_weight = F.softmax(torch.bmm(query.unsqueeze(1), history_keys.transpose(1, 2)), dim=-1)
            weighted_values = visit_weight.bmm(history_values)
            fact2 = torch.bmm(weighted_values, drug_memory)
        else:
            fact2 = fact1
        
        
        # Concatenate and pass through the final output layers
        output = self.output(torch.cat([query.unsqueeze(1), fact1, fact2], dim=-1))
        
        # Training logic
        if self.training:
            neg_pred_prob = torch.sigmoid(output)
            neg_pred_prob = torch.matmul(neg_pred_prob.transpose(1, 2), neg_pred_prob)
            neg_pred_prob = neg_pred_prob.view(-1, 147)
            batch_neg = torch.sparse.mm(self.ddi_adj, neg_pred_prob.T).T
            batch_neg = batch_neg.view(batch_size, 147, 147)
            batch_neg = batch_neg.mean(dim=(1, 2))
            return output, batch_neg
        else:
            return output

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)

        self.inter.data.uniform_(-initrange, initrange)
    
        
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        training_start_time = time.time()
        batch_size = self.cfg.batch_size
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        logger.info(f"Set random seed to {self.cfg.seed}")

        self.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)
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
        ignore_ids = getattr(self.cfg, "ignore_ids", [0, 1])  # Optional[Set[int]]

        # pick what "best" means
        # default: maximise jaccard, tie-break: minimise ddi_rate_pred
        def compute_val_score(metrics: Dict[str, float]) -> float:
            """
            Compute validation score for model selection.
            Score = Jaccard - 0.1 * DDI_rate (higher is better)
            """
            return float(metrics["jaccard"] - 0.1 * metrics["ddi_rate_pred"])

        # Setup checkpoint directory
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
                medication_logits, ddi_loss = self(batch, batch_size)
                _, _, meds_history = batch
                target = meds_history[:, -1:, :]
                bce_loss = bce_loss_fn(medication_logits, target)
                ddi_loss = ddi_loss.mean()
                total_loss = 0.9 * bce_loss + self.cfg.ddi_lambda * ddi_loss
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
                        f"DDI={ddi_loss.item():.4f})"
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
                    medication_logits = self(batch, batch_size)
                    _, _, meds_history = batch
                    target = meds_history[:, -1, :].squeeze(1)
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
    
    # @torch.no_grad()
    # def _eval_loss(self, samples: List[Sample], *, bce: nn.Module) -> float:
    #     self.eval()
    #     total = 0.0
    #     n = 0
    #     for prefix, target_meds in samples:
    #         logits = self(prefix)  # eval forward returns only output
    #         y = self._multi_hot(target_meds, self.vocab_size[2], self.device).unsqueeze(0)
    #         loss = bce(logits, y)
    #         total += float(loss.item())
    #         n += 1
    #     return total / max(n, 1)
    
    def predict(self, X):
        return super().predict(X)

    # @torch.no_grad()
    # def predict(
    #     self,
    #     prefix: List[Visit],
    #     *,
    #     threshold: float = 0.5,
    #     topk: Optional[int] = None,
    #     return_probs: bool = False,
    # ):
    #     self.eval()
    #     logits = self(prefix)  # (1, size)
    #     probs = torch.sigmoid(logits).squeeze(0)

    #     if topk is not None:
    #         idx = torch.topk(probs, int(topk)).indices.tolist()
    #         return (idx, probs.detach().cpu().numpy()) if return_probs else idx

    #     idx = (probs >= threshold).nonzero(as_tuple=False).squeeze(-1).tolist()
    #     return (idx, probs.detach().cpu().numpy()) if return_probs else idx
    
    # @torch.no_grad()
    # def _eval_val_metrics(
    #     self,
    #     samples: List["Sample"],
    #     *,
    #     ddi_adj: np.ndarray,
    #     threshold: float = 0.5,
    #     ignore_ids: Optional[Set[int]] = None,
    # ) -> Dict[str, float]:
    #     y_true, y_pred = [], []
    #     for prefix, target_meds in samples:
    #         pred = self.predict(prefix, threshold=threshold)  # uses your method
    #         y_true.append(target_meds)
    #         y_pred.append(pred)
    #     return evaluate_multilabel_sets(
    #         y_true=y_true,
    #         y_pred=y_pred,
    #         ddi_adj=ddi_adj,
    #         ignore_ids=ignore_ids,
    #     )
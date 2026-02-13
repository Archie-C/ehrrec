import math
import time
import copy
import torch
import einops

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from pathlib import Path
from einops import rearrange
from torch.nn.parameter import Parameter
from typing import Dict, Any, List, Optional, Set
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR, CyclicLR

from src.utils.logging import get_logger
from src.models.BaseModel import BaseModel
from src.utils.metrics import evaluate_multilabel_sets
from src.adapters.original.GamenetAdapter import Sample, Visit

logger = get_logger("FastRx Original Model")

class FastRxOriginalConfig:
    seed: int = 42
    
    dropout_rate: float = 0.2
    embedding_dim: int = 256
    embedding_dim_fastformer: int = 128
    
    optim: str = "Adam"
    sched: str = "StepLR"
    lr: float = 5e-4
    weight_decay: float = 1e-5
    ddi_lambda: float = 0.1
    max_grad_norm: float = 5.0
    log_every: int = 200
    epochs: int = 50
    
    save_dir : str = "saved/FastRxOriginal"
    ckpt_name : str = "best_model.pt"


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, adj, device=torch.device('cpu:0')):
        super(GCN, self).__init__()
        self.voc_size = vocab_size
        self.emb_dim = embedding_dim
        self.device = device

        adj = self.normalize(adj + np.eye(adj.shape[0]))

        self.adj = torch.FloatTensor(adj).to(device)
        self.x = torch.eye(vocab_size).to(device)

        self.gcn1 = GraphConvolution(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(embedding_dim, embedding_dim)

    def forward(self):
        node_embedding = self.gcn1(self.x, self.adj)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, self.adj)
        return node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

class Fastformer(nn.Module):
    def __init__(self, dim = 3, decode_dim = 16):
        super(Fastformer, self).__init__()
        # Generate weight for Wquery、Wkey and Wvalue
        self.to_qkv = nn.Linear(dim, decode_dim * 3, bias = False)
        self.weight_q = nn.Linear(dim, decode_dim, bias = False)
        self.weight_k = nn.Linear(dim, decode_dim, bias = False)
        self.weight_v = nn.Linear(dim, decode_dim, bias = False)
        self.weight_r = nn.Linear(decode_dim, decode_dim, bias = False)
        self.weight_alpha = nn.Parameter(torch.randn(decode_dim))
        self.weight_beta = nn.Parameter(torch.randn(decode_dim))
        self.scale_factor = decode_dim ** -0.5

    def forward(self, x, mask = None):
        query = self.weight_q(x)
        key = self.weight_k(x)
        value = self.weight_v(x)
        b, n, d = query.shape

        mask_value = -torch.finfo(x.dtype).max
        mask = rearrange(mask, 'b n -> b () n')

        # Caculate the global query
        alpha_weight = (torch.mul(query, self.weight_alpha) * self.scale_factor).masked_fill(~mask, mask_value)
        alpha_weight = torch.softmax(alpha_weight, dim = -1)
        global_query = query * alpha_weight
        global_query = torch.einsum('b n d -> b d', global_query)

        # Model the interaction between global query vector and the key vector
        repeat_global_query = einops.repeat(global_query, 'b d -> b copy d', copy = n)
        p = repeat_global_query * key
        beta_weight = (torch.mul(p, self.weight_beta) * self.scale_factor).masked_fill(~mask, mask_value)
        beta_weight = torch.softmax(beta_weight, dim = -1)
        global_key = p * beta_weight
        global_key = torch.einsum('b n d -> b d', global_key)

        # key-value
        key_value_interaction = torch.einsum('b j, b n j -> b n j', global_key, value)
        key_value_interaction_out = self.weight_r(key_value_interaction)
        result = key_value_interaction_out + query
        return result

class FastRxOriginal(BaseModel, nn.Module):
    def __init__(
        self,
        vocab_size, 
        device,
        ehr_adj,
        ddi_adj,
        cfg: FastRxOriginalConfig = FastRxOriginalConfig(),
    ):
        super().__init__()
        nn.Module.__init__(self)
        self.cfg = cfg
        
        self.vocab_size = vocab_size
        self.device = device
        self.ddi_adj = ddi_adj
        
        self.fastformer = Fastformer(dim=cfg.embedding_dim_fastformer * 2, decode_dim=cfg.embedding_dim)
        self.dropout = nn.Dropout(p=cfg.dropout_rate)
        
        self.ehr_gcn = GCN(vocab_size=vocab_size[2], embedding_dim=cfg.embedding_dim, adj=ehr_adj, device=device)
        self.ddi_gcn = GCN(vocab_size=vocab_size[2], embedding_dim=cfg.embedding_dim, adj=ddi_adj, device=device)
        self.inter = nn.Parameter(torch.FloatTensor(1))
        self.embedding = nn.Embedding(vocab_size[0] + vocab_size[1] + 2, cfg.embedding_dim_fastformer)
        
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        
        self.cnn1d = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3, padding="same", stride=1),
            nn.ReLU(),
            nn.Dropout(p=cfg.dropout_rate)
        )
        
        self.output = nn.Sequential(
            nn.Linear(cfg.embedding_dim * 3, cfg.embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(cfg.embedding_dim * 2, vocab_size[2])
        )
    
    def forward(self, x):
	    # patient health representation
        i1_seq, i2_seq = [], []
        def mean_embedding(embedding):
            return embedding.mean(dim=1).unsqueeze(dim=0)  # (1,1,dim)

        for adm in x:
            i1 = mean_embedding(self.dropout(self.embedding(torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device))))
            i2 = mean_embedding(self.dropout(self.embedding(torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device))))

            i1_seq.append(i1)
            i2_seq.append(i2)

        i1_seq = torch.cat(i1_seq, dim=1) #(1,seq,dim)
        i2_seq = torch.cat(i2_seq, dim=1) #(1,seq,dim)

        i1_seq = self.cnn1d(i1_seq.permute(1, 0, 2))
        i2_seq = self.cnn1d(i2_seq.permute(1, 0, 2))
        i1_seq = i1_seq.permute(1, 0, 2)
        i2_seq = i2_seq.permute(1, 0, 2)

        h = torch.cat([i1_seq, i2_seq], dim=-1) # (seq, dim*2)

        mask = torch.ones(1, self.cfg.embedding_dim).to(torch.bool).to(self.device)
        feat = self.fastformer(h, mask).squeeze(0)

        # graph memory module
        '''I:generate current x'''
        query = feat[-1:] # (1,dim)
        '''G:generate graph memory bank and insert history information'''
        drug_memory = self.ehr_gcn() - self.ddi_gcn() * self.inter  # (size, dim)

        if len(x) > 1:
            history_keys = feat[:(feat.size(0)-1)] # (seq-1, dim)
            history_values = np.zeros((len(x)-1, self.vocab_size[2]))
            for idx, adm in enumerate(x):
                if idx == len(x)-1:
                    break
                history_values[idx, adm[2]] = 1
            history_values = torch.FloatTensor(history_values).to(self.device) # (seq-1, size)

        '''O:read from global memory bank and dynamic memory bank'''
        # print(query.shape, drug_memory.t().shape)
        key_weights1 = F.softmax(torch.mm(query, drug_memory.t()), dim=-1)  # (1, size)
        fact1 = torch.mm(key_weights1, drug_memory)  # (1, dim)

        if len(x) > 1:
            visit_weight = F.softmax(torch.mm(query, history_keys.t())) # (1, seq-1)
            weighted_values = visit_weight.mm(history_values) # (1, size)
            fact2 = torch.mm(weighted_values, drug_memory) # (1, dim)
        else:
            fact2 = fact1
        '''R:convert O and predict'''
        result = self.output(torch.cat([query, fact1, fact2], dim=-1)) # (1, dim)

        neg_pred_prob = F.sigmoid(result)
        neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)
        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

        return result, batch_neg
    
    def fit(self, train_samples, val_samples=None):
        
        training_start_time = time.time()
        
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        logger.info(f"Set random seed to {self.cfg.seed}")
        
        self.to(self.device)
        
        if self.cfg.optim == 'Adam':
            optimizer = optim.Adam(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        elif self.cfg.optim == 'AdamW':
            optimizer = optim.AdamW(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        elif self.cfg.optim == 'RMSProp':
            optimizer = optim.RMSprop(self.parameters(), lr=self.cfg.lr, momentum=0.9, weight_decay=self.cfg.weight_decay)
        elif self.cfg.optim == 'SGD':
            optimizer = optim.SGD(self.parameters(), lr=self.cfg.lr, momentum=0.9, weight_decay=self.cfg.weight_decay)

        if self.cfg.sched == 'StepLR':
            scheduler = StepLR(optimizer, step_size=25, gamma=0.95)
        elif self.cfg.sched == 'Reduce':
            scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True, factor=0.95)
        elif self.cfg.sched == 'Cosine':
            scheduler = CosineAnnealingLR(optimizer,
                                    T_max = self.cfg.num_epochs,    # Maximum number of iterations.
                                    eta_min = 5e-5)             # Minimum learning rate.
        elif self.cfg.sched == 'Cyclic':
            scheduler = CyclicLR(optimizer, base_lr=9.75e-5, max_lr=1.25e-4, cycle_momentum=False,
                                step_size_up=10, step_size_down=20, mode='triangular') # triangular, triangular2, exp_range

        bce_loss_fn = nn.BCEWithLogitsLoss()
        
        logger.info(f"Initialized {self.cfg.optim} optimizer with lr={self.cfg.lr}, weight_decay={self.cfg.weight_decay}")
        
        history: Dict[str, Any] = {
            "train_loss": [],
            "val_metrics": [],
            "best_epoch": None,
            "best_score": None,
            "best_ckpt_path": None,
            "training_time": None,
        }
        
        global_step = 0

        threshold = getattr(self.cfg, "threshold", 0.5)
        ignore_ids = getattr(self.cfg, "ignore_ids", [0, 1])

        if val_samples is not None:
            assert self.ddi_adj is not None, "Need self.ddi_adj (np.ndarray) to compute validation DDI metrics"
        
        
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

        # Track best model
        best_score = -float("inf")
        best_epoch = -1
        best_state = None
        
        logger.info(f"Starting training for {self.cfg.epochs} epochs on {len(train_samples)} samples")
        
        for epoch in range(1, self.cfg.epochs + 1):
            epoch_start_time = time.time()
            self.train()

            sample_indices = np.random.permutation(len(train_samples))

            epoch_loss_sum = 0.0
            num_batches = 0

            for idx in sample_indices:
                visit_history, target_medications = train_samples[idx]
                
                medication_logits, ddi_loss = self(visit_history)
            
                # Prepare target as multi-hot vector
                target_multi_hot = self._multi_hot(
                    target_medications, 
                    self.vocab_size[2],  # medication vocabulary size
                    self.device
                ).unsqueeze(0)  # (1, num_medications)

                # Compute losses
                bce_loss = bce_loss_fn(medication_logits, target_multi_hot)
                
                # Combined loss: 90% BCE + weighted DDI penalty
                total_loss = 0.9 * bce_loss + self.cfg.ddi_lambda * ddi_loss

                # Backward pass and optimization
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                if self.cfg.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.cfg.max_grad_norm)
                
                optimizer.step()

                # Accumulate metrics
                epoch_loss_sum += float(total_loss.item())
                num_batches += 1
                global_step += 1

                # Periodic logging during epoch
                if self.cfg.log_every and global_step % self.cfg.log_every == 0:
                    logger.debug(
                        f"[Epoch {epoch}/{self.cfg.epochs}] Step {global_step} | "
                        f"Loss={total_loss.item():.4f} (BCE={bce_loss.item():.4f}, "
                        f"DDI={ddi_loss.item():.4f})"
                    )
            
            avg_train_loss = epoch_loss_sum / max(num_batches, 1)
            epoch_time = time.time() - epoch_start_time
            history["train_loss"].append(avg_train_loss)
            
            if self.cfg.sched == 'Reduce':
                scheduler.step(avg_train_loss)
            else:
                scheduler.step()
            
            logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s | Train Loss={avg_train_loss:.4f}")
            if val_samples is not None:
                val_start_time = time.time()
                
                # Compute validation metrics
                val_metrics = self._eval_val_metrics(
                    val_samples,
                    ddi_adj=self.ddi_adj,
                    threshold=threshold,
                    ignore_ids=ignore_ids,
                )
                history["val_metrics"].append(val_metrics)
                
                val_time = time.time() - val_start_time

                # Log validation results
                logger.info(
                    f"Epoch {epoch} Validation ({val_time:.2f}s) | "
                    f"Precision={val_metrics['precision']:.4f} "
                    f"Recall={val_metrics['recall']:.4f} "
                    f"F1={val_metrics['f1']:.4f} "
                    f"Jaccard={val_metrics['jaccard']:.4f} "
                    f"DDI_pred={val_metrics['ddi_rate_pred']:.4f} "
                    f"DDI_true={val_metrics['ddi_rate_true']:.4f}"
                )

                # Check if this is the best model so far
                current_score = compute_val_score(val_metrics)
                if current_score > best_score:
                    best_score = current_score
                    best_epoch = epoch
                    best_state = copy.deepcopy(self.state_dict())
                    
                    logger.info(
                        f"New best model! Score={best_score:.4f} "
                        f"(Jaccard={val_metrics['jaccard']:.4f}, "
                        f"DDI={val_metrics['ddi_rate_pred']:.4f})"
                    )

                    # Save best checkpoint
                    torch.save(
                        {
                            "model_state_dict": best_state,
                            "epoch": epoch,
                            "score": float(best_score),
                            "val_metrics": val_metrics,
                            "cfg": self.cfg,
                        },
                        ckpt_path,
                    )
                    logger.info(f"Checkpoint saved to {ckpt_path}")

        # ========================================================================
        # Load best model and finalize
        # ========================================================================
        if val_samples is not None and best_state is not None:
            logger.info(f"Loading best checkpoint from epoch {best_epoch}")
            
            # Load best model weights
            checkpoint = torch.load(ckpt_path, weights_only=False)
            self.load_state_dict(checkpoint["model_state_dict"])

            # Update history with best model info
            history["best_epoch"] = int(checkpoint.get("epoch", best_epoch))
            history["best_score"] = float(checkpoint.get("score", best_score))
            history["best_ckpt_path"] = str(ckpt_path)

            best_metrics = checkpoint.get("val_metrics", {})
            if best_metrics:
                logger.info(
                    f"Best model loaded: Epoch {history['best_epoch']} | "
                    f"Score={history['best_score']:.4f} | "
                    f"Jaccard={best_metrics.get('jaccard', float('nan')):.4f} | "
                    f"DDI_pred={best_metrics.get('ddi_rate_pred', float('nan')):.4f}"
                )

        # Record total training time
        total_training_time = time.time() - training_start_time
        history["training_time"] = total_training_time
        
        logger.info(f"Training completed in {total_training_time:.2f}s ({total_training_time/60:.2f} minutes)")
        
        return history

    @staticmethod
    def _multi_hot(med_ids: List[int], size: int, device: torch.device) -> torch.Tensor:
        """
        Convert a list of medication IDs to a multi-hot encoded tensor.
        
        Creates a binary vector where positions corresponding to prescribed medications
        are set to 1.0, and all other positions are 0.0. This is used to represent
        medication sets as targets for multi-label classification.
        
        Args:
            med_ids: List of medication indices that should be set to 1.
                Can be empty for visits with no medications.
            size: Total vocabulary size (number of possible medications).
            device: Device to create the tensor on.
        
        Returns:
            Binary tensor of shape (size,) with 1.0 at positions in med_ids,
            0.0 elsewhere.
        
        Example:
            >>> _multi_hot([0, 2, 5], size=10, device='cpu')
            tensor([1., 0., 1., 0., 0., 1., 0., 0., 0., 0.])
        """
        # Initialize zero vector
        y = torch.zeros(size, device=device)
        
        # Set positions corresponding to prescribed medications to 1
        if med_ids:
            y[torch.tensor(med_ids, dtype=torch.long, device=device)] = 1.0
        
        return y
    
    @torch.no_grad()
    def predict(
        self,
        prefix: List[Visit],
        *,
        threshold: float = 0.5,
        topk: Optional[int] = None,
        return_probs: bool = False,
    ):
        """
        Predict medications for a patient given their visit history.
        
        Two prediction modes:
        1. Threshold-based: Return all medications with probability >= threshold
        2. Top-k: Return the k medications with highest probabilities
        
        Args:
            prefix: Patient's visit history as a list of visits. Each visit is
                a tuple of [diagnoses, procedures, medications].
            threshold: Probability threshold for binary prediction (default: 0.5).
                Only used when topk is None.
            topk: If specified, return the top-k medications by probability instead
                of using threshold-based prediction.
            return_probs: If True, return both medication indices and their probabilities.
                If False, return only medication indices (default: False).
        
        Returns:
            If return_probs=False: List of predicted medication indices
            If return_probs=True: Tuple of (medication_indices, probability_array)
        """
        self.eval()
        
        # Forward pass to get medication logits
        medication_logits, _ = self(prefix)  # (1, num_medications)
        
        # Convert logits to probabilities
        medication_probs = torch.sigmoid(medication_logits).squeeze(0)  # (num_medications,)

        # Top-k prediction mode
        if topk is not None:
            # Get indices of top-k medications
            top_indices = torch.topk(medication_probs, int(topk)).indices.tolist()
            
            if return_probs:
                return (top_indices, medication_probs.detach().cpu().numpy())
            else:
                return top_indices

        # Threshold-based prediction mode
        # Find all medications with probability >= threshold
        predicted_indices = (medication_probs >= threshold).nonzero(as_tuple=False).squeeze(-1).tolist()
        
        if return_probs:
            return (predicted_indices, medication_probs.detach().cpu().numpy())
        else:
            return predicted_indices

    @torch.no_grad()
    def _eval_val_metrics(
        self,
        samples: List["Sample"],
        *,
        ddi_adj: np.ndarray,
        threshold: float = 0.5,
        ignore_ids: Optional[Set[int]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate comprehensive validation metrics for medication prediction.
        
        Computes multi-label classification metrics (Precision, Recall, F1, Jaccard)
        as well as drug-drug interaction (DDI) rates for both predictions and
        ground truth medication sets.
        
        Args:
            samples: List of (visit_history, target_medications) tuples where
                visit_history contains patient history and target_medications are
                the ground truth medications to predict.
            ddi_adj: Drug-drug interaction adjacency matrix of shape 
                (num_medications, num_medications) for computing DDI metrics.
            threshold: Probability threshold for binary classification (default: 0.5).
            ignore_ids: Optional set of medication IDs to exclude from evaluation.
                Useful for filtering out rare medications or special codes.
        
        Returns:
            Dictionary containing:
                - precision: Precision score
                - recall: Recall score
                - f1: F1 score
                - jaccard: Jaccard similarity (intersection over union)
                - ddi_rate_pred: DDI rate in predicted medication sets
                - ddi_rate_true: DDI rate in ground truth medication sets
        """
        self.eval()
        
        num_samples = len(samples)
        logger.debug(f"Evaluating {num_samples} validation samples with threshold={threshold}")
        
        # Collect predictions and ground truth for all samples
        ground_truth_sets = []
        predicted_sets = []
        
        for visit_history, target_medications in samples:
            # Get model predictions using the specified threshold
            predicted_medications = self.predict(visit_history, threshold=threshold)
            
            ground_truth_sets.append(target_medications)
            predicted_sets.append(predicted_medications)
        
        logger.debug(f"Generated predictions for {len(predicted_sets)} samples")
        
        # Compute comprehensive evaluation metrics
        metrics = evaluate_multilabel_sets(
            y_true=ground_truth_sets,
            y_pred=predicted_sets,
            ddi_adj=ddi_adj,
            ignore_ids=ignore_ids,
        )
        
        # Log key metrics at debug level (fit() will log at info level)
        logger.debug(
            f"Metrics computed: Jaccard={metrics.get('jaccard', 0):.4f}, "
            f"F1={metrics.get('f1', 0):.4f}, "
            f"DDI_pred={metrics.get('ddi_rate_pred', 0):.4f}"
        )
        
        return metrics
    
    def load_model(self, path, weights_only=True):
        checkpoint = torch.load(path, weights_only=weights_only)
        self.load_state_dict(checkpoint["model_state_dict"])
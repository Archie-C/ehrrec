import math
import time
import copy
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List
from torch.autograd import Variable

from src.utils.logging import get_logger
from src.models.BaseModel import BaseModel
from src.utils.metrics import evaluate_multilabel_sets

logger = get_logger("RareMed Original Model")

@dataclass
class RareMedOriginalConfig:
    seed: int = 42
    epochs: int = 40
    
    embedding_dim: int = 512
    encoder_layers: int = 3
    n_layers_visit: int = 3
    n_layers_procedure: int = 3
    n_heads: int = 4
    batch_size: int = 1
    adapter_dim: int = 128
    
    lr: float = 1e-5
    dropout_rate: float = 0.3
    weight_decay: float = 0.1
    max_grad_norm: float = 5.0
    weight_multi: float = 0.005
    ddi_lambda: float = 0.1
    patient_separate: bool = False
    log_every: int = 200
    
    save_dir: str = "saved/RAREMedOriginal"
    ckpt_name: str = "best_model.pt"

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0, max_len=1000):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embeddings = nn.Embedding(max_len, d_model)

        initrange = 0.1
        self.embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        pos = torch.arange(0, x.size(1), device=x.device).int().unsqueeze(0)
        x = x + self.embeddings(pos).expand_as(x)
        return x
    
class PositionalEncoding(nn.Module): 
    def __init__(self, d_model, dropout=0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        pe *= 0.1
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

class PatientEncoder(nn.Module):
    def __init__(self, cfg: RareMedOriginalConfig, vocab_size, device):
        super(PatientEncoder, self).__init__()
        self.args = cfg
        self.vocab_size = vocab_size
        self.embedding_dim = cfg.embedding_dim
        self.device = device
        
        self.special_token_count = 2
        self.special_tokens = {'<PAD>': torch.LongTensor([0,]).to(self.device), '<UNK>': torch.LongTensor([1,]).to(self.device)}
        
        self.segment_embedding = nn.Embedding(2, self.embedding_dim)
        
        if not cfg.patient_separate:
            self.embeddings = nn.ModuleList([
                nn.Embedding(vocab_size[i], self.embedding_dim) for i in range(2)
            ])
            self.special_embeddings = nn.Embedding(4, self.embedding_dim)
            self.transformer_visit = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=cfg.n_heads, dropout=cfg.dropout_rate),
                num_layers=cfg.n_layers_visit
            )
            self.positional_embedding_layer_diagnosis = LearnablePositionalEncoding(d_model=cfg.embedding_dim)
            self.positional_embedding_layer_procedure = LearnablePositionalEncoding(d_model=cfg.embedding_dim)
            self.patient_encoder = self.patient_encoder_unified
        else:
            self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], self.embedding_dim//2) for i in range(2)])  # 疾病 手术 
            self.special_embeddings = nn.Embedding(4, self.embedding_dim//2)
            self.transformer_diagnoses = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=self.embedding_dim//2, nhead=cfg.n_heads, dropout=cfg.dropout_rate),
                num_layers=cfg.n_layers_visit
            )
            self.transformer_procedure = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=self.embedding_dim//2, nhead=cfg.n_heads, dropout=cfg.dropout_rate),
                num_layers=cfg.n_layers_procedure
            )
            
            self.patient_layer = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.ReLU(),
                nn.Linear(self.embedding_dim, self.embedding_dim)
            )
            
            self.positional_embedding_layer_diagnosis = LearnablePositionalEncoding(d_model=cfg.embedding_dim//2)
            self.positional_embedding_layer_procedure = LearnablePositionalEncoding(d_model=cfg.embedding_dim//2)
            
            self.patient_encoder = self.patient_encoder_separate
        
    def patient_encoder_unified(self, batch_visits):
        batch_repr = []
        for adm in batch_visits:
            diagnoses = adm[0]
            procedures = adm[1]
            
            diagnoses_embedding = self.embeddings[0](torch.LongTensor(diagnoses).unsqueeze(dim=1).to(self.device)) # (n, 1, dim)
            procedure_embedding = self.embeddings[1](torch.LongTensor(procedures).unsqueeze(dim=1).to(self.device))  # (m, 1, dim)

            combined_embedding = torch.cat((diagnoses_embedding, procedure_embedding), dim=0)
            
            segments = torch.tensor([0] * len(diagnoses) + [1] * len(procedures)).to(self.device)
            segment_embeddings = self.segment_embedding(segments).unsqueeze(dim=1)
            combined_embedding += segment_embeddings
            
            visit_representation = self.transformer_visit(combined_embedding)[0]
            visit_representation = torch.reshape(visit_representation, (1, 1, -1))
            batch_repr.append(visit_representation)
        
        batch_repr = torch.cat(batch_repr, dim=1).to(self.device)
        batch_repr = batch_repr.squeeze(dim=0)
        return batch_repr
    
    def patient_encoder_separate(self, batch_visits):
        batch_repr_diag, batch_repr_proc = [], []
        for adm in batch_visits:
            diagnoses = adm[0]
            procedures = adm[1]
            
            diagnoses_embedding = self.embeddings[0](torch.LongTensor(diagnoses).unsqueeze(dim=1).to(self.device))
            procedure_embedding = self.embeddings[1](torch.LongTensor(procedures).unsqueeze(dim=1).to(self.device))
        
            diagnoses_embedding = self.positional_embedding_layer_disease(diagnoses_embedding)
            procedure_embedding = self.positional_embedding_layer_procedure(procedure_embedding)
            
            diagnoses_representation = self.transformer_diagnoses(diagnoses_embedding)[0]
            procedures_representation = self.transformer_procedure(procedure_embedding)[0]
            
            diagnoses_representation = diagnoses_representation.mean(dim=0)
            procedures_representation = procedures_representation.mean(dim=0)
            
            diagnoses_representation = torch.reshape(diagnoses_representation, (1, 1, -1))
            procedures_representation = torch.reshape(procedures_representation, (1, 1, -1))
            
            batch_repr_diag.append(diagnoses_representation)
            batch_repr_proc.append(procedures_representation)
        
        batch_repr_diag = torch.cat(batch_repr_diag, dim=1).to(self.device)
        batch_repr_proc = torch.cat(batch_repr_proc, dim=1).to(self.device)
        
        batch_repr = torch.cat((batch_repr_diag, batch_repr_proc), dim=-1)
        batch_repr = batch_repr.squeeze(dim=0)
        return batch_repr

class RareMedOriginal(PatientEncoder, BaseModel):
    def __init__(self, vocab_size, ddi_adj, device, cfg: RareMedOriginalConfig = RareMedOriginalConfig()):
        super(RareMedOriginal, self).__init__(cfg=cfg, vocab_size=vocab_size, device=device)
        
        self.cfg = cfg
        self.ddi_adj = ddi_adj
        self.vocab_size = vocab_size
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        
        self.init_weights()
        
        self.cls_mask = nn.Linear(self.embedding_dim, self.vocab_size[0]+self.vocab_size[1])
        
        self.cls_nsp = nn.Linear(self.embedding_dim, 1)
        
        self.cls_final = nn.Linear(self.embedding_dim, self.vocab_size[2])
    
    def forward_finetune(self, data):
        patient_repr = self.patient_encoder(data)
        result = self.cls_final(patient_repr)
        
        neg_pred_prob = F.sigmoid(result)
        neg_pred_prob = torch.matmul(neg_pred_prob.t(), neg_pred_prob)
        
        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()
        return result, batch_neg

    def forward(self, data, mode="fine_tune"):
        assert mode in ['fine_tune', 'pretrain_mask', 'pretrain_nsp']
        if mode == "fine_tune":
            result, batch_neg = self.forward_finetune(data)
            return result, batch_neg
        elif mode == "pretrain_mask":
            patient_repr = self.patient_encoder(data)
            result = self.cls_mask(patient_repr)
            return result
        elif mode == "pretrain_nsp":
            patient_repr = self.patient_encoder(data)
            result = self.cls_nsp(patient_repr)
            result = result.squeeze(dim=1)
            logit = torch.sigmoid(result)
            return logit

    def init_weights(self):
        """Initialize embedding weights."""
        initrange = 0.1
        self.embeddings[0].weight.data.uniform_(-initrange, initrange)      # disease
        self.embeddings[1].weight.data.uniform_(-initrange, initrange)      # procedure

        self.segment_embedding.weight.data.uniform_(-initrange, initrange)
        self.special_embeddings.weight.data.uniform_(-initrange, initrange)
        
    
    def fit(
        self, 
        train_samples, 
        val_samples=None,
    ):
        training_start_time = time.time()
        
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        logger.info(f"Set random seed to {self.cfg.seed}")
        
        self.to(self.device)
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr = self.cfg.lr,
            weight_decay=self.cfg.weight_decay
        )
        bce_loss_fn = nn.BCEWithLogitsLoss()
        
        logger.info(f"Initialised optimiser with lr={self.cfg.lr}, weight_decay={self.cfg.weight_decay}")
        
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
        threshold = getattr(self.cfg, "threshold", 0.5)
        ignore_ids = getattr(self.cfg, "ignore_ids", [0, 1])
        
        if val_samples is not None:
            assert self.ddi_adj is not None, \
                "Need self.ddi_adj (np.ndarray) to compute validation DDI metrics"
        
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
            
            # Shuffle training samples
            sample_indices = np.random.permutation(len(train_samples))

            epoch_loss_sum = 0.0
            num_batches = 0

            # Iterate over shuffled training samples
            for idx in sample_indices:
                visit_history, target_medications = train_samples[idx]

                # Forward pass
                medication_logits, ddi_loss = self(visit_history)  # (1, num_medications)
                medication_logits = medication_logits.mean(dim=0, keepdim=True)
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

            # Compute epoch metrics
            avg_train_loss = epoch_loss_sum / max(num_batches, 1)
            epoch_time = time.time() - epoch_start_time
            history["train_loss"].append(avg_train_loss)
            
            logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s | Train Loss={avg_train_loss:.4f}")

            # ====================================================================
            # Validation
            # ====================================================================
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
    
    @torch.no_grad()
    def predict(
        self,
        prefix,
        *,
        threshold: float = 0.5,
        topk = None,
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
        samples,
        *,
        ddi_adj: np.ndarray,
        threshold: float = 0.5,
        ignore_ids = None,
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
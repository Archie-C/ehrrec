import time
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Set

from src.utils.logging import get_logger
from src.models.BaseModel import BaseModel
from src.adapters.original.MICRONAdapter import Sample, Visit
from src.utils.metrics import evaluate_multilabel_sets


logger = get_logger("Original MICRON")

@dataclass
class MICRONOriginalConfig:
    epochs: int = 40
    seed: int = 42
    lr: int = 2e-4
    weight_decay: int = 0.0
    ddi_lambda: int = 0.1
    max_grad_norm: float = 5.0
    log_every: int = 200
    save_dir: str = "saved/MICRONOG/"
    ckpt_name: str = "best_model.pt"


class MICRONOriginal(BaseModel, nn.Module):
    def __init__(self, vocab_size, ddi_adj, emb_dim=256, device=torch.device('cpu:0'), cfg: MICRONOriginalConfig = MICRONOriginalConfig()):
        super(MICRONOriginal, self).__init__()
        nn.Module.__init__(self)
        self.cfg = cfg
        self.device = device
        self.ddi_adj = ddi_adj
        self.vocab_size = vocab_size

        # pre-embedding
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim) for i in range(2)])
        self.dropout = nn.Dropout(p=0.5)
        
        self.health_net = nn.Sequential(
                nn.Linear(2 * emb_dim, emb_dim)
        )

        # 
        self.prescription_net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.ReLU(),
            nn.Linear(emb_dim * 4, vocab_size[2])
        )

        # graphs, bipartite matrix
        self.tensor_ddi_adj = torch.tensor(ddi_adj, dtype=torch.float32)
        self.init_weights()

    def forward(self, input):
	    # patient health representation
        def sum_embedding(embedding):
            return embedding.sum(dim=1).unsqueeze(dim=0)  # (1,1,dim)
        
        diag_emb = sum_embedding(self.dropout(self.embeddings[0](torch.LongTensor(input[-1][0]).unsqueeze(dim=0).to(self.device)))) # (1,1,dim)
        prod_emb = sum_embedding(self.dropout(self.embeddings[1](torch.LongTensor(input[-1][1]).unsqueeze(dim=0).to(self.device))))

        if len(input) < 2:
            diag_emb_last = diag_emb * torch.tensor(0.0)
            prod_emb_last = diag_emb * torch.tensor(0.0)
        else:
            diag_emb_last = sum_embedding(self.dropout(self.embeddings[0](torch.LongTensor(input[-2][0]).unsqueeze(dim=0).to(self.device)))) # (1,1,dim)
            prod_emb_last = sum_embedding(self.dropout(self.embeddings[1](torch.LongTensor(input[-2][1]).unsqueeze(dim=0).to(self.device))))

        health_representation = torch.cat([diag_emb, prod_emb], dim=-1).squeeze(dim=0) # (seq, dim*2)
        health_representation_last = torch.cat([diag_emb_last, prod_emb_last], dim=-1).squeeze(dim=0) # (seq, dim*2)

        health_rep = self.health_net(health_representation)[-1:, :] # (seq, dim)
        health_rep_last = self.health_net(health_representation_last)[-1:, :] # (seq, dim)
        health_residual_rep = health_rep - health_rep_last

	    # drug representation
        drug_rep = self.prescription_net(health_rep)
        drug_rep_last = self.prescription_net(health_rep_last)
        drug_residual_rep = self.prescription_net(health_residual_rep)  

        # reconstructon loss
        rec_loss = 1 / self.tensor_ddi_adj.shape[0] * torch.sum(torch.pow((F.sigmoid(drug_rep) - F.sigmoid(drug_rep_last + drug_residual_rep)), 2))
        
        # ddi_loss
        neg_pred_prob = F.sigmoid(drug_rep)
        neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)

        batch_neg = 1 / self.tensor_ddi_adj.shape[0] *  neg_pred_prob.mul(self.tensor_ddi_adj).sum()
        return drug_rep, drug_rep_last, drug_residual_rep, batch_neg, rec_loss

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)
            
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
    
    def fit(self, train_samples, val_samples):
        training_start_time = time.time()
        
        # Set random seeds for reproducibility
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        logger.info(f"Set random seed to {self.cfg.seed}")
        
        # Setup model and optimizer
        self.to(self.device)
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay
        )
        bce_loss_fn = nn.BCEWithLogitsLoss()
        
        logger.info(f"Initialized optimizer with lr={self.cfg.lr}, weight_decay={self.cfg.weight_decay}")
        
                # Initialize training history
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
        ignore_ids = getattr(self.cfg, "ignore_ids", None)
        
        if val_samples is not None:
            assert self.ddi_adj is not None, \
                "Need self.ddi_adj (np.ndarray) to compute validation DDI metrics"
        
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
    
        # Track best model
        best_score = -float("inf")
        best_epoch = -1
        best_state = None
        
        # ========================================================================
        # Training Loop
        # ========================================================================
        logger.info(f"Starting training for {self.cfg.epochs} epochs on {len(train_samples)} samples")
        
        for epoch in range(1, self.cfg.epochs + 1):
            epoch_start_time = time.time()
            self.train()
            
            sample_indices = np.random.permutation(len(train_samples))
            
            epoch_loss_sum = 0.0
            num_batches = 0
            
            for idx in sample_indices:
                prefix, target_medications = train_samples[idx]
                
                predictions, last_predictions, residuals, ddi_loss, rec_loss = self(prefix)
                
                target_multi_hot = self._multi_hot(
                    target_medications, 
                    self.vocab_size[2],  # medication vocabulary size
                    self.device
                ).unsqueeze(0)
                
                bce_loss = bce_loss_fn(predictions, target_multi_hot)
                
                total_loss = 0.9 * bce_loss + self.cfg.ddi_lambda * ddi_loss
                
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                
                if self.cfg.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.cfg.max_grad_norm)
                
                optimizer.step()
                
                epoch_loss_sum += float(total_loss.item())
                num_batches += 1
                global_step += 1
                
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
            checkpoint = torch.load(ckpt_path, map_location=self.device)
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
        predictions, last_predictions, residuals, ddi_loss, rec_loss = self(prefix)  # (1, num_medications)
        
        # Convert logits to probabilities
        medication_probs = torch.sigmoid(predictions).squeeze(0)  # (num_medications,)

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
        
        for prefix, target_medications in samples:
            # Get model predictions using the specified threshold
            predicted_medications = self.predict(prefix, threshold=threshold)
            
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
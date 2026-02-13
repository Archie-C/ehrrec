# GamenetOG.py
import math
import copy
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from dataclasses import dataclass
from torch.nn.parameter import Parameter
from typing import Any, Dict, List, Optional, Set, Tuple

from src.models.BaseModel import BaseModel
from src.utils.logging import get_logger
from src.utils.metrics import evaluate_multilabel_sets

logger = get_logger("Original Gamenet")

Visit = List[List[int]]                 # [diag, proc, med]
Sample = Tuple[List[Visit], List[int]]

@dataclass
class GameNetTrainConfig:
    """
    Training configuration for the GAMENet model.
    
    Attributes:
        epochs: Number of training epochs (default: 20).
        lr: Learning rate for optimizer (default: 1e-3).
        weight_decay: L2 regularization penalty (default: 0.0).
        ddi_lambda: Weight for drug-drug interaction penalty in loss function.
            Higher values encourage safer medication combinations (default: 0.1).
        max_grad_norm: Maximum gradient norm for gradient clipping to prevent
            exploding gradients (default: 5.0).
        log_every: Log training metrics every N batches (default: 200).
        seed: Random seed for reproducibility (default: 42).
        batch_size: Number of samples per batch. Note: GAMENet typically uses
            batch_size=1 for sequential patient visits (default: 1).
        save_dir: Directory to save model checkpoints (default: "saved/GAMENET/").
        ckpt_name: Filename for the best model checkpoint (default: "best_model.pt").
    """
    epochs: int = 40
    lr: float = 2e-4
    weight_decay: float = 0.0
    ddi_lambda: float = 0.1
    max_grad_norm: float = 5.0
    log_every: int = 200
    seed: int = 42
    batch_size: int = 1
    save_dir: str = "saved/GAMENETOG/"
    ckpt_name: str = "best_model.pt"

class GraphConvolution(nn.Module):
    """
    Graph Convolutional Network (GCN) layer.
    
    Implements a simple graph convolution operation as described in
    Kipf & Welling (2017): https://arxiv.org/abs/1609.02907
    
    The layer performs: output = A * X * W + b
    where A is the adjacency matrix, X is the input features, W is the weight matrix,
    and b is the optional bias vector.
    
    Args:
        in_features: Number of input features per node.
        out_features: Number of output features per node.
        bias: If True, adds a learnable bias to the output (default: True).
    
    Shape:
        - Input: (num_nodes, in_features)
        - Adjacency: (num_nodes, num_nodes)
        - Output: (num_nodes, out_features)
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Learnable weight matrix for feature transformation
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        
        # Optional learnable bias vector
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters using uniform distribution.
        """
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """
        Forward pass of the graph convolution layer.
        
        Args:
            input: Node feature matrix of shape (num_nodes, in_features).
            adj: Adjacency matrix of shape (num_nodes, num_nodes).
                Should be preprocessed (e.g., normalized, self-loops added).
        
        Returns:
            Transformed node features of shape (num_nodes, out_features).
        """
        # Transform input features: X * W
        support = torch.mm(input, self.weight)
        
        # Aggregate features from neighbors: A * (X * W)
        output = torch.mm(adj, support)
        
        # Add bias if present
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        """String representation showing input and output dimensions."""
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GCN(nn.Module):
    """
    Two-layer Graph Convolutional Network for learning node embeddings.
    
    This GCN uses a fixed identity matrix as input features and learns embeddings
    for each node in the graph through two graph convolution layers with ReLU
    activation and dropout.
    
    Args:
        voc_size: Number of nodes in the graph (vocabulary size).
        emb_dim: Dimension of the learned node embeddings.
        adj: Adjacency matrix as numpy array of shape (voc_size, voc_size).
            Will be normalized and have self-loops added automatically.
        device: Device to place tensors on (default: CPU). Note: currently unused.
    
    Shape:
        - Output: (voc_size, emb_dim) - learned embedding for each node
    """
    
    def __init__(self, voc_size, emb_dim, adj):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim

        # Add self-loops and normalize adjacency matrix
        # Self-loops: each node includes itself in aggregation
        adj = self.normalize(adj + np.eye(adj.shape[0]))
        
        # Register adjacency matrix and identity features as buffers
        # Buffers are moved with the model but not trained
        self.register_buffer('adj', torch.FloatTensor(adj))
        # Identity matrix: each node starts with a one-hot feature vector
        self.register_buffer('x', torch.eye(voc_size))

        # Two-layer GCN architecture
        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        """
        Compute node embeddings through two GCN layers.
        
        Returns:
            Node embeddings of shape (voc_size, emb_dim).
        """
        # First GCN layer: voc_size -> emb_dim
        node_embedding = self.gcn1(self.x, self.adj)
        
        # Non-linearity
        node_embedding = F.relu(node_embedding)
        
        # Regularization to prevent overfitting
        node_embedding = self.dropout(node_embedding)
        
        # Second GCN layer: emb_dim -> emb_dim
        node_embedding = self.gcn2(node_embedding, self.adj)
        
        return node_embedding

    def normalize(self, mx):
        """
        Row-normalize a matrix so each row sums to 1.
        
        Args:
            mx: Input matrix as numpy array.
        
        Returns:
            Row-normalized matrix as numpy array.
        """
        # Sum each row
        rowsum = np.array(mx.sum(1))
        
        # Compute inverse: 1 / rowsum for each row
        r_inv = np.power(rowsum, -1).flatten()
        
        # Handle division by zero: set inf values to 0
        r_inv[np.isinf(r_inv)] = 0.
        
        # Create diagonal matrix from inverse row sums
        r_mat_inv = np.diagflat(r_inv)
        
        # Multiply to normalize: D^-1 * A
        mx = r_mat_inv.dot(mx)
        
        return mx


class GamenetOriginal(BaseModel, nn.Module):
    def __init__(
        self, 
        cfg: GameNetTrainConfig, 
        vocab_size, 
        ehr_adj, 
        ddi_adj, 
        emb_dim=64, 
        device=torch.device("cpu:0"), 
        ddi_in_memory=True
    ):
        """
        Initialize the GAMENet model for medication recommendation.
        
        Args:
            cfg: Training configuration containing hyperparameters.
            vocab_size: List of vocabulary sizes [diagnoses, procedures, medications].
                Typically K=3 where vocab_size[0] is diagnoses, vocab_size[1] is 
                procedures, and vocab_size[2] is medications.
            ehr_adj: EHR adjacency matrix of shape (num_meds, num_meds) representing
                co-occurrence patterns of medications in patient records.
            ddi_adj: Drug-drug interaction adjacency matrix of shape (num_meds, num_meds)
                where 1 indicates known dangerous interactions.
            emb_dim: Embedding dimension for all representations (default: 64).
            device: Device to place model tensors on (default: CPU).
            ddi_in_memory: If True, keeps DDI adjacency in memory for faster computation
                during training (default: True).
        """
        super().__init__()
        nn.Module.__init__(self)
        
        self.name: str = "gamenet"
        self.cfg = cfg
        
        # K = number of input modalities (diagnoses, procedures, medications)
        K = len(vocab_size)
        self.K = K
        self.vocab_size = vocab_size
        self.device = device
        
        # Store adjacency matrices for graph convolutions and metrics
        self.ehr_adj = ehr_adj  # EHR co-occurrence graph
        self.ddi_adj = ddi_adj  # DDI graph (for evaluation metrics)
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)  # DDI graph (for loss computation)
        self.ddi_in_memory = ddi_in_memory
        
        # Embedding layers for diagnoses and procedures (K-1 modalities)
        # Note: Medications use GCN embeddings instead
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim) for i in range(K-1)]
        )
        self.dropout = nn.Dropout(p=0.4)

        # GRU encoders for sequential modeling of diagnoses and procedures
        # Each encoder outputs emb_dim*2 hidden states
        self.encoders = nn.ModuleList(
            [nn.GRU(emb_dim, emb_dim*2, batch_first=True) for _ in range(K-1)]
        )

        # Query network to combine diagnosis and procedure representations
        # Input: concatenated diagnosis + procedure encodings (emb_dim*4 total)
        # Output: unified query vector (emb_dim)
        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )

        # Graph convolutional networks for medication embeddings
        self.ehr_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ehr_adj)  # EHR patterns
        self.ddi_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ddi_adj)  # DDI knowledge
        
        # Learnable interpolation parameter between EHR and DDI graphs
        self.inter = nn.Parameter(torch.FloatTensor(1))

        # Final output layer to predict medication probabilities
        # Input: query (emb_dim) + ehr_embedding (emb_dim) + ddi_embedding (emb_dim) = emb_dim*3
        # Output: logits for each medication (vocab_size[2])
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 3, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, vocab_size[2])
        )

        self.init_weights()
        
    def forward(self, input):
        """
        Forward pass through GAMENet for medication recommendation.
        
        Args:
            input: List of visits, where each visit is [diagnoses, procedures, medications].
                Shape: [(visit_1), (visit_2), ..., (visit_n)]
                Each visit_i is a tuple of 3 lists containing code indices.
        
        Returns:
            If training: (medication_logits, ddi_loss)
                - medication_logits: Predictions for current visit, shape (1, num_medications)
                - ddi_loss: Drug-drug interaction penalty term (scalar)
            If evaluation: medication_logits only
        """
        # ========================================================================
        # Step 1: Generate embeddings and encode sequences for diagnoses/procedures
        # ========================================================================
        
        diagnosis_seq = []
        procedure_seq = []
        
        def mean_embedding(embedding):
            """Average embeddings across codes within a single visit."""
            return embedding.mean(dim=1).unsqueeze(dim=0)  # (1, 1, emb_dim)
        
        # Process each visit in the patient history
        for visit in input:
            # Embed and average diagnoses for this visit
            diagnosis_codes = torch.LongTensor(visit[0]).unsqueeze(dim=0).to(self.device)
            diag_emb = self.dropout(self.embeddings[0](diagnosis_codes))
            diag_avg = mean_embedding(diag_emb)  # (1, 1, emb_dim)
            
            # Embed and average procedures for this visit
            procedure_codes = torch.LongTensor(visit[1]).unsqueeze(dim=0).to(self.device)
            proc_emb = self.dropout(self.embeddings[1](procedure_codes))
            proc_avg = mean_embedding(proc_emb)  # (1, 1, emb_dim)
            
            diagnosis_seq.append(diag_avg)
            procedure_seq.append(proc_avg)
        
        # Stack visits into sequences
        diagnosis_seq = torch.cat(diagnosis_seq, dim=1)  # (1, num_visits, emb_dim)
        procedure_seq = torch.cat(procedure_seq, dim=1)  # (1, num_visits, emb_dim)

        # ========================================================================
        # Step 2: Encode sequences with GRUs and generate query vectors
        # ========================================================================
        
        # Encode diagnosis and procedure sequences
        diag_output, diag_hidden = self.encoders[0](diagnosis_seq)  # output: (1, num_visits, emb_dim*2)
        proc_output, proc_hidden = self.encoders[1](procedure_seq)  # output: (1, num_visits, emb_dim*2)
        
        # Combine diagnosis and procedure representations for each visit
        combined_visits = torch.cat([diag_output, proc_output], dim=-1).squeeze(dim=0)  # (num_visits, emb_dim*4)
        
        # Transform combined representations into query vectors
        queries = self.query(combined_visits)  # (num_visits, emb_dim)

        # ========================================================================
        # Step 3: Graph Memory - Retrieve medication knowledge from EHR and DDI graphs
        # ========================================================================
        
        # Current visit query (the visit we're predicting medications for)
        current_query = queries[-1:]  # (1, emb_dim)

        # Generate graph-based medication memory bank
        # Combines EHR patterns with DDI safety constraints
        if self.ddi_in_memory:
            # Interpolate between EHR patterns and DDI safety
            # Higher inter → stronger DDI penalty
            drug_memory = self.ehr_gcn() - self.ddi_gcn() * self.inter  # (num_medications, emb_dim)
        else:
            # Use only EHR patterns
            drug_memory = self.ehr_gcn()  # (num_medications, emb_dim)

        # Build historical context from previous visits (if any exist)
        if len(input) > 1:
            # Keys: query vectors from all previous visits
            history_queries = queries[:-1]  # (num_visits-1, emb_dim)

            # Values: one-hot medication sets from previous visits
            history_medications = np.zeros((len(input) - 1, self.vocab_size[2]))
            for visit_idx, visit in enumerate(input[:-1]):  # Exclude current visit
                history_medications[visit_idx, visit[2]] = 1  # Set prescribed medications to 1
            history_medications = torch.FloatTensor(history_medications).to(self.device)  # (num_visits-1, num_medications)

        # ========================================================================
        # Step 4: Read from memory banks and generate predictions
        # ========================================================================
        
        # Read from global graph memory (EHR + DDI knowledge)
        # Attention over all medications based on current query
        global_attention = F.softmax(torch.mm(current_query, drug_memory.t()), dim=-1)  # (1, num_medications)
        global_context = torch.mm(global_attention, drug_memory)  # (1, emb_dim)

        # Read from dynamic historical memory (patient's medication history)
        if len(input) > 1:
            # Attention over previous visits based on current query
            visit_attention = F.softmax(torch.mm(current_query, history_queries.t()), dim=-1)  # (1, num_visits-1)
            
            # Weighted sum of historical medications
            historical_med_weights = visit_attention.mm(history_medications)  # (1, num_medications)
            
            # Retrieve medication embeddings weighted by history
            historical_context = torch.mm(historical_med_weights, drug_memory)  # (1, emb_dim)
        else:
            # First visit: no history available, use global context only
            historical_context = global_context
        
        # Combine query, global knowledge, and historical context to predict medications
        combined_representation = torch.cat([current_query, global_context, historical_context], dim=-1)  # (1, emb_dim*3)
        medication_logits = self.output(combined_representation)  # (1, num_medications)

        # ========================================================================
        # Training: Compute DDI penalty
        # ========================================================================
        
        if self.training:
            # Convert logits to probabilities
            med_probs = torch.sigmoid(medication_logits)
            
            # Compute pairwise medication probabilities: P(med_i AND med_j)
            pairwise_probs = med_probs.t() * med_probs  # (num_medications, num_medications)
            
            # DDI loss: expected number of dangerous interactions
            # Multiply by DDI adjacency and average
            ddi_loss = pairwise_probs.mul(self.tensor_ddi_adj).mean()

            return medication_logits, ddi_loss
        else:
            return medication_logits

    def init_weights(self):
        """
        Initialize model parameters with uniform distribution.
        
        Initializes:
        - Embedding layers for diagnoses and procedures with uniform(-0.1, 0.1)
        - Interpolation parameter (inter) between EHR and DDI graphs with uniform(-0.1, 0.1)
        
        Note: GCN layers, GRU encoders, and output MLP use their default PyTorch
        initialization and are not re-initialized here.
        """
        initrange = 0.1
        
        # Initialize diagnosis and procedure embeddings
        for embedding_layer in self.embeddings:
            embedding_layer.weight.data.uniform_(-initrange, initrange)

        # Initialize EHR-DDI interpolation parameter
        self.inter.data.uniform_(-initrange, initrange)
    
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
        
    def fit(
        self,
        train_samples: List["Sample"],
        val_samples: Optional[List["Sample"]] = None,
    ) -> Dict[str, Any]:
        """
        Train the GAMENet model on patient visit sequences.
        
        Args:
            train_samples: List of training samples, where each sample is a tuple
                (visit_history, target_medications). visit_history is a list of visits,
                and target_medications is a list of medication indices to predict.
            val_samples: Optional validation samples in the same format as train_samples.
                If provided, validation metrics are computed after each epoch and the
                best model checkpoint is saved.
        
        Returns:
            Dictionary containing training history:
                - train_loss: List of average training loss per epoch
                - val_metrics: List of validation metric dicts per epoch
                - best_epoch: Epoch number with best validation score
                - best_score: Best validation score achieved
                - best_ckpt_path: Path to saved best model checkpoint
                - training_time: Total training time in seconds
        """
        training_start_time = time.time()
        
        # Validate configuration
        assert self.cfg.batch_size == 1, \
            "Current forward() implementation processes single samples; keep batch_size=1"

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
        ignore_ids = getattr(self.cfg, "ignore_ids", [0, 1])  # Optional[Set[int]]
        
        # Validate DDI adjacency matrix is available for metrics
        if val_samples is not None:
            assert self.ddi_adj is not None, \
                "Need self.ddi_adj (np.ndarray) to compute validation DDI metrics"

        # Define scoring function for model selection
        # Higher score is better: prioritize Jaccard similarity, penalize DDI rate
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
            
            # Shuffle training samples
            sample_indices = np.random.permutation(len(train_samples))

            epoch_loss_sum = 0.0
            num_batches = 0

            # Iterate over shuffled training samples
            for idx in sample_indices:
                visit_history, target_medications = train_samples[idx]

                # Forward pass
                medication_logits, ddi_loss = self(visit_history)  # (1, num_medications)

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
        medication_logits = self(prefix)  # (1, num_medications)
        
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
        
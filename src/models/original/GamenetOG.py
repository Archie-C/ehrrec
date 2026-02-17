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
from src.utils.metrics import Metrics, ddi_rate_score, evaluate_multilabel_sets
from src.utils.one_hot_encode import multihot_to_margin_target, pred_indices_from_logits

logger = get_logger("Original Gamenet")

Visit = List[List[int]]                 # [diag, proc, med]
Sample = Tuple[List[Visit], List[int]]


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
        
        self.bce_loss_fn = nn.BCEWithLogitsLoss()
        self.mlm_loss_fn = nn.MultiLabelMarginLoss()

        self.init_weights()
        
    def forward(self, batch, artefacts):
        """
        Forward pass through GAMENet for medication recommendation.
        
        Args:
            batch: List of visits, where each visit is [diagnoses, procedures, medications].
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
        
        history_medications = batch["med_history"]
        diagnoses = batch["diagnoses"]
        procedures = batch["procedures"]
        
        diagnosis_seq = []
        procedure_seq = []
        
        def mean_embedding(embedding):
            """Average embeddings across codes within a single visit."""
            return embedding.mean(dim=1).unsqueeze(dim=0)  # (1, 1, emb_dim)
        
        for diag in diagnoses:
            diagnosis_codes = torch.tensor([diag], dtype=torch.long, device=self.device)
            diag_emb = self.dropout(self.embeddings[0](diagnosis_codes))
            diag_avg = mean_embedding(diag_emb)
            diagnosis_seq.append(diag_avg)
            
        for proc in procedures:
            procedure_codes = torch.tensor([proc], dtype=torch.long, device=self.device)
            proc_emb = self.dropout(self.embeddings[1](procedure_codes))
            proc_avg = mean_embedding(proc_emb)
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
        # if len(batch) > 1:
        #     # Keys: query vectors from all previous visits
        #     history_queries = queries[:-1]  # (num_visits-1, emb_dim)

        #     # Values: one-hot medication sets from previous visits
        #     history_medications = np.zeros((len(batch) - 1, self.vocab_size[2]))
        #     for visit_idx, visit in enumerate(batch[:-1]):  # Exclude current visit
        #         history_medications[visit_idx, visit[2]] = 1  # Set prescribed medications to 1
        #     history_medications = torch.FloatTensor(history_medications).to(self.device)  # (num_visits-1, num_medications)
        # ========================================================================
        # Step 4: Read from memory banks and generate predictions
        # ========================================================================
        
        # Read from global graph memory (EHR + DDI knowledge)
        # Attention over all medications based on current query
        global_attention = F.softmax(torch.mm(current_query, drug_memory.t()), dim=-1)  # (1, num_medications)
        global_context = torch.mm(global_attention, drug_memory)  # (1, emb_dim)

        # Read from dynamic historical memory (patient's medication history)
        if history_medications is not None:
            history_queries = queries[:-1]
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
        
    def compute_loss(self, batch, artefacts, target_ddi=0.05, T=0.5, threshold=0.5):
        """Computes a custom loss for a batch input

        Args:
            batch (_type_): _description_
        """

        logits, ddi_loss = self(batch, artefacts)
        y = batch["y"].to(self.device).float()
        
        margin_target = multihot_to_margin_target(y)
        
        bce_loss = self.bce_loss_fn(logits, y)
        mlm_loss = self.mlm_loss_fn(logits, margin_target)
        pred_loss = 0.9 * bce_loss + 0.1 * mlm_loss
        
        with torch.no_grad():
            pred_lists = pred_indices_from_logits(logits, threshold=threshold)
            current_ddi_rate = ddi_rate_score(pred_lists, ddi_adj=artefacts.get("ddi_adj", None))
        
        if current_ddi_rate <= target_ddi:
            return pred_loss
        else:
            rnd = torch.exp(torch.tensor((target_ddi - current_ddi_rate) / T, device=self.device)).item()
            if torch.rand(1, device=self.device).item() < rnd:
                return ddi_loss
            else:
                return pred_loss
    
    def compute_metrics(self, batch, artefacts) -> Metrics:
        logits = self(batch, artefacts)
        pred_multi_hot = (torch.sigmoid(logits) >= 0.5).int()
        pred_probs = torch.sigmoid(logits)
        
        metrics = Metrics()
        metrics.compute_ddi(pred_multi_hot, artefacts.get("ddi_adj", None))
        metrics.compute_jaccard(pred_multi_hot, batch["y"])
        metrics.compute_f1(pred_multi_hot, batch["y"])
        metrics.compute_prauc(pred_probs, batch["y"])
        metrics.compute_num_meds(pred_multi_hot)
        
        return metrics
    
    def load_model(self, path, weights_only=True):
        checkpoint = torch.load(path, weights_only=weights_only)
        self.load_state_dict(checkpoint["model_state_dict"])
        
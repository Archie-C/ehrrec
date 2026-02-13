import time
import copy
import math
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any
from torch_geometric.utils import degree
from torch.nn.parameter import Parameter
from torch_geometric.nn.inits import uniform
from torch_scatter import scatter_mean, scatter_add, scatter_std
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.nn import global_mean_pool, global_add_pool, MessagePassing, global_max_pool, GlobalAttention, Set2Set

from src.utils.logging import get_logger
from src.models.BaseModel import BaseModel
from src.utils.metrics import evaluate_multilabel_sets

logger = get_logger("Original SubRec Model")

@dataclass
class SubRecOriginalConfig:
    seed: int = 42
    epochs: int = 40
    lr: float = 2e-4
    weight_decay: float = 1e-5
    max_grad_norm: float = 5.0
    ddi_lambda: float = 0.1
    log_every: int = 200
    
    mu: float = 1e-4
    gamma: float = 2e-4
    
    save_dir: str = "saved/SubRecOriginal"
    ckpt_name: str = "best_model.pt"
    
    embedding_dim: int = 128
    weighted_rnn: bool = False
    n_heads: int = 4
    n_layers: int = 1
    dropout_rate: float = 0.5
    
    # GNN Graph
    gnn_num_layer: int = 5
    gnn_embedding_dim: int = 128
    gnn_type: str = 'gin'
    gnn_virtual_node: bool = True
    gnn_residual: bool = False
    gnn_drop_ratio: float = 0.5
    gnn_JK: str = "last"
    gnn_graph_pooling: str = "mean"
    
    codebook_size: int = 128
    alpha: float = 1
    
class VQVAE(nn.Module):
    def __init__(self, input_key_dim, input_value_dim, codebook_size, hidden_size,alpha):
        super(VQVAE, self).__init__()
        self.input_key_dim = input_key_dim
        self.input_value_dim = input_value_dim
        self.hidden_key_size = hidden_size
        self.hidden_value_size = input_value_dim
        self.codebook_size = codebook_size
        self.alpha = alpha
        # Encoder and Decoder for `history_keys`
        self.encoder_keys = nn.Linear(input_key_dim, hidden_size)
        self.decoder_keys = nn.Linear(hidden_size, input_key_dim)
        
        # Encoder and Decoder for `history_values`
        self.encoder_values = nn.Linear(input_value_dim, input_value_dim)
        self.decoder_values = nn.Linear(input_value_dim, input_value_dim)
        
        # Codebooks for both `history_keys` and `history_values`
        self.codebook_keys = nn.Embedding(codebook_size, hidden_size)
        self.codebook_values = nn.Embedding(codebook_size, input_value_dim)
        
        # Codebook initialization
        nn.init.uniform_(self.codebook_keys.weight, -1.0 / codebook_size, 1.0 / codebook_size)
        nn.init.uniform_(self.codebook_values.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, keys, values):
        # Encode keys and values
        encoded_keys = self.encoder_keys(keys)
        encoded_values = self.encoder_values(values)
        
        # Vector Quantization for keys
        keys_flat = encoded_keys.view(-1, self.hidden_key_size)
        key_embeddings = self.codebook_keys.weight
        key_indices = torch.argmin(torch.cdist(keys_flat, key_embeddings), dim=1)
        quantized_keys = self.codebook_keys(key_indices).view_as(encoded_keys)
        
        # Vector Quantization for values
        values_flat = encoded_values.view(-1, self.hidden_value_size)
        value_embeddings = self.codebook_values.weight
        value_indices = torch.argmin(torch.cdist(values_flat, value_embeddings), dim=1)
        quantized_values = self.codebook_values(value_indices).view_as(encoded_values)
        
        # Straight-Through Estimator for backpropagation
        quantized_keys = encoded_keys + (quantized_keys - encoded_keys).detach()
        quantized_values = encoded_values + (quantized_values - encoded_values).detach()
        
        # Decode quantized keys and values
        decoded_keys = self.decoder_keys(quantized_keys)
        decoded_values = self.decoder_values(quantized_values)
        
        return decoded_keys, decoded_values, encoded_keys, encoded_values, quantized_keys, quantized_values

    def compute_loss(self, keys, values, decoded_keys, decoded_values, quantized_keys, quantized_values):
        reconstruction_loss = F.mse_loss(decoded_keys, keys) + F.mse_loss(decoded_values, values)
        vq_loss = F.mse_loss(keys.detach(), quantized_keys) + F.mse_loss(values.detach(), quantized_values)
        commitment_loss = F.mse_loss(quantized_keys.detach(), keys) + F.mse_loss(quantized_values.detach(), values)
        total_loss = reconstruction_loss + vq_loss + self.alpha * commitment_loss
        return total_loss

class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr="add")

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2*emb_dim),
            torch.nn.BatchNorm1d(2*emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2*emb_dim, emb_dim)
        )
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) * x + self.propagate(
            edge_index, x=x, edge_attr=edge_embedding
        ))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

# GCN convolution along the graph structure


class GCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(
            edge_index, x=x, edge_attr=edge_embedding, norm=norm
        ) + F.relu(x + self.root_emb.weight) * 1. / deg.view(-1, 1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


# GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(
        self, num_layer, emb_dim, drop_ratio=0.5,
        JK="last", residual=False, gnn_type='gin'
    ):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers

        '''

        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        # add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        # List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            else:
                raise ValueError(
                    'Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data):
        x, edge_index = batched_data.x, batched_data.edge_index
        edge_attr, batch = batched_data.edge_attr, batched_data.batch
        # computing input node embedding

        h_list = [self.atom_encoder(x)]
        for layer in range(self.num_layer):

            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio,
                              training=self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        # Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation


# Virtual GNN to generate node embedding
class GNN_node_Virtualnode(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(
        self, num_layer, emb_dim, drop_ratio=0.5,
        JK="last", residual=False, gnn_type='gin'
    ):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GNN_node_Virtualnode, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        # add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        # set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        # List of GNNs
        self.convs = torch.nn.ModuleList()
        # batch norms applied to node embeddings
        self.batch_norms = torch.nn.ModuleList()

        # List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            else:
                raise ValueError(
                    'Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        for layer in range(num_layer - 1):
            self.mlp_virtualnode_list.append(torch.nn.Sequential(
                torch.nn.Linear(emb_dim, 2*emb_dim),
                torch.nn.BatchNorm1d(2*emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(2*emb_dim, emb_dim),
                torch.nn.BatchNorm1d(emb_dim),
                torch.nn.ReLU()
            ))

    def forward(self, batched_data):

        x, edge_index = batched_data.x, batched_data.edge_index
        edge_attr, batch = batched_data.edge_attr, batched_data.batch
        # virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding(torch.zeros(
            batch[-1].item() + 1
        ).to(edge_index.dtype).to(edge_index.device))

        h_list = [self.atom_encoder(x)]
        for layer in range(self.num_layer):
            # add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            # Message passing among graph nodes
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)

            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(
                    F.relu(h), self.drop_ratio,
                    training=self.training
                )

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

            # update the virtual nodes
            if layer < self.num_layer - 1:
                # add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = global_add_pool(
                    h_list[layer], batch) + virtualnode_embedding
                # transform virtual nodes using MLP

                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(
                        self.mlp_virtualnode_list[layer](
                            virtualnode_embedding_temp
                        ), self.drop_ratio, training=self.training
                    )
                else:
                    virtualnode_embedding = F.dropout(
                        self.mlp_virtualnode_list[layer](
                            virtualnode_embedding_temp
                        ), self.drop_ratio, training=self.training
                    )

        # Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation

class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, adj, device=torch.device("cpu:0")):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        # adj = self.normalize(adj + np.eye(adj.shape[0]))
        adj = self.normalize((adj + torch.eye(adj.shape[0], device=adj.device)).cpu().numpy())

        self.adj = torch.FloatTensor(adj).to(device)
        self.x = torch.eye(voc_size).to(device)

        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)

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
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx
    
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
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )

class GNNGraph(torch.nn.Module):

    def __init__(
        self, num_layer=5, emb_dim=300,
        gnn_type='gin', virtual_node=True, residual=False,
        drop_ratio=0.5, JK="last", graph_pooling="mean"
    ):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNNGraph, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        # GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(
                num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio,
                residual=residual, gnn_type=gnn_type
            )
        else:
            self.gnn_node = GNN_node(
                num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio,
                residual=residual, gnn_type=gnn_type
            )

        # Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=torch.nn.Sequential(
                torch.nn.Linear(emb_dim, 2 * emb_dim),
                torch.nn.BatchNorm1d(2 * emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(2*emb_dim, 1)
            ))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)
        return h_graph

class GNNGraph_CGIB(torch.nn.Module):

    def __init__(
        self,device, num_layer=5, emb_dim=300,
        gnn_type='gin', virtual_node=True, residual=False,
        drop_ratio=0.5, JK="last", graph_pooling="mean"
    ):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNNGraph_CGIB, self).__init__()
        self.device = device
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.graph_pooling = graph_pooling
        self.mse_loss = torch.nn.MSELoss()

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        # GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(
                num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio,
                residual=residual, gnn_type=gnn_type
            )
        else:
            self.gnn_node = GNN_node(
                num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio,
                residual=residual, gnn_type=gnn_type
            )
        self.compressor = nn.Sequential(
                    nn.Linear(self.emb_dim, self.emb_dim),
                    nn.BatchNorm1d(self.emb_dim),
                    nn.ReLU(),
                    nn.Linear(self.emb_dim, 1)
                    )
        self.patient_predictor = nn.Linear(self.emb_dim, 1)
        # Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=torch.nn.Sequential(
                torch.nn.Linear(emb_dim, 2 * emb_dim),
                torch.nn.BatchNorm1d(2 * emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(2*emb_dim, 1)
            ))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

    def compress(self, drug_features):
        p = self.compressor(drug_features)
        temperature = 1.0
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * torch.rand(p.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(self.device)
        gate_inputs = (gate_inputs + p) / temperature
        gate_inputs = torch.sigmoid(gate_inputs).squeeze()
        return gate_inputs, p
    
    def forward(self,patient_repr,batched_data):
        drug = batched_data
        drug_features = self.gnn_node(drug) # (7122,dim)
        # patient_repr (1,64)
        lambda_pos, p = self.compress(drug_features)
        lambda_pos = lambda_pos.reshape(-1, 1)
        lambda_neg = 1 - lambda_pos
        
        sim_scores = torch.matmul(drug_features, patient_repr.T)  # 维度 (7122, 1)
        sim_weights = F.softmax(sim_scores, dim=0)  # 维度 (7122, 1)
        drug_features = drug_features * sim_weights  # 广播至 (7122, dim)
        drug_features = F.normalize(drug_features, dim = 1)
        
        # Get Stats
        preserve_rate = (torch.sigmoid(p) > 0.5).float().mean()
        static_drug_feature = drug_features.clone().detach()
        node_feature_mean = scatter_mean(static_drug_feature, drug.batch, dim = 0)[drug.batch]
        node_feature_std = scatter_std(static_drug_feature, drug.batch, dim = 0)[drug.batch]
        
        noisy_node_feature_mean = lambda_pos * drug_features + lambda_neg * node_feature_mean
        noisy_node_feature_std = lambda_neg * node_feature_std

        noisy_node_feature = noisy_node_feature_mean + torch.rand_like(noisy_node_feature_mean) * noisy_node_feature_std
        noisy_drug_subgraphs = self.pool(noisy_node_feature, drug.batch) #(283,dim)

        epsilon = 1e-7

        KL_tensor = 0.5 * scatter_add(((noisy_node_feature_std ** 2) / (node_feature_std + epsilon) ** 2).mean(dim = 1), drug.batch).reshape(-1, 1) + \
                    scatter_add((((noisy_node_feature_mean - node_feature_mean)/(node_feature_std + epsilon)) ** 2), drug.batch, dim = 0)
        KL_Loss = torch.mean(KL_tensor)
        
        patient_pred_loss = self.mse_loss(patient_repr, self.patient_predictor(noisy_drug_subgraphs))
        return noisy_drug_subgraphs,KL_Loss,preserve_rate,patient_pred_loss


class GNN(torch.nn.Module):
    def __init__(
        self, num_tasks, num_layer=5, emb_dim=300,
        gnn_type='gin', virtual_node=True, residual=False,
        drop_ratio=0.5, JK="last", graph_pooling="mean"
    ):
        super(GNN, self).__init__()
        self.model = GNNGraph(
            num_layer, emb_dim, gnn_type,
            virtual_node, residual, drop_ratio, JK, graph_pooling
        )
        self.num_tasks = num_tasks
        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(
                2 * self.model.emb_dim, self.num_tasks
            )
        else:
            self.graph_pred_linear = torch.nn.Linear(
                self.model.emb_dim, self.num_tasks
            )

    def forward(self, batched_data):
        h_graph = self.model(batched_data)
        return self.graph_pred_linear(h_graph)


class SubRecOriginal(BaseModel, nn.Module):
    
    def __init__(
        self, 
        vocab_size, 
        device,
        ddi_adj,
        cfg: SubRecOriginalConfig = SubRecOriginalConfig()
    ):
        super(SubRecOriginal, self).__init__()
        nn.Module.__init__(self)
        
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.device = device
        self.ddi_adj = ddi_adj
        self.tensor_ddi_adj = torch.Tensor(ddi_adj).to(device)
        
        self.global_encoder = GNNGraph(
            num_layer=cfg.gnn_num_layer,
            emb_dim=cfg.gnn_embedding_dim,
            gnn_type=cfg.gnn_type,
            virtual_node=cfg.gnn_virtual_node,
            residual=cfg.gnn_residual,
            drop_ratio=cfg.gnn_drop_ratio,
            JK=cfg.gnn_JK,
            graph_pooling=cfg.gnn_graph_pooling
        )
        self.cgib = GNNGraph_CGIB(
            device=device,
            num_layer=cfg.gnn_num_layer,
            emb_dim=cfg.gnn_embedding_dim,
            gnn_type=cfg.gnn_type,
            virtual_node=cfg.gnn_virtual_node,
            residual=cfg.gnn_residual,
            drop_ratio=cfg.gnn_drop_ratio,
            JK=cfg.gnn_JK,
            graph_pooling=cfg.gnn_graph_pooling,
        )
        self.inter = nn.Parameter(torch.FloatTensor(1))
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size[0], cfg.embedding_dim),
            nn.Embedding(vocab_size[1], cfg.embedding_dim)
        ])
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=cfg.embedding_dim, nhead=cfg.n_heads)
        self.seq_encoders = nn.ModuleList([
            nn.TransformerEncoder(transformer_encoder_layer, num_layers=cfg.n_layers),
            nn.TransformerEncoder(transformer_encoder_layer, num_layers=cfg.n_layers)
        ])
        
        if cfg.dropout_rate > 0 and cfg.dropout_rate < 1:
            self.rnn_dropout = nn.Dropout(p=cfg.dropout_rate)
        else:
            self.rnn_dropout = nn.Sequential()
        
        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(cfg.embedding_dim * 2, cfg.embedding_dim)
        )
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(cfg.embedding_dim * 3, cfg.embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(cfg.embedding_dim * 2, vocab_size[2])
        )
        
        self.vqvae = VQVAE(
            input_key_dim=cfg.embedding_dim,
            input_value_dim=vocab_size[2],
            codebook_size=cfg.codebook_size,
            hidden_size=cfg.embedding_dim,
            alpha=cfg.alpha,
        )
        self.ddi_gcn = GCN(
            voc_size=vocab_size[2],
            emb_dim=cfg.embedding_dim,
            adj=self.tensor_ddi_adj,
            device=device
        )

        self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)
        self.inter.data.uniform_(-initrange, initrange)

    def forward(
        self, 
        patient_data, 
        mol_data, 
        average_projection, 
        bottleneck = False
    ):
        seq1, seq2 = [], []
        for adm in patient_data:
            Idx1 = torch.LongTensor([adm[0]]).to(self.device)
            Idx2 = torch.LongTensor([adm[1]]).to(self.device)
            repr1 = self.rnn_dropout(self.embeddings[0](Idx1))
            repr2 = self.rnn_dropout(self.embeddings[1](Idx2))
            seq1.append(torch.sum(repr1, keepdim=True, dim=1))
            seq2.append(torch.sum(repr2, keepdim=True, dim=1))
        seq1 = torch.cat(seq1, dim=1)
        seq2 = torch.cat(seq2, dim=1)
        seq1 = seq1.permute(1, 0, 2)
        seq2 = seq2.permute(1, 0, 2)
        output1 = self.seq_encoders[0](seq1)
        output2 = self.seq_encoders[1](seq2)
        output1 = output1.permute(1, 0, 2) 
        output2 = output2.permute(1, 0, 2)
        patient_reprs = torch.cat([output1,output2], dim=-1).squeeze(dim=0)
        queries = self.query(patient_reprs)
        query = queries[-1:]  # (1,dim)
        if bottleneck:
            global_embeddings,KL_Loss,preserve_rate,patient_pred_loss = self.cgib(query,mol_data) #(283,dim) (283 is smile type num)
        else :
            global_embeddings = self.global_encoder(mol_data)
        global_embeddings = torch.mm(average_projection, global_embeddings) #(131,dim)
        molecule_embeddings = global_embeddings 
        # Initialize vq_loss
        vq_loss = 0.0
        if len(patient_data) > 1 and not bottleneck:
            history_keys, history_values = [], []
            for idx, adm in enumerate(patient_data[:-1]):
                history_keys.append(queries[idx:idx+1].detach())
                history_value = torch.zeros(self.vocab_size[2]).to(self.device)
                history_value[adm[2]] = 1
                history_values.append(history_value.unsqueeze(0))
            
            history_keys = torch.cat(history_keys, dim=0)
            history_values = torch.cat(history_values, dim=0)
            
            decoded_keys, decoded_values, encoded_keys, encoded_values, quantized_keys, quantized_values = self.vqvae(history_keys, history_values)
            vq_loss = self.vqvae.compute_loss(history_keys, history_values, decoded_keys, decoded_values, quantized_keys, quantized_values)
        else:
            quantized_keys = self.vqvae.codebook_keys.weight  
            quantized_values = self.vqvae.codebook_values.weight 
        key_weights1 = F.softmax(torch.mm(query, molecule_embeddings.t()), dim=-1) 
        fact1 = torch.mm(key_weights1, molecule_embeddings) 
        if quantized_keys.size(0) > 0:
            visit_weight = F.softmax(torch.mm(query, quantized_keys.t()))
            weighted_values = visit_weight.mm(quantized_values)
            fact2 = torch.mm(weighted_values, molecule_embeddings)
        else:
            fact2 = fact1
            
        output_embedding = torch.cat([query, fact1, fact2], dim=-1)
        score = self.output(output_embedding)
        neg_pred_prob = torch.sigmoid(score)
        neg_pred_prob = torch.matmul(neg_pred_prob.t(), neg_pred_prob)
        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()
        if bottleneck:
            return score, batch_neg, KL_Loss, patient_pred_loss, preserve_rate
        else:
            return score, batch_neg, vq_loss 
    
    def fit(self, train_samples, mol_data, average_projection, val_samples=None):
        training_start_time = time.time()
        
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        logger.info(f"Set random seed to {self.cfg.seed}")
        
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
                result, ddi_loss, vq_loss = self(visit_history, mol_data, average_projection)  # (1, num_medications)
                result_cgib, ddi_loss_cgib, KL_loss, patient_pred_loss, preserve_rate = self(visit_history, mol_data, average_projection, bottleneck=True)
                
                # Prepare target as multi-hot vector
                target_multi_hot = self._multi_hot(
                    target_medications, 
                    self.vocab_size[2],  # medication vocabulary size
                    self.device
                ).unsqueeze(0)  # (1, num_medications)

                # Compute losses
                bce_loss = bce_loss_fn(result, target_multi_hot)
                bce_loss_cgib = bce_loss_fn(result_cgib, target_multi_hot)
                
                # Combined loss: 90% BCE + weighted DDI penalty
                total_loss = 0.95 * bce_loss + + 0.95 * bce_loss_cgib + self.cfg.ddi_lambda * ddi_loss
                total_loss += self.cfg.mu * KL_loss
                total_loss += self.cfg.mu * patient_pred_loss
                total_loss += self.cfg.gamma * vq_loss

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
                    mol_data=mol_data,
                    average_projection=average_projection,
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
        mol_data,
        average_projection,
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
        medication_logits, _, _= self(prefix, mol_data, average_projection)  # (1, num_medications)
        
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
        mol_data, 
        average_projection,
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
            predicted_medications = self.predict(visit_history, mol_data=mol_data, average_projection=average_projection, threshold=threshold)
            
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
    
    def load_model(self, path, weights_only=True):
        checkpoint = torch.load(path, weights_only=weights_only)
        self.load_state_dict(checkpoint["model_state_dict"])
from collections import defaultdict
from typing import Any, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import time
import numpy as np

from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from dataclasses import dataclass
from torch_geometric.utils import degree
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from pathlib import Path

from src.models.BaseModel import BaseModel
from src.utils.logging import get_logger
from src.utils.metrics import ddi_rate_score, ddi_rate_single_multihot

logger = get_logger("Original MoleRec")

@dataclass
class MoleRecConfig():
    seed: int = 42
    epochs: int = 50
    lr: float = 5e-4
    target_ddi: float = 0.06
    coef: float = 2.5
    use_embedding: bool = True
    emb_dim: int = 64
    model_dim: int = 64
    dropout: float = 0.7
    
    save_dir: str = "saved/MoleRecOG"
    ckpt_name: str = "best_model" + str(int(time.time()))
    
    substruct_dim: int = 64
    global_dim: int = 64
    substruct_num: int = 460
    
    substruct_num_layers: int = 4
    substruct_emb_dim: int = 64
    substruct_gnn_type: str = "gin"
    substruct_virtual_node: bool = False
    substruct_residual: bool = False
    substruct_dropout_ratio: float = 0.7
    substruct_jk_type: str = "last"
    substruct_graph_pooling: str = "mean"
    
    global_num_layers: int = 4
    global_emb_dim: int = 64
    global_gnn_type: str = "gin"
    global_virtual_node: bool = False
    global_residual: bool = False
    global_dropout_ratio: float = 0.7
    global_jk_type: str = "last"
    global_graph_pooling: str = "mean"


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
        edge_attr, _ = batched_data.edge_attr, batched_data.batch
        # computing input node embedding

        h_list = [self.atom_encoder(x)]
        for layer in range(self.num_layer):

            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

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

class GNNGraph(nn.Module):
    
    def __init__(self, num_layers=5, emb_dim=300, gnn_type="gin", virtual_node=True, residual=False, drop_ratio=0.5, JK="last", graph_pooling="mean"):
        super(GNNGraph, self).__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.graph_pooling = graph_pooling
        
        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layers, emb_dim, JK=JK, drop_ratio=drop_ratio, residual=residual, gnn_type=gnn_type)
        else:
            self.gnn_node = GNN_node(num_layers, emb_dim, JK=JK, drop_ratio=drop_ratio, residual=residual, gnn_type=gnn_type)
        
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

class GNN(nn.Module):
    def __init__(
        self, num_tasks, num_layers=5, emb_dim=300,
        gnn_type='gin', virtual_node=True, residual=False,
        drop_ratio=0.5, JK="last", graph_pooling="mean"
    ):
        super(GNN, self).__init__()
        self.model = GNNGraph(
            num_layers, emb_dim, gnn_type,
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

class MAB(torch.nn.Module):
    def __init__(
        self, Qdim, Kdim, Vdim, number_heads,
        use_ln=False, *args, **kwargs
    ):
        super(MAB, self).__init__(*args, **kwargs)
        self.Vdim = Vdim
        self.number_heads = number_heads

        assert self.Vdim % self.number_heads == 0, \
            'the dim of features should be divisible by number_heads'

        self.Qdense = torch.nn.Linear(Qdim, self.Vdim)
        self.Kdense = torch.nn.Linear(Kdim, self.Vdim)
        self.Vdense = torch.nn.Linear(Kdim, self.Vdim)
        self.Odense = torch.nn.Linear(self.Vdim, self.Vdim)

        self.use_ln = use_ln
        if self.use_ln:
            self.ln1 = torch.nn.LayerNorm(self.Vdim)
            self.ln2 = torch.nn.LayerNorm(self.Vdim)

    def forward(self, X, Y):
        Q, K, V = self.Qdense(X), self.Kdense(Y), self.Vdense(Y)
        batch_size, dim_split = Q.shape[0], self.Vdim // self.number_heads

        Q_split = torch.cat(Q.split(dim_split, 2), 0)
        K_split = torch.cat(K.split(dim_split, 2), 0)
        V_split = torch.cat(V.split(dim_split, 2), 0)

        Attn = torch.matmul(Q_split, K_split.transpose(1, 2))
        Attn = torch.softmax(Attn / math.sqrt(dim_split), dim=-1)
        O = Q_split + torch.matmul(Attn, V_split)
        O = torch.cat(O.split(batch_size, 0), 2)

        O = O if not self.use_ln else self.ln1(O)
        O = self.Odense(O)
        O = O if not self.use_ln else self.ln2(O)

        return O


class SAB(torch.nn.Module):
    def __init__(
        self, in_dim, out_dim, number_heads,
        use_ln=False, *args, **kwargs
    ):
        super(SAB, self).__init__(*args, **kwargs)
        self.net = MAB(in_dim, in_dim, out_dim, number_heads, use_ln)

    def forward(self, X):
        return self.net(X, X)

class AdjAttenAgger(torch.nn.Module):
    def __init__(self, Qdim, Kdim, mid_dim, *args, **kwargs):
        super(AdjAttenAgger, self).__init__(*args, **kwargs)
        self.model_dim = mid_dim
        self.Qdense = torch.nn.Linear(Qdim, mid_dim)
        self.Kdense = torch.nn.Linear(Kdim, mid_dim)
        # self.use_ln = use_ln

    def forward(self, main_feat, other_feat, fix_feat, mask=None):
        Q = self.Qdense(main_feat)
        K = self.Kdense(other_feat)
        Attn = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(self.model_dim)

        if mask is not None:
            Attn = torch.masked_fill(Attn, mask, -(1 << 32))

        Attn = torch.softmax(Attn, dim=-1)
        # print(Attn[0])
        # print(mask[0])
        fix_feat = torch.diag(fix_feat)
        other_feat = torch.matmul(fix_feat, other_feat)
        O = torch.matmul(Attn, other_feat)

        return O


class OriginalMoleRec(BaseModel, nn.Module):
    def __init__(
        self,
        cfg: MoleRecConfig,
        ddi_adj,
        voc_size,
        molecule_graph,
        average_projection,
        substruct_data,
        ddi_mask_H,
        device=torch.device("cpu"),
    ):
        super().__init__()
        nn.Module.__init__(self)
        
        self.cfg = cfg
        self.device = device
        self.ddi_adj = ddi_adj
        self.vocab_size = voc_size
        self.tensor_ddi_adj = torch.from_numpy(ddi_adj).to(device)
                
        self.molecule_graph = molecule_graph.to(device)
        self.average_projection = average_projection.to(device)
        self.substruct_data = substruct_data.to(device)
        self.ddi_mask_H = ddi_mask_H.to(device)
        
        self.use_embedding = cfg.use_embedding
        
        if self.use_embedding:
            self.substruct_emb = torch.nn.Parameter(
                torch.zeros(cfg.substruct_num, cfg.emb_dim)
            )
        else:
            self.substruct_encoder = GNNGraph(
                num_layers=cfg.substruct_num_layers,
                emb_dim=cfg.substruct_emb_dim,
                gnn_type=cfg.substruct_gnn_type,
                virtual_node=cfg.substruct_virtual_node,
                residual=cfg.substruct_residual,
                drop_ratio=cfg.substruct_dropout_ratio,
                JK=cfg.substruct_jk_type,
                graph_pooling=cfg.substruct_graph_pooling
            )

        self.global_encoder = GNNGraph(
            num_layers=cfg.global_num_layers,
            emb_dim=cfg.global_emb_dim,
            gnn_type=cfg.global_gnn_type,
            virtual_node=cfg.global_virtual_node,
            residual=cfg.global_residual,
            drop_ratio=cfg.global_dropout_ratio,
            JK=cfg.global_jk_type,
            graph_pooling=cfg.global_graph_pooling
        )

        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(voc_size[0], cfg.emb_dim),
            torch.nn.Embedding(voc_size[1], cfg.emb_dim)
        ])
        self.seq_encoders = torch.nn.ModuleList([
            torch.nn.GRU(cfg.emb_dim, cfg.emb_dim, batch_first=True),
            torch.nn.GRU(cfg.emb_dim, cfg.emb_dim, batch_first=True)
        ])
        if cfg.dropout > 0 and cfg.dropout < 1:
            self.rnn_dropout = torch.nn.Dropout(p=cfg.dropout)
        else:
            self.rnn_dropout = torch.nn.Sequential()
        self.sab = SAB(cfg.substruct_dim, cfg.substruct_dim, 2, use_ln=True)
        self.query = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(cfg.emb_dim * 4, cfg.emb_dim)
        )
        self.substruct_rela = torch.nn.Linear(cfg.emb_dim, cfg.substruct_num)
        self.aggregator = AdjAttenAgger(
            cfg.global_dim, cfg.substruct_dim, max(cfg.global_dim, cfg.substruct_dim)
        )
        score_extractor = [
            torch.nn.Linear(cfg.substruct_dim, cfg.substruct_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(cfg.substruct_dim // 2, 1)
        ]
        self.score_extractor = torch.nn.Sequential(*score_extractor)
        self.init_weights()
            
    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)
        if self.use_embedding:
            torch.nn.init.xavier_uniform_(self.substruct_emb)
        
    def forward(
        self, patient_data
    ):
        diagnosis_seq, procedure_seq = [], []
        
        for adm in patient_data:
            diagnoses_codes = torch.LongTensor([adm[0]]).to(self.device)
            diagnoses_representation = self.rnn_dropout(self.embeddings[0](diagnoses_codes))
            diagnosis_seq.append(torch.sum(diagnoses_representation, keepdim=True, dim=1))
            
            procedure_codes = torch.LongTensor([adm[1]]).to(self.device)
            procedure_representation = self.rnn_dropout(self.embeddings[1](procedure_codes))
            procedure_seq.append(torch.sum(procedure_representation, keepdim=True, dim=1))
        
        diagnoses_seq = torch.cat(diagnosis_seq, dim=1)
        procedure_seq = torch.cat(procedure_seq, dim=1)
        diagnoses_output, diagnoses_hidden = self.seq_encoders[0](diagnoses_seq)
        procedures_output, procedures_hidden = self.seq_encoders[1](procedure_seq)
        
        seq_repr = torch.cat([diagnoses_hidden, procedures_hidden], dim=-1)
        last_repr = torch.cat([diagnoses_output[:, -1],  procedures_output[:, -1]], dim=-1)
        patient_repr = torch.cat([seq_repr.flatten(), last_repr.flatten()])

        query = self.query(patient_repr)
        substruct_weight = torch.sigmoid(self.substruct_rela(query))

        global_embeddings = self.global_encoder(self.molecule_graph)
        global_embeddings = torch.mm(self.average_projection, global_embeddings)
        substruct_embeddings = self.sab(
            self.substruct_emb.unsqueeze(0) if self.use_embedding else
            self.substruct_encoder(self.substruct_data).unsqueeze(0)
        ).squeeze(0)
        molecule_embeddings = self.aggregator(
            global_embeddings, substruct_embeddings,
            substruct_weight, mask=torch.logical_not(self.ddi_mask_H > 0)
        )

        score = self.score_extractor(molecule_embeddings).t()

        neg_pred_prob = torch.sigmoid(score)
        neg_pred_prob = torch.matmul(neg_pred_prob.t(), neg_pred_prob)
        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()
        return score, batch_neg
    
    def fit(self, train_samples, val_samples):
        training_start_time = time.time()
        
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        logger.info(f"Set random seed to {self.cfg.seed}")
        
        self.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)
        bce_loss_fn = nn.BCEWithLogitsLoss()
        mml_loss_fn = nn.MultiLabelMarginLoss()
        
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
        
        threshold = getattr(self.cfg, "threshold", 0.5)
        ignore_ids = getattr(self.cfg, "ignore_ids", [0, 1])
        
        save_dir = Path(self.cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = save_dir / self.cfg.ckpt_name
        logger.info(f"Checkpoints will be saved to: {ckpt_path}")
        
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
                
                target_multi_hot = self._multi_hot(
                    target_medications, 
                    self.vocab_size[2], 
                    self.device
                ).unsqueeze(0)
                
                # N, C = medication_logits.shape
                
                # target = torch.full((N, C), -1, dtype=torch.long, device=self.device)

                # for i, labels in enumerate(target_medications):
                #     labels = labels[:C]  # safety
                #     if len(labels) > 0:
                #         target[i, :len(labels)] = torch.tensor(labels, dtype=torch.long, device=self.device)
                
                medication_probs = torch.sigmoid(medication_logits)
                
                bce_loss = bce_loss_fn(medication_logits, target_multi_hot)
                # loss_multi = mml_loss_fn(medication_probs, target)
                
                predicted_indices = (medication_probs >= threshold)
                
                current_ddi_score = ddi_rate_single_multihot(predicted_indices, self.tensor_ddi_adj)
                
                if current_ddi_score < self.cfg.target_ddi:
                    loss = 0.95 * bce_loss
                else:
                    beta = self.cfg.coef * (1 - (current_ddi_score / self.cfg.target_ddi))
                    beta = min(math.exp(beta), 1)
                    loss = beta * (0.95 * bce_loss) + (1 - beta) * ddi_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
                epoch_loss_sum += float(loss.item())
                num_batches += 1
                global_step += 1
            
            avg_train_loss = epoch_loss_sum / max(num_batches, 1)
            epoch_time = time.time() - epoch_start_time
            history["train_loss"].append(avg_train_loss)

            logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s | Train Loss={avg_train_loss:.4f}")
            
            if val_samples is not None:
                self._eval_val_metrics(
                    val_samples, 
                    threshold=threshold, 
                    ignore_ids=ignore_ids,
                    epoch=epoch
                )
        total_training_time = time.time() - training_start_time
        history["training_time"] = total_training_time
        
        logger.info(f"Training completed in {total_training_time:.2f}s ({total_training_time/60:.2f} minutes)")
        
        return history
    
    def predict(self, X):
        return super().predict(X)
    
    def _eval_val_metrics(self, val_samples, threshold, ignore_ids, epoch):
        val_start_time = time.time()
        self.eval()
        jaccard = []
        batch_precisions = []
        batch_recalls = []
        batch_f1_scores = []
        for visit in val_samples:
            visit_history, target_medications = visit
            target = self._multi_hot(target_medications, self.vocab_size[2], self.device)
            medication_logits, ddi_loss = self(visit_history)
            medication_probs = torch.sigmoid(medication_logits).squeeze(1)
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
import torch

import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from torch_geometric.utils import degree
from torch.nn.modules.transformer import TransformerEncoderLayer
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.nn import global_max_pool, GlobalAttention, Set2Set, global_add_pool, global_mean_pool, MessagePassing

from src.models.BaseModel import BaseModel

class DrugDoctorOriginalConfig:
    seed: int = 42
    
    embedding_dim: int = 128
    dropout_rate: float = 0.3
    
    transformer_dropout_rate: float = 0.2
    n_head: int = 4
    
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

        h_graph = self.pool(h_node, batched_data['batched_data'].batch)
        return h_graph
        # return self.graph_pred_linear(h_graph)


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
        x, edge_index = batched_data['batched_data'].x, batched_data['batched_data'].edge_index
        edge_attr, batch = batched_data['batched_data'].edge_attr, batched_data['batched_data'].batch
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


# GIN convolution along the graph structure


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

class DrugDoctorOriginal(BaseModel, nn.Module):
    def __init__(
        self, 
        vocab_size,
        device,
        ddi_adj, 
        cfg: DrugDoctorOriginalConfig = DrugDoctorOriginalConfig()
    ):
        super(DrugDoctorOriginal, self).__init__()
        nn.Module.__init__(self)
        
        self.cfg = cfg
        self.device = device
        self.vocab_size = vocab_size,
        self.ddi_adj = ddi_adj
        
        self.diag_embedding = nn.Sequential(
            nn.Embedding(vocab_size[0], embedding_dim=cfg.embedding_dim, padding_idx=0),
            nn.Dropout(p=cfg.dropout_rate)
        )
        self.proc_embedding = nn.Sequential(
            nn.Embedding(vocab_size[1], embedding_dim=cfg.embedding_dim, padding_idx=0),
            nn.Dropout(p=cfg.dropout_rate)
        )
        self.used_med_embedding = nn.Sequential(
            nn.Embedding(vocab_size[2], embedding_dim=cfg.embedding_dim, padding_idx=0),
            nn.Dropout(p=cfg.dropout_rate)
        )
        
        self.diagnoses_encoder = TransformerEncoderLayer(cfg.embedding_dim, cfg.n_head, dropout=cfg.transformer_dropout_rate, batch_first=True)
        self.procedures_encoder = TransformerEncoderLayer(cfg.embedding_dim, cfg.n_head, dropout=cfg.transformer_dropout_rate, batch_first=True)
        
        self.crossAttention1 = nn.MultiheadAttention(cfg.embedding_dim, 1, dropout=cfg.transformer_dropout_rate, batch_first=True)
        self.crossAttention2 = nn.MultiheadAttention(cfg.embedding_dim, 1, dropout=cfg.transformer_dropout_rate, batch_first=True)
        self.crossAttention3 = nn.MultiheadAttention(cfg.embedding_dim * 2, 2, dropout=cfg.transformer_dropout_rate, batch_first=True)
        self.crossAttention4 = nn.MultiheadAttention(cfg.embedding_dim, 1, dropout=cfg.transformer_dropout_rate, batch_first=True)
        
        self.crossAttention5 = nn.MultiheadAttention(cfg.embedding_dim, 1, dropout=cfg.transformer_dropout_rate, batch_first=True)
        self.crossAttention6 = nn.MultiheadAttention(cfg.embedding_dim * 3, 1, dropout=cfg.transformer_dropout_rate, batch_first=True)
        
        self.encoders = nn.Sequential(
            nn.GRU(self.vocab_size[2], self.vocab_size[2], batch_first=True)
        )
        
        self.substruct_encoder = GNNGraph()
        
        self.output_layer1 = nn.Linear(cfg.embedding_dim * 3, vocab_size[2])
        self.output_layer2 = nn.Linear(cfg.embedding_dim * 4, vocab_size[2])
        
    def sum_embedding(self, embedding):
        return embedding.sum(dim=1).unsqueeze(dim=0)
    
    def used_med_learning(self, used_medications):
        batch_size = len(used_medications)
        results = []
        for i in range(batch_size):
            if len(used_medications[i]) != 0:
                used_med_emb = self.used_med_embedding(torch.tensor(used_medications[i]).to(self.device)).sum(dim=1).unsqueeze(dim=0)
                results.append(self.encoders(used_med_emb)[0][:, -1, :])
            else:
                results.append([])
        return results
    
def forward(self, diagnose, procedures,used_medications,used_diag,used_proc,substruct_data,d_mask_matrix, p_mask_matrix):

        
        batch_size = diagnose.shape[0]
        max_diag_num = diagnose.shape[1]
        max_proc_num = procedures.shape[1]
        
        diag_emb = self.diag_embedding(diagnose) # SHAPECOGNet：[batch_size, max_visit_length, max_diag_num, emb]
        proc_emb = self.proc_embedding(procedures) # batch_size, max_diag_num, emb]
        
        d_enc_mask_matrix = d_mask_matrix.unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, self.nhead, max_diag_num,1) 
        d_enc_mask_matrix = d_enc_mask_matrix.view(batch_size * self.nhead, max_diag_num, max_diag_num)
        p_enc_mask_matrix = p_mask_matrix.unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, self.nhead, max_proc_num,1)
        p_enc_mask_matrix = p_enc_mask_matrix.view(batch_size * self.nhead, max_proc_num, max_proc_num)
        diagnose_embdding = self.diagnoses_encoder(diag_emb, src_mask=d_enc_mask_matrix) 
        procedure_embedding = self.procedure_encoder(proc_emb, src_mask=p_enc_mask_matrix)
        diag_enc = torch.sum(diagnose_embdding, dim=1)
        proc_enc = torch.sum(procedure_embedding, dim=1)


        
        visit_enc = torch.cat([diag_enc, proc_enc], dim=-1) # [batch_size, 2*emb]          
        substruct_information = self.substruct_encoder(substruct_data['substruct_data'])# torch.Size([491, 112])
        embedding1 = self.crosssAtte1(substruct_information.unsqueeze(dim=0).repeat(batch_size, 1, 1),diagnose_embdding,diagnose_embdding)[0] #torch.Size([batch_size, 491, 112])
        embedding2 = self.crosssAtte2(substruct_information.unsqueeze(dim=0).repeat(batch_size, 1, 1),procedure_embedding,procedure_embedding)[0]
        diag_enc_cross = torch.sum(embedding1, dim=1)
        proc_enc_cross = torch.sum(embedding2, dim=1)
        visit_enc_cross = torch.cat([diag_enc_cross, proc_enc_cross], dim=-1) # [batch_size, 2*emb]

        embedding3 = self.crosssAtte3(visit_enc_cross.unsqueeze(dim=1),visit_enc_cross.unsqueeze(dim=1),visit_enc_cross.unsqueeze(dim=1))[0] # [batch_size, 1, 2*emb]
        
        visit_enc_cross = torch.sum(embedding3, dim=1)# [batch_size, 2*emb]


        result_visit_enc = torch.cat([visit_enc, visit_enc_cross], dim=-1) # [batch_size, 4*emb]
        decoder_output = self.output_layer2(result_visit_enc) # [batch_size, vocab_size[2]]       
        
        
        
        last_med_pred = []
        for i in range(batch_size):
            if len(used_diag[i])!=0:
                used_visit_num = len(used_diag[i])
                j = used_visit_num-1
                
                usedDiag = used_diag[i][j] # len:15
                usedProc = used_proc[i][j] # len:6
                usedMed = used_medications[i][j]  # len:112
                
                usedDiagEmbedding = self.diagnoses_encoder(self.diag_embedding(torch.tensor(usedDiag).to(self.device)).unsqueeze(dim=0)) # Transformer torch.Size 
                usedProcEmbedding = self.procedure_encoder(self.proc_embedding(torch.tensor(usedProc).to(self.device)).unsqueeze(dim=0)) #torch.Size([1, 6, 112])
                # used_diag_enc = torch.sum(usedDiagEmbedding, dim=1)# torch.Size([1, 112])
                # used_proc_enc = torch.sum(usedProcEmbedding, dim=1)# torch.Size([1, 112])
                
                
                usedMedembedding = self.used_med_embedding(torch.tensor(usedMed).to(self.device)).sum(dim=1).unsqueeze(dim=0) # torch.Size([1, 112])
                last_embed1 = self.crosssAtte4(usedMedembedding.unsqueeze(dim=0),usedDiagEmbedding,usedDiagEmbedding)[0] # shape:[1, 1, emb]
                last_embed2 = self.crosssAtte5(usedMedembedding.unsqueeze(dim=0),usedProcEmbedding,usedProcEmbedding)[0] # [1, 1, emb]
                used_diag_enc = torch.sum(last_embed1, dim=1)
                used_proc_enc = torch.sum(last_embed2, dim=1)
                
                
                used_visit_enc = torch.cat([used_diag_enc, used_proc_enc, usedMedembedding], dim=-1) # [1, 3*emb]
                used_embedding3 = self.crosssAtte6(used_visit_enc.unsqueeze(dim=1),used_visit_enc.unsqueeze(dim=1),used_visit_enc.unsqueeze(dim=1))[0] # [1, 1, 3*emb]
                used_visit_enc = torch.sum(used_embedding3, dim=1)# [1, 3*emb]
                last_decoder_output = self.output_layer1(used_visit_enc) # [1, vocab_size[2]]
                last_med_pred.append(last_decoder_output)
            else:
                last_med_pred.append([])

        


        results_from_history = self.used_med_learning(used_medications)

        for i in range(batch_size):
            if len(results_from_history[i])!=0:
                decoder_output[i] = decoder_output[i] + results_from_history[i][0] +last_med_pred[i][0] 

        sigmoid_output = F.sigmoid(decoder_output)
        sigmoid_output_ddi = torch.matmul(sigmoid_output.unsqueeze(1).transpose(-1, -2),sigmoid_output.unsqueeze(1)) #[batch_size,vocab_size[2],vocab_size[2]]
        kg_ddi = torch.from_numpy(self.ddi_adj).to(sigmoid_output.device).unsqueeze(0).repeat(batch_size, 1, 1) #[batch_size,vocab_size[2],vocab_size[2]]
        kg_ddi_score = 0.001 *  self.kgloss_alpha * torch.sum(kg_ddi * sigmoid_output_ddi, dim=[-1,-2]).mean()
        return decoder_output,kg_ddi_score
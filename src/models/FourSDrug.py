from torch.nn import Embedding, Linear
import torch.nn as nn
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
import math
import dill

from src.core.interfaces.basemodel import Model

class Attention(nn.Module):
    def __init__(self, embed_dim=64, output_dim=1):
        super(Attention, self).__init__()
        self.embedding_dim, self.output_dim = embed_dim, output_dim
        self.aggregation = Linear(self.embedding_dim, self.output_dim)

    def _aggregate(self, x):
        weight = self.aggregation(x)  # [b, num_learn, 1]
        return torch.tanh(weight)

    def forward(self, x, mask=None):
        device = x.device
        if mask is None:
            weight = torch.softmax(self._aggregate(x), dim=-2)
        else:
            mask = torch.where(mask == 0, torch.tensor(-1e7, device=device), torch.tensor(0.0, device=device))
            weight = torch.softmax(self._aggregate(x).squeeze(-1) + mask, dim=-1).float().unsqueeze(-1)
            weight = torch.where(torch.isnan(weight), torch.tensor(0.0, device=device), weight)
        agg_embeds = torch.matmul(x.transpose(-1, -2).float(), weight).squeeze(-1)
        return agg_embeds


class FourSDrug(Model, nn.Module):

    def __init__(
        self,
        vocab_path,
        ddi_adj_path,
        sym_sets=None,
        drug_multihots=None,
        device=torch.device("cpu"),
        embed_dim=64,
        dropout=0.4,
    ):
        nn.Module.__init__(self)
        Model.__init__(self)
        self.device = torch.device(device)
        with open(vocab_path, 'rb') as f:
            vocab = dill.load(f)
        vocab_size = [len(vocab['diagnoses_vocab'].word_to_idx), len(vocab['procedures_vocab'].word_to_idx), len(vocab['medication_vocab'].word_to_idx)]
        self.n_sym, self.n_drug = vocab_size[0], vocab_size[2]
        self.embed_dim, self.dropout = embed_dim, dropout
        # self.sym_sets, self.drug_multihots = sym_sets, drug_multihots
        self.sym_embeddings = Embedding(self.n_sym, self.embed_dim)
        self.drug_embeddings = Embedding(self.n_drug, self.embed_dim)
        self.sym_agg = Attention(self.embed_dim)
        self.sym_counts = None
        ddi_adj = torch.tensor(dill.load(open(ddi_adj_path, 'rb'))).float()
        self.register_buffer("tensor_ddi_adj", ddi_adj.to(self.device))
        self.sparse_ddi_adj = csr_matrix(ddi_adj.detach().cpu().numpy())
        self.init_weights()
    
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.embed_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    
    def forward(self, syms, drugs, similar_idx):
        '''
        :param syms: [batch_size, sym_set_size]
        :param drugs: [batch_size, num_drugs]
        :param device: 'cpu' or 'gpu
        :param similar_idx: [batch_size]
        :return:
        '''

        current_device = next(self.parameters()).device
        self.device = current_device
        syms = syms.to(current_device)
        drugs = drugs.to(current_device)
        similar_idx = similar_idx.to(current_device)
        all_drugs = torch.arange(self.n_drug, device=current_device)
        sym_embeds, all_drug_embeds = self.sym_embeddings(syms.long()), self.drug_embeddings(all_drugs)
        s_set_embeds = self.sym_agg(sym_embeds)
        all_drug_embeds = all_drug_embeds.unsqueeze(0).repeat(s_set_embeds.shape[0], 1, 1)

        scores = torch.bmm(s_set_embeds.unsqueeze(1), all_drug_embeds.transpose(-1, -2)).squeeze(-2)  # [batch_size, n_drug]
        scores_aug, batch_neg = 0.0, 0.0

        neg_pred_prob = torch.sigmoid(scores)
        neg_pred_prob = torch.mm(neg_pred_prob.transpose(-1, -2), neg_pred_prob)  # (voc_size, voc_size)
        batch_neg = 0.00001 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

        if syms.shape[0] > 2 and syms.shape[1] > 2:
            scores_aug = self._intraset_augmentation(syms, drugs, all_drug_embeds, similar_idx)
            batch_neg += self._intersect_ddi(syms, s_set_embeds, drugs, all_drug_embeds, similar_idx)

        return scores, scores_aug, batch_neg

    def evaluate(self, syms, device=None):
        target_device = torch.device(device) if device is not None else next(self.parameters()).device
        self.device = target_device
        syms = syms.to(target_device)
        drug_ids = torch.arange(0, self.n_drug, device=target_device).long()
        sym_embeds, drug_embeds = self.sym_embeddings(syms.long()), self.drug_embeddings(drug_ids)
        s_set_embed = self.sym_agg(sym_embeds)
        scores = torch.mm(s_set_embed, drug_embeds.transpose(-1, -2)).squeeze(0)

        return scores

    def _intraset_augmentation(self, syms, drugs, all_drug_embeds, similar_idx):
        selected_drugs = drugs[similar_idx]
        r = torch.arange(drugs.shape[0], device=self.device).unsqueeze(1)
        sym_multihot = torch.zeros((drugs.shape[0], self.n_sym), device=self.device)
        selected_sym_multihot = torch.zeros((drugs.shape[0], self.n_sym), device=self.device)
        sym_multihot[r, syms], selected_sym_multihot[r, syms[similar_idx]] = 1, 1

        common_sym = sym_multihot * selected_sym_multihot
        common_sym_sq = common_sym.unsqueeze(-1).repeat(1, 1, self.embed_dim)
        all_sym_embeds = self.sym_embeddings(torch.arange(self.n_sym, device=self.device)).unsqueeze(0).expand_as(common_sym_sq)
        common_sym_embeds = common_sym_sq * all_sym_embeds
        common_set_embeds = self.sym_agg(common_sym_embeds, common_sym)
        common_drug, diff_drug = drugs * selected_drugs, drugs - selected_drugs
        diff_drug[diff_drug == -1] = 1

        common_drug_sum, diff_drug = torch.sum(common_drug, -1, True), torch.sum(diff_drug, -1, True)
        common_drug_sum[common_drug_sum == 0], diff_drug[diff_drug == 0] = 1, 1

        scores = torch.bmm(common_set_embeds.unsqueeze(1), all_drug_embeds.transpose(-1, -2)).squeeze(1)
        scores = F.binary_cross_entropy_with_logits(scores, common_drug)

        return scores

    def _intersect_ddi(self, syms, s_set_embed, drugs, all_drug_embeds, similar_idx):
        selected_drugs = drugs[similar_idx]
        r = torch.arange(drugs.shape[0], device=self.device).unsqueeze(1)
        sym_multihot = torch.zeros((drugs.shape[0], self.n_sym), device=self.device)
        selected_sym_multihot = torch.zeros((drugs.shape[0], self.n_sym), device=self.device)
        sym_multihot[r, syms], selected_sym_multihot[r, syms[similar_idx]] = 1, 1

        common_sym = sym_multihot * selected_sym_multihot
        common_sym_sq = common_sym.unsqueeze(-1).repeat(1, 1, self.embed_dim)
        all_sym_embeds = self.sym_embeddings(torch.arange(self.n_sym, device=self.device)).unsqueeze(0).expand_as(
            common_sym_sq)
        common_sym_embeds = common_sym_sq * all_sym_embeds
        common_set_embeds = self.sym_agg(common_sym_embeds, common_sym)
        diff_drug = drugs - selected_drugs
        diff_drug_2 = torch.zeros_like(diff_drug)
        diff_drug_2[diff_drug == -1], diff_drug[diff_drug == -1] = 1, 0

        diff_drug_exp, diff2_exp = diff_drug.unsqueeze(1), diff_drug_2.unsqueeze(1)
        diff_drug = torch.sum(diff_drug, -1, True)
        diff_drug_2 = torch.sum(diff_drug_2, -1, True)
        diff_drug[diff_drug == 0] = 1
        diff_drug_2[diff_drug_2 == 0] = 1
        diff_drug_embed = torch.bmm(diff_drug_exp.float(), all_drug_embeds).squeeze() / diff_drug
        diff2_embed = torch.bmm(diff2_exp.float(), all_drug_embeds).squeeze() / diff_drug_2

        diff_score = torch.sigmoid(common_set_embeds * diff_drug_embed.float())
        diff2_score = torch.sigmoid(common_set_embeds * diff2_embed.float())
        score_aug = 0.0001 * torch.sum(diff2_score * diff_score)

        return score_aug

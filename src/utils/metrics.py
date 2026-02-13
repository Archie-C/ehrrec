from typing import Iterable, Sequence, Dict, Optional, Set
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
import torch

def multi_label_metric(y_gt, y_pred, y_prob=None):
    
    def jaccard(y_gt, y_pred):
        if len(y_pred) ==0:
            return 0
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            union = set(out_list) | set(target)
            jaccard_score = 0 if union == 0 else len(inter) / len(union)
            score.append(jaccard_score)
        return np.mean(score)

    def average_prc(y_gt, y_pred):
        if len(y_pred) ==0:
            return [0]
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            prc_score = 0 if len(out_list) == 0 else len(inter) / len(out_list)
            score.append(prc_score)
        return score

    def average_recall(y_gt, y_pred):
        if len(y_pred) ==0:
            return [0]
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            recall_score = 0 if len(target) == 0 else len(inter) / len(target)
            score.append(recall_score)
        return score

    def average_f1(average_prc, average_recall):
        score = []
        for idx in range(len(average_prc)):
            if average_prc[idx] + average_recall[idx] == 0:
                score.append(0)
            else:
                score.append(2*average_prc[idx]*average_recall[idx] / (average_prc[idx] + average_recall[idx]))
        return score

    def f1(y_gt, y_pred):
        if len(y_pred) ==0:
            return 0
        all_micro = []
        for b in range(y_gt.shape[0]):
            all_micro.append(f1_score(y_gt[b], y_pred[b], average='macro'))
        return np.mean(all_micro)

    def roc_auc(y_gt, y_prob):
        
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(roc_auc_score(y_gt[b], y_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(average_precision_score(y_gt[b], y_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_at_k(y_gt, y_prob, k=3):
        precision = 0
        sort_index = np.argsort(y_prob, axis=-1)[:, ::-1][:, :k]
        for i in range(len(y_gt)):
            TP = 0
            for j in range(len(sort_index[i])):
                if y_gt[i, sort_index[i, j]] == 1:
                    TP += 1
            precision += TP / len(sort_index[i])
        return precision / len(y_gt)

    if y_prob is None:
        f1 = f1(y_gt, y_pred)
        ja = jaccard(y_gt, y_pred)
        avg_prc = average_prc(y_gt, y_pred)
        avg_recall = average_recall(y_gt, y_pred)
        avg_f1 = average_f1(avg_prc, avg_recall)
        return ja, np.mean(avg_prc), np.mean(avg_recall), np.mean(avg_f1)
    
    auc = roc_auc(y_gt, y_prob)
    p_1 = precision_at_k(y_gt, y_prob, k=1)
    p_3 = precision_at_k(y_gt, y_prob, k=3)
    p_5 = precision_at_k(y_gt, y_prob, k=5)
    f1 = f1(y_gt, y_pred)
    prauc = precision_auc(y_gt, y_prob)
    ja = jaccard(y_gt, y_pred)
    avg_prc = average_prc(y_gt, y_pred)
    avg_recall = average_recall(y_gt, y_pred)
    avg_f1 = average_f1(avg_prc, avg_recall)

    return ja, prauc, np.mean(avg_prc), np.mean(avg_recall), np.mean(avg_f1)

def ddi_rate_score(record, ddi_adj):
    all_cnt, dd_cnt = 0, 0
    for patient in record:
        for adm in patient:
            med_code_set = adm
            for i, med_i in enumerate(med_code_set):
                for j, med_j in enumerate(med_code_set):
                    if j <= i:
                        continue
                    all_cnt += 1
                    if ddi_adj[med_i, med_j] == 1 or ddi_adj[med_j, med_i] == 1:
                        dd_cnt += 1
    return dd_cnt / all_cnt if all_cnt > 0 else 0


def ddi_rate_single_multihot(adm, ddi_adj):
    meds = torch.nonzero(adm, as_tuple=False).reshape(-1)
    m = meds.numel()
    if m < 2:
        return 0.0

    sub = ddi_adj[meds][:, meds]
    dd_cnt = torch.triu(sub, diagonal=1).sum().item()
    all_cnt = m * (m - 1) // 2

    return dd_cnt / all_cnt


def evaluate_multilabel_sets(
    y_true: Sequence[Iterable[int]],
    y_pred: Sequence[Iterable[int]],
    ddi_adj: np.ndarray,
    ignore_ids: Optional[Set[int]] = [0, 1],
) -> Dict[str, float]:
    """
    Set-based multilabel metrics + DDI rate for both prediction and ground truth.

    Returns mean:
        - precision
        - recall
        - f1
        - jaccard
        - ddi_rate_pred
        - ddi_rate_true
    """

    if ignore_ids is None:
        ignore_ids = set()

    ddi_bool = ddi_adj.astype(bool, copy=False)

    def as_set(x):
        s = set(x) if x is not None else set()
        if ignore_ids:
            s.difference_update(ignore_ids)
        return s

    def ddi_rate(med_set):
        meds = sorted(med_set)
        m = len(meds)
        if m < 2:
            return 0.0
        total_pairs = m * (m - 1) // 2
        ddi_pairs = 0
        for i in range(m):
            mi = meds[i]
            for j in range(i + 1, m):
                mj = meds[j]
                if ddi_bool[mi, mj]:
                    ddi_pairs += 1
        return ddi_pairs / total_pairs

    precs, recs, f1s, jaccs = [], [], [], []
    ddi_pred, ddi_true = [], []

    for t, p in zip(y_true, y_pred):
        T, P = as_set(t), as_set(p)

        inter = len(T & P)
        union = len(T | P)

        # precision
        if len(P) == 0:
            prec = 1.0 if len(T) == 0 else 0.0
        else:
            prec = inter / len(P)

        # recall
        rec = 1.0 if len(T) == 0 else inter / len(T)

        # f1
        f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)

        # jaccard
        jacc = 1.0 if union == 0 else inter / union

        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
        jaccs.append(jacc)

        ddi_pred.append(ddi_rate(P))
        ddi_true.append(ddi_rate(T))

    return {
        "precision": float(np.mean(precs)) if precs else 0.0,
        "recall": float(np.mean(recs)) if recs else 0.0,
        "f1": float(np.mean(f1s)) if f1s else 0.0,
        "jaccard": float(np.mean(jaccs)) if jaccs else 0.0,
        "ddi_rate_pred": float(np.mean(ddi_pred)) if ddi_pred else 0.0,
        "ddi_rate_true": float(np.mean(ddi_true)) if ddi_true else 0.0,
    }
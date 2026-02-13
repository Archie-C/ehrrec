import numpy as np
import torch
from scipy.sparse import csr_matrix, hstack

def multihot_csr(df, ids_col: str, vocab_size: int) -> csr_matrix:
    rows, cols = [], []
    for i, ids in enumerate(df[ids_col]):
        for j in set(ids):
            rows.append(i)
            cols.append(j)
    data = np.ones(len(rows), dtype=np.float32)
    return csr_matrix((data, (rows, cols)), shape=(len(df), vocab_size))

def csr_to_torch_sparse(mat: csr_matrix) -> torch.Tensor:
    coo = mat.tocoo()
    idx = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)
    val = torch.tensor(coo.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, size=coo.shape).coalesce()

def make_xy_sparse(df, diag_vocab_size: int, proc_vocab_size: int, med_vocab_size: int):
    X_diag = multihot_csr(df, "DIAG_IDS", diag_vocab_size)
    X_proc = multihot_csr(df, "PROC_IDS", proc_vocab_size)
    X = hstack([X_diag, X_proc], format="csr")  # features

    Y = multihot_csr(df, "MED_IDS", med_vocab_size)  # labels/targets

    return csr_to_torch_sparse(X), csr_to_torch_sparse(Y)

def multihot_dense(df, ids_col: str, vocab_size: int) -> torch.Tensor:
    X = torch.zeros((len(df), vocab_size), dtype=torch.float32)
    for i, ids in enumerate(df[ids_col]):
        X[i, list(set(ids))] = 1.0
    return X

def make_xy_dense(df, diag_vocab_size: int, proc_vocab_size: int, med_vocab_size: int):
    X = torch.cat([
        multihot_dense(df, "DIAG_IDS", diag_vocab_size),
        multihot_dense(df, "PROC_IDS", proc_vocab_size),
    ], dim=1)
    Y = multihot_dense(df, "MED_IDS", med_vocab_size)
    return X, Y

# ------------------------------------------------------------- 
# One hot encode to lists
# ------------------------------------------------------------- 

def multihot_to_id_lists(preds: torch.Tensor):
    """
    preds: (N, V) multi-hot tensor (0/1)
    returns: List[List[int]]
    """
    return [
        torch.nonzero(row, as_tuple=True)[0].tolist()
        for row in preds
    ]

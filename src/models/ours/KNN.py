from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Literal, Tuple

from src.models.BaseModel import BaseModel


Agg = Literal["mean", "sum", "softmax"]

class KNN(BaseModel, nn.Module):
    """
    Multi-hot -> multi-hot kNN.

    - train(X, Y): stores the training set (optionally normalised for cosine).
    - predict(X): batched, vectorised top-k retrieval + neighbour aggregation.

    Similarities:
        - "cosine":
        - "jaccard"
    """
    def __init__(
        self, 
        k: int = 50, 
        similarity: Literal["cosine", "jaccard"] = "cosine",
        agg: Agg = "mean",
        threshold: float = 0.5,
        chunk_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        nn.Module.__init__(self)
        self.name: str = "k-nearest neighbours"
        
        self.k = int(k)
        self.similarity = similarity
        self.agg = agg
        self.threshold = float(threshold)
        self.chunk_size = chunk_size
        self.device = device
        self.dtype = dtype

        # Stored training tensors
        self.register_buffer("X_train", torch.empty(0), persistent=False)
        self.register_buffer("Y_train", torch.empty(0), persistent=False)
        self._trained = False
    
    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """
        X: [N, D] multi-hot (0/1 or counts)
        Y: [N, M] multi-hot (0/1)
        """
        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError("X and Y must be 2D tensors: X[N,D], Y[N,M].")
        if X.size(0) != Y.size(0):
            raise ValueError("X and Y must have the same N (rows).")

        dev = self.device if self.device is not None else X.device
        X = X.to(device=dev)
        Y = Y.to(device=dev)
        
        if self.similarity == "cosine":
            Xn = F.normalize(X.to(self.dtype), p=2, dim=1)
            self.X_train = Xn
        elif self.similarity == "jaccard":
            # Store as 0/1 float for intersection matmul.
            self.X_train = (X != 0).to(self.dtype)
        else:
            raise ValueError("similarity must be 'cosine' or 'jaccard'")
        
        self.Y_train = (Y != 0).to(torch.uint8) if Y.dtype != torch.uint8 else Y
        self._trained = True
    
    @torch.no_grad()
    def predict(self, X: torch.Tensor, return_scores: bool = False, return_neighbours: bool = False,):
        """
        Returns:
            pred: [B, M] multi-hot (uint8)
            (optional) scores: [B, M] float scores
            (optional) nn_idx: [B, k] neighbour indices
            (optional) nn_sim: [B, k] neighbour similarities
        """
        if not self._trained or self.X_train.numel() == 0:
            raise RuntimeError("Call train(X_train, Y_train) before predict().")
        if X.ndim != 2:
            raise ValueError("X must be 2D tensor: X[B,D].")
        if X.size(1) != self.X_train.size(1):
            raise ValueError(f"X has D={X.size(1)} but trained D={self.X_train.size(1)}.")
        
        dev = self.X_train.device
        Xq = X.to(device=dev)

        if self.similarity == "cosine":
            Xq = F.normalize(Xq.to(self.dtype), p=2, dim=1)
            nn_sim, nn_idx = self._topk_over_train_cosine(Xq)
        else:
            Xq = (Xq != 0).to(self.dtype)
            nn_sim, nn_idx = self._topk_over_train_jaccard(Xq)

        Yn = self.Y_train[nn_idx].to(torch.float32)  # [B,k,M]

        # Aggregate neighbour labels
        if self.agg == "sum":
            scores = Yn.sum(dim=1)  # [B,M]
        elif self.agg == "mean":
            scores = Yn.mean(dim=1)  # [B,M] in [0,1]
        elif self.agg == "softmax":
            w = torch.softmax(nn_sim, dim=1)  # [B,k]
            scores = (Yn * w.unsqueeze(-1)).sum(dim=1)  # [B,M]
        else:
            raise ValueError("agg must be 'sum', 'mean', or 'softmax'.")

        pred = (scores >= self.threshold).to(torch.uint8)

        outs = (pred,)
        if return_scores:
            outs += (scores,)
        if return_neighbours:
            outs += (nn_idx, nn_sim)
        return outs[0] if len(outs) == 1 else outs
    
    def _topk_over_train_cosine(self, Xq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Xq: [B,D] float normalised
        returns (nn_sim [B,k], nn_idx [B,k])
        """
        Xtr = self.X_train  # [N,D]
        B = Xq.size(0)
        N = Xtr.size(0)
        k = min(self.k, N)

        if self.chunk_size is None or N <= self.chunk_size:
            sims = Xq @ Xtr.T  # [B,N]
            nn_sim, nn_idx = torch.topk(sims, k=k, dim=1)
            return nn_sim, nn_idx

        # Chunked: keep running top-k
        best_sim = None
        best_idx = None
        offset = 0

        for Xc in Xtr.split(self.chunk_size, dim=0):
            sims_c = Xq @ Xc.T  # [B,nc]
            sim_c, idx_c = torch.topk(sims_c, k=min(k, Xc.size(0)), dim=1)
            idx_c = idx_c + offset

            if best_sim is None:
                best_sim, best_idx = sim_c, idx_c
            else:
                merged_sim = torch.cat([best_sim, sim_c], dim=1)
                merged_idx = torch.cat([best_idx, idx_c], dim=1)
                best_sim, top_pos = torch.topk(merged_sim, k=k, dim=1)
                best_idx = merged_idx.gather(1, top_pos)

            offset += Xc.size(0)

        return best_sim, best_idx

    def _topk_over_train_jaccard(self, Xq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Xq: [B,D] 0/1 float
        returns (nn_sim [B,k], nn_idx [B,k])
        """
        Xtr = self.X_train  # [N,D] 0/1 float
        B = Xq.size(0)
        N = Xtr.size(0)
        k = min(self.k, N)

        # Always chunk unless explicitly small, because we need union terms too.
        chunk = self.chunk_size if self.chunk_size is not None else 20000

        q_count = Xq.sum(dim=1, keepdim=True)  # [B,1]

        best_sim = None
        best_idx = None
        offset = 0

        for Xc in Xtr.split(chunk, dim=0):
            # inter: [B,nc]
            inter = Xq @ Xc.T
            tr_count = Xc.sum(dim=1).unsqueeze(0)  # [1,nc]
            union = q_count + tr_count - inter
            jac = inter / union.clamp_min(1e-8)

            sim_c, idx_c = torch.topk(jac, k=min(k, Xc.size(0)), dim=1)
            idx_c = idx_c + offset

            if best_sim is None:
                best_sim, best_idx = sim_c, idx_c
            else:
                merged_sim = torch.cat([best_sim, sim_c], dim=1)
                merged_idx = torch.cat([best_idx, idx_c], dim=1)
                best_sim, top_pos = torch.topk(merged_sim, k=k, dim=1)
                best_idx = merged_idx.gather(1, top_pos)

            offset += Xc.size(0)

        return best_sim, best_idx
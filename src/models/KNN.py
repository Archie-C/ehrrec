from __future__ import annotations

from dataclasses import dataclass
import dataclasses
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from torch import nn

from src.core.interfaces.basemodel import Model


@dataclass
class KNNConfig:
    k: int = 50
    threshold: float = 0.3
    gamma: float = 1.0
    temperature: float = 1.0
    min_sim: float = 0.0
    mode: str = "mean"  # {"mean", "softmax", "max"}


class KNNModel(Model, nn.Module):
    """
    Memory-based KNN model for medication recommendation.

    Stores normalised diagnosis/procedure vectors + target multi-hot labels
    and performs weighted voting over the top-K most similar admissions.
    """

    def __init__(self, config: KNNConfig, device: Optional[torch.device] = None):
        super().__init__()
        self.config = config
        self.device = device or torch.device("cpu")
        self.register_buffer("memory_features", torch.empty(0))
        self.register_buffer("memory_labels", torch.empty(0))

    def init_weights(self):
        # No learnable parameters
        pass

    def fit(self, features: torch.Tensor, labels: torch.Tensor):
        """Cache the training admissions (already normalised)."""
        self.memory_features = F.normalize(features.to(self.device), p=2, dim=1)
        self.memory_labels = labels.to(self.device)

    def to(self, *args, **kwargs):
        obj = super().to(*args, **kwargs)
        device = None
        if args:
            if isinstance(args[0], torch.device):
                device = args[0]
            elif isinstance(args[0], str):
                device = torch.device(args[0])
        if device is None and "device" in kwargs:
            val = kwargs["device"]
            device = val if isinstance(val, torch.device) else torch.device(val)
        if device is not None:
            self.device = device
        return obj

    def forward(self, query_features: torch.Tensor) -> torch.Tensor:
        preds, _ = self.predict_with_scores(query_features)
        return preds

    def save_memory(self, path):
        payload = {
            "config": dataclasses.asdict(self.config),
            "features": self.memory_features.detach().cpu(),
            "labels": self.memory_labels.detach().cpu(),
        }
        torch.save(payload, path)

    def load_memory(self, path, map_location=None):
        payload = torch.load(path, map_location=map_location)
        self.config = KNNConfig(**payload["config"])
        self.memory_features = payload["features"].to(self.device)
        self.memory_labels = payload["labels"].to(self.device)

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------
    def predict_with_scores(
        self,
        query_features: torch.Tensor,
        **overrides,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.memory_features.numel() == 0:
            raise RuntimeError("KNNModel.fit must be called before inference.")

        config = self._with_overrides(overrides)
        queries = self._prepare(query_features)
        sims = self._similarity(queries)
        preds, scores = self._batch_knn_predict(
            sims,
            self.memory_labels,
            k=config.k,
            threshold=config.threshold,
            gamma=config.gamma,
            temperature=config.temperature,
            min_sim=config.min_sim,
            mode=config.mode,
        )
        return preds, scores

    def predict_probabilities(self, query_features: torch.Tensor) -> torch.Tensor:
        _, scores = self.predict_with_scores(query_features)
        return scores

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _with_overrides(self, overrides: dict) -> KNNConfig:
        if not overrides:
            return self.config
        return KNNConfig(
            k=overrides.get("k", self.config.k),
            threshold=overrides.get("threshold", self.config.threshold),
            gamma=overrides.get("gamma", self.config.gamma),
            temperature=overrides.get("temperature", self.config.temperature),
            min_sim=overrides.get("min_sim", self.config.min_sim),
            mode=overrides.get("mode", self.config.mode),
        )

    def _prepare(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got shape {tensor.shape}")
        tensor = tensor.to(self.device)
        return F.normalize(tensor, p=2, dim=1)

    def _similarity(self, queries: torch.Tensor) -> torch.Tensor:
        return torch.mm(queries, self.memory_features.t())

    def _batch_knn_predict(
        self,
        sims: torch.Tensor,
        train_labels: torch.Tensor,
        *,
        k: int,
        threshold: float,
        gamma: float,
        temperature: float,
        min_sim: float,
        mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if min_sim > 0.0:
            sims = sims.clone()
            sims[sims < min_sim] = 0.0

        n_query, n_train = sims.shape
        k = int(min(k, n_train))
        if k <= 0 or n_train == 0:
            zeros = torch.zeros((n_query, train_labels.shape[1]), device=sims.device)
            return zeros.int(), zeros

        topk = torch.topk(sims, k, dim=1)
        if mode == "softmax":
            weights = torch.softmax((topk.values.pow(gamma)) / max(temperature, 1e-8), dim=1).unsqueeze(2)
        else:
            weights = topk.values.pow(gamma).unsqueeze(2)

        neighbours = train_labels[topk.indices]

        if mode == "max":
            scores, _ = (neighbours * (weights > 0)).max(dim=1)
        else:
            numerator = (weights * neighbours).sum(dim=1)
            denominator = weights.sum(dim=1) + 1e-8
            scores = numerator / denominator

        preds = (scores >= threshold).int()
        return preds, scores

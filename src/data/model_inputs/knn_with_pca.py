from __future__ import annotations

from typing import List
from pathlib import Path

import numpy as np
import torch
import logging

from src.core.interfaces.preprocessor import Preprocessor

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm optional
    tqdm = None


class KNNModelInputBuilderWithPCA(Preprocessor):
    """Convert GAMENet-style sequences into flat tensors and apply dimensionality reduction for a KNN model."""
    def __init__(
        self,
        n_components: int = 100,
        cache_path: str | None = None,
        refit: bool = True,
        log_level: int = logging.INFO,
        show_progress: bool = True,
    ):
        self.mean = None
        self.std = None
        self.n_components = n_components
        self.W = None
        self.cache_path = cache_path
        self.refit = refit
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
            self.logger.propagate = False

        self.show_progress = show_progress and (tqdm is not None)
        if show_progress and tqdm is None:
            self.logger.warning("show_progress=True but tqdm not available; progress bars disabled.")

    def run(self, *, source: str, context, train_data, val_data, test_data, voc_size=None):
        if source != "gamenet":
            raise ValueError(f"Unsupported source '{source}' for KNN inputs")
        sizes = voc_size or context.vocab_sizes()
        train = self._build_split(train_data, sizes)
        cache_path = self._resolve_cache_path(context)
        if self.refit or not cache_path.exists():
            self.logger.info("Fitting PCA projection (refit=%s)", self.refit)
            self._create_pca_projection_matrix(train["features"])
            self._save_pca(cache_path)
        else:
            self.logger.info("Loading PCA projection from %s", cache_path)
            self._load_pca(cache_path)
        train["features"] = self._apply_pca(train["features"])
        val = self._build_split(val_data, sizes)
        val["features"] = self._apply_pca(val["features"])
        test = self._build_split(test_data, sizes)
        test["features"] = self._apply_pca(test["features"])
        return train, val, test

    def _resolve_cache_path(self, context):
        output_dir = context.metadata.get("output_dir")
        if self.cache_path:
            path = Path(self.cache_path)
            if not path.is_absolute():
                if output_dir:
                    path = Path(output_dir) / path
                else:
                    path = Path.cwd() / path
            return path
        if output_dir:
            return Path(output_dir) / "knn_pca.pt"
        return Path.cwd() / "knn_pca.pt"

    def _save_pca(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mean": self.mean, "std": self.std, "W": self.W, "n_components": self.n_components}, path)

    def _load_pca(self, path: Path):
        state = torch.load(path, map_location="cpu")
        self.mean = state["mean"]
        self.std = state["std"]
        self.W = state["W"]
        self.n_components = state.get("n_components", self.n_components)

    def _admission_to_X_y(self, adm, voc_size):
        D, P, M = voc_size
        X = np.zeros(D + P, dtype=np.float32)
        y = np.zeros(M, dtype=np.float32)

        for d in adm[0]:
            X[d] = 1.0
        for p in adm[1]:
            X[D + p] = 1.0 
        for m in adm[2]:
            y[m] = 1.0
        return X, y
    
    def _create_pca_projection_matrix(self, X):
        self.logger.info("Creating PCA Projection Matrix")
        X_scaled = self._scale(X)
        n_samples = X_scaled.shape[0]
        cov_matrix = torch.matmul(X_scaled.T, X_scaled) / (n_samples - 1)
        self.logger.info(f"Covariance Matrix Shape: {cov_matrix.shape}")
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
        idx = torch.argsort(eigenvalues, descending=True)
        W = eigenvectors[:, idx[:self.n_components]]
        total_variance = torch.sum(eigenvalues)
        selected_variance = torch.sum(eigenvalues[idx[:self.n_components]])
        explained_variance_ratio = (selected_variance / total_variance) * 100
        
        self.logger.info(f"Number of Principal Components Selected: {self.n_components}")
        self.logger.info(f"Cumulative Explained Variance: {explained_variance_ratio:.2f}%")
        self.W = W
    
    def _apply_pca(self, X):
        X = self._scale(X)
        x_pca = torch.matmul(X, self.W)
        return x_pca
    
    def _scale(self, X):
        mean = None
        std = None
        if self.mean is not None and self.std is not None:
            mean = self.mean
            std = self.std
        
        if mean is None:
            mean = torch.mean(X, dim=0, keepdim=True)
        if std is None:
            std = torch.std(X, dim=0, keepdim=True)
        
        self.mean = mean
        self.std = std
        
        std[std == 0] = 1e-12
        X_scaled = (X - mean)
        # more of a shift than a scale
        return X_scaled
        

    def _build_split(self, data, voc_size):
        X, y = [], []
        patient_counts: List[int] = []
        for patient in data:
            patient_counts.append(len(patient))
            for adm in patient:
                X_i, y_i = self._admission_to_X_y(adm, voc_size)
                X.append(X_i)
                y.append(y_i)
        if not X:
            features = torch.empty(0, sum(voc_size[:2]), dtype=torch.float32)
            labels = torch.empty(0, voc_size[2], dtype=torch.float32)
        else:
            features_np = np.stack(X).astype(np.float32, copy=False)
            labels_np = np.stack(y).astype(np.float32, copy=False)
            features = torch.from_numpy(features_np)
            labels = torch.from_numpy(labels_np)
        return {
            "features": features,
            "labels": labels,
            "patient_counts": patient_counts,
        }

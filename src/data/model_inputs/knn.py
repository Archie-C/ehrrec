
from __future__ import annotations

from typing import List

import numpy as np
import torch
from torch.nn.functional import normalize

from src.core.interfaces.preprocessor import Preprocessor


class KNNModelInputBuilder(Preprocessor):
    """Convert GAMENet-style sequences into flat tensors for a KNN model."""

    def run(self, *, source: str, context, train_data, val_data, test_data, voc_size=None):
        if source != "gamenet":
            raise ValueError(f"Unsupported source '{source}' for KNN inputs")
        sizes = voc_size or context.vocab_sizes()
        train = self._build_split(train_data, sizes)
        val = self._build_split(val_data, sizes)
        test = self._build_split(test_data, sizes)
        return train, val, test

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
            features = normalize(features, p=2, dim=1)
            labels = torch.from_numpy(labels_np)
        return {
            "features": features,
            "labels": labels,
            "patient_counts": patient_counts,
        }

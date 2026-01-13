from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List

import torch

from src.core.interfaces.basemodel import Model


@dataclass
class MostPopularConfig:
    top_k: int = 10


class MostPopularModel(Model):
    """
    Baseline recommender that assigns each diagnosis/procedure its most
    frequently co-prescribed medications from the training set.
    """

    def __init__(self, config: MostPopularConfig):
        super().__init__()
        self.config = config
        self.diag_to_meds: Dict[int, List[int]] = {}
        self.proc_to_meds: Dict[int, List[int]] = {}

    def init_weights(self):
        # No trainable parameters.
        pass

    def forward(self, admission):
        diag_codes, proc_codes, _ = admission
        return self.predict(admission=admission)

    def fit(self, patients: List[List[List[List[int]]]]):
        diag_counter: Dict[int, Counter] = defaultdict(Counter)
        proc_counter: Dict[int, Counter] = defaultdict(Counter)

        for patient in patients:
            for diag_codes, proc_codes, med_codes in patient:
                meds = list(med_codes)
                for d in diag_codes:
                    diag_counter[d].update(meds)
                for p in proc_codes:
                    proc_counter[p].update(meds)

        self.diag_to_meds = {
            d: [m for m, _ in counter.most_common(self.config.top_k)]
            for d, counter in diag_counter.items()
        }
        self.proc_to_meds = {
            p: [m for m, _ in counter.most_common(self.config.top_k)]
            for p, counter in proc_counter.items()
        }

    def predict(self, admission) -> List[int]:
        diag_codes, proc_codes, _ = admission
        meds = set()
        for d in diag_codes:
            meds.update(self.diag_to_meds.get(d, []))
        for p in proc_codes:
            meds.update(self.proc_to_meds.get(p, []))
        return list(meds)

    def save(self, path, extra=None):
        payload = {
            "config": asdict(self.config),
            "diag_to_meds": self.diag_to_meds,
            "proc_to_meds": self.proc_to_meds,
            "extra": extra,
        }
        torch.save(payload, path)

    def load(self, path, map_location=None):
        payload = torch.load(path, map_location=map_location)
        self.config = MostPopularConfig(**payload["config"])
        self.diag_to_meds = payload["diag_to_meds"]
        self.proc_to_meds = payload["proc_to_meds"]
        return payload.get("extra")

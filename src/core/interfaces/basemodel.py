from __future__ import annotations
from abc import ABC, abstractmethod
import torch

class Model(ABC):
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        """Run forward pass and return model outputs."""
        pass

    @abstractmethod
    def init_weights(self):
        """Initialise model parameters (if custom init needed)."""
        pass

    def save(self, path, extra=None):
        """Save model weights + optionally metadata."""
        payload = {
            "state_dict": self.state_dict(),
            "extra": extra,   # config, vocab sizes, etc.
        }
        torch.save(payload, path)

    def load(self, path, map_location=None):
        """Load model weights + extra metadata."""
        payload = torch.load(path, map_location=map_location)
        self.load_state_dict(payload["state_dict"])
        return payload.get("extra", None)
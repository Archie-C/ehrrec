from abc import ABC
from typing import Optional, Any

class BaseTrainer(ABC):
    def __init__(self, train_config):
        self.__name__ = "Base Trainer"
        self.config = train_config
    
    def train(self, train_loader, val_loader: Optional[Any] = None):
        raise NotImplementedError
    
    def __repr__(self):
        config_representation = repr(self.config)
        return self.__name__ + config_representation
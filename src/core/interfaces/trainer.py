from abc import ABC, abstractmethod


class Trainer(ABC):
    
    @abstractmethod
    def train(self, model, train_data, val_data, context):
        pass
    
    

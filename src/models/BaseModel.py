
from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self) -> None:
        self.name: str = "Base"
    
    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        raise NotImplementedError
    
    @abstractmethod
    def predict(self, X):
        raise NotImplementedError
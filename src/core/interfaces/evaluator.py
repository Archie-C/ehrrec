from abc import ABC, abstractmethod


class Evaluator(ABC):
    
    @abstractmethod
    def evaluate(self, model, data_eval, context, **kwargs):
        pass

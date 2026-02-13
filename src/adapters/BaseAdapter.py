from __future__ import annotations

from abc import ABC, abstractmethod
class BaseAdapter(ABC):
    def __init__(self):
        ...
    
    @abstractmethod
    def adapt(self):
        raise NotImplementedError
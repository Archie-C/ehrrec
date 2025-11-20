from abc import ABC, abstractmethod


class Splitter(ABC):
    @abstractmethod
    def split(self, context):
        """Split processed dataset into train/val/test sets."""
        pass

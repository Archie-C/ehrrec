import logging

from src.core.interfaces.preprocessor import Preprocessor

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm optional
    tqdm = None


class MostPopularInputBuilder(Preprocessor):
    """Most Popular baseline works directly on GAMENet-style records."""
    def __init__(self, log_level: int = logging.INFO, show_progress: bool = True):

        
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

    def run(self, *, source: str, context, train_data, val_data, test_data, **_):
        if source != "gamenet":
            raise ValueError(f"Unsupported source '{source}' for MostPopular inputs")
        return train_data, val_data, test_data

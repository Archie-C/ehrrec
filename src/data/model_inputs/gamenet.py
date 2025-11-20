
from src.core.interfaces.preprocessor import Preprocessor


class GAMENetModelInputBuilder(Preprocessor):
    """Placeholder for additional GAMENet-specific input shaping."""

    def run(self, *, source: str, context, train_data, val_data, test_data, **kwargs):
        if source != "gamenet":
            raise ValueError(f"Unsupported source '{source}' for GAMENet model inputs")
        return train_data, val_data, test_data

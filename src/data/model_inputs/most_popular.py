from src.core.interfaces.preprocessor import Preprocessor


class MostPopularInputBuilder(Preprocessor):
    """Most Popular baseline works directly on GAMENet-style records."""

    def run(self, *, source: str, context, train_data, val_data, test_data, **_):
        if source != "gamenet":
            raise ValueError(f"Unsupported source '{source}' for MostPopular inputs")
        return train_data, val_data, test_data

from src.core.interfaces.loader import Loader


class PklLoader(Loader):
    def __init__(self, backend=None):
        self.backend = backend

    def load(self, data_path, override_backend: str = "dill"):
        if override_backend == "dill":
            import dill as pkl
            with open(data_path, "rb") as f:
                return pkl.load(f)

        if self.backend is None:
            raise ValueError("Backend-aware pickle loading requires a backend instance")

        if self.backend.backend == "pandas":
            import pandas as pd
            return pd.read_pickle(data_path)
        if self.backend.backend == "polars":
            import polars as pl
            return pl.read_pickle(data_path)
        raise ValueError("Unsupported backend")

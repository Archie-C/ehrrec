from src.core.interfaces.loader import Loader


class CSVLoader(Loader):
    def __init__(self, backend_factory):
        self.backend = backend_factory
    
    def load(self, path: str, **kwargs):
        if self.backend.backend == "pandas":
            import pandas as pd
            df = pd.read_csv(path, **kwargs)
        elif self.backend.backend == "polars":
            import polars as pl
            df = pl.read_csv(path, **kwargs)
        else:
            raise ValueError("Unsupported backend")
        return self.backend.wrap(df)
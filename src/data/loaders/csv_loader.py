from core.interfaces.loader import Loader


class CSVLoader(Loader):
    def __init__(self, backend_factory, **kwargs):
        self.backend = backend_factory
        self.kwargs = kwargs
    
    def load(self, path: str):
        if self.backend.backend == "pandas":
            import pandas as pd
            df = pd.read_csv(path, **self.kwargs)
        elif self.backend.backend == "polars":
            import polars as pl
            df = pl.read_csv(path, **self.kwargs)
        else:
            raise ValueError("Unsupported backend")
        return self.backend.wrap(df)
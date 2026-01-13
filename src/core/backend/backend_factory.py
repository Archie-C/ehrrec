from src.core.backend.pandas_table import PandasTable
from src.core.backend.polars_table import PolarsTable

def create_backend(cfg_backend):
    """
    cfg_backend.name: "pandas" or "polars"
    """
    return BackendFactory(cfg_backend.name)


class BackendFactory:
    def __init__(self, backend: str):
        self.backend = backend
    
    def wrap(self, obj):
        if self.backend == "pandas":
            return PandasTable(obj)
        elif self.backend == "polars":
            return PolarsTable(obj)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
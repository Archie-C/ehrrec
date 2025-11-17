from src.core.interfaces.column import Column
import polars as pl

class PolarsColumn(Column):
    def __init__(self, series):
        self.series = series

    def map(self, func):
        vals = [func(v) for v in self.series]
        return PolarsColumn(pl.Series(self.series.name, vals))
    
    def to_list(self):
        return self.series.to_list()

    def unique(self):
        return self.series.unique().to_list()
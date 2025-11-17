from src.core.interfaces.column import Column


class PandasColumn(Column):
    def __init__(self, series):
        self.series = series

    def map(self, func):
        return PandasColumn(self.series.map(func))

    def to_list(self):
        return self.series.tolist()

    def unique(self):
        return self.series.unique().tolist()
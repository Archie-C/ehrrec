from src.core.interfaces.column import Column


class PandasColumn(Column):
    def __init__(self, series):
        self.series = series

    def astype(self, dtype):
        return PandasColumn(self.series.astype(dtype))

    def map(self, func):
        return PandasColumn(self.series.map(func))


    def to_list(self):
        return self.series.tolist()

    def unique(self):
        return self.series.unique().tolist()

    @property
    def str(self):
        return self.series.str
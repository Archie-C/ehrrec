from core.interfaces.column import Column


class PandasColumn(Column):
    def __init__(self, series):
        self.series = series

    def map(self, func):
        return PandasColumn(self.series.map(func))

    def unique(self):
        return self.series.unique().tolist()
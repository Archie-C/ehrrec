from core.backend.pandas_table import PandasTable
from core.interfaces.groupby import GroupBy

class PandasGroupBy(GroupBy):
    def __init__(self, gb):
        self.gb = gb

    def agg(self, agg_dict):
        df = self.gb.agg(agg_dict)
        return PandasTable(df.reset_index())
    
    def head(self, n):
        df = self.gb.head(n)
        return PandasTable(df)

    def size(self):
        df = self.gb.size().reset_index(name='size')
        return PandasTable(df)
from core.backend.polars_table import PolarsTable
from core.interfaces.groupby import GroupBy
import polars as pl

class PolarsGroupBy(GroupBy):
    def __init__(self, gb):
        self.gb = gb

    def agg(self, agg_dict):
        exprs = []
        for col, fn in agg_dict.items():
            if fn == "unique":
                exprs.append(pl.col(col).unique().alias(col))
            else:
                exprs.append(getattr(pl.col(col), fn)().alias(col))
        df = self.gb.agg(exprs)
        return PolarsTable(df)
    
    def head(self, n):
        df = self.gb.head(n)
        return PolarsTable(df)

    def size(self):
        df = self.gb.count()
        df = df.rename({'count': 'size'})
        return PolarsTable(df)
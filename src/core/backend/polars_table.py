
from src.core.interfaces.groupby import GroupBy
from src.core.backend.polars_column import PolarsColumn
from src.core.interfaces.table import Table
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

class PolarsTable(Table):
    def __init__(self, df: pl.DataFrame):
        self.df = df
        
    def assign(self, **kwargs):
        df = self.df
        for col, func in kwargs.items():
            col_values = func(self)  # get a Column object
            df = df.with_columns(pl.Series(col, col_values.series))
        return PolarsTable(df)
    def drop_duplicates(self, subset=None, inplace=False):
        df = self.df.unique(subset=subset)
        return PolarsTable(df)
    
    def filter(self, predicate):
        # brute-force for now: convert rows one-by-one
        mask = [predicate(row) for row in self.df.to_dicts()]
        df = self.df.with_row_index().filter(pl.Series("mask", mask)).drop("mask")
        return PolarsTable(df)
        
    def groupby(self, by):
        return PolarsGroupBy(self.df.group_by(by))

    def merge(self, right, on=None, left_on=None, right_on=None, how="inner", suffixes=("_x", "_y")):
        # Normalise arguments
        if on is not None:
            left_cols = right_cols = on
        else:
            left_cols = left_on
            right_cols = right_on

        df = self.df.join(
            right.df,
            left_on=left_cols,
            right_on=right_cols,
            how=how,
            suffix= suffixes[1]
        )
        return PolarsTable(df)
    
    def reset_index(self, drop=True):
        df = self.df.with_row_index(name="index")
        return PolarsTable(df.drop("index")) if drop else PolarsTable(df)

    def row_iter(self):
        for row in self.df.to_dicts():
            yield row
    
    def select(self, columns):
        return PolarsTable(self.df.select(columns))
    
    def slice_rows(self, start, end):
        length = end - start
        return PolarsTable(self.df.slice(start, length))
    
    def sort_values(self, by, ascending=True):
        if isinstance(by, str):
            by = [by]
        df = self.df.sort(by, descending=not ascending)
        return PolarsTable(df)
    
    def to_numpy(self):
        return self.df.to_numpy()
    
    def to_datetime(self, columns, format=None, errors="raise", utc=None):
        df = self.df
        for col in columns:
            df = df.with_columns(
                pl.col(col).str.strptime(pl.Datetime, fmt=format, strict=(errors=="raise"))
            )
            if utc:
                df = df.with_columns(pl.col(col).dt.replace_time_zone("UTC"))
        return PolarsTable(df)
    
    def to_pickle(self, path: str):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self.df, f)
    
    def head(self, n):
        return PolarsTable(self.df.head(n))
    
    def __getitem__(self, col):
        return PolarsColumn(self.df[col])
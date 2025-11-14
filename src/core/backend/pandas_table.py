import pandas as pd

from core.backend.pandas_groupby import PandasGroupBy
from core.backend.pandas_column import PandasColumn
from core.interfaces.table import Table


class PandasTable(Table):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def assign(self, **kwargs):
        df = self.df.copy()
        for col, func in kwargs.items():
            df[col] = func(PandasTable(df)).df[col] 
        return PandasTable(df)
        
    def astype(self, dtypes):
        new_df = self.df.astype(dtypes)
        return PandasTable(new_df)
    
    def drop_duplicates(self, subset=None):
        return PandasTable(self.df.drop_duplicates(subset=subset))
    
    def groupby(self, by):
        return PandasGroupBy(self.df.groupby(by))

    def merge(self, right, on=None, left_on=None, right_on=None, how="inner", suffixes=("_x", "_y")):
        merged = self.df.merge(
            right.df,
            how=how,
            on=on,
            left_on=left_on,
            right_on=right_on,
            suffixes=suffixes
        )
        return PandasTable(merged)
    
    def reset_index(self, drop=True):
        df = self.df.reset_index(drop=drop)
        return PandasTable(df)

    def row_iter(self):
        for _, row in self.df.iterrows():
            yield row.to_dict()

    def filter(self, predicate):
        mask = self.df.apply(lambda row: predicate(row.to_dict()), axis=1)
        return PandasTable(self.df[mask])
    
    def select(self, columns):
        return PandasTable(self.df[columns])

    def slice_rows(self, start, end):
        return PandasTable(self.df.iloc[start:end])

    def sort_values(self, by, ascending=True):
        df = self.df.sort_values(by=by, ascending=ascending)
        return PandasTable(df)
    
    def to_numpy(self):
        return self.df.to_numpy()
    
    def to_datetime(self, columns, format=None, errors="raise", utc=None):
        df = self.df.copy()
        for col in columns:
            df[col] = pd.to_datetime(df[col], format=format, errors=errors, utc=utc)
        return PandasTable(df)
    
    def to_pickle(self, path: str):
        self.df.to_pickle(path)
    
    def head(self, n):
        return PandasTable(self.df.head(n))

    def __getitem__(self, col):
        return PandasColumn(self.df[col])
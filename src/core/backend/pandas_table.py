import pandas as pd

from src.core.interfaces.groupby import GroupBy
from src.core.backend.pandas_column import PandasColumn
from src.core.interfaces.table import Table

class PandasGroupBy(GroupBy):
    def __init__(self, gb):
        self.gb = gb

    def agg(self, agg_dict):
        df = self.gb.agg(agg_dict)
        from src.core.backend.pandas_table import PandasTable
        return PandasTable(df.reset_index())
    
    def head(self, n):
        df = self.gb.head(n)
        from src.core.backend.pandas_table import PandasTable
        return PandasTable(df)

    def size(self):
        df = self.gb.size().reset_index(name='size')
        from src.core.backend.pandas_table import PandasTable
        return PandasTable(df)


class PandasTable(Table):
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def __repr__(self):
        return repr(self.df)
        
    def as_type(self, dtypes: dict):
        df = self.df.astype(dtypes)
        return PandasTable(df)

    def assign(self, **kwargs):
        df = self.df.copy()
        for col, val in kwargs.items():
            if callable(val):
                result = val(self) 
                if isinstance(result, PandasColumn):
                    df[col] = result.series
                else:
                    df[col] = result
            else:
                df[col] = val
        return PandasTable(df)
        
    def astype(self, dtypes):
        new_df = self.df.astype(dtypes)
        return PandasTable(new_df)
    
    def drop(self, columns=None, index=None, axis=1):
        new_df = self.df.drop(columns=columns, index=index, axis=axis)
        return PandasTable(new_df)
    
    def drop_duplicates(self, subset=None):
        return PandasTable(self.df.drop_duplicates(subset=subset))
    
    def drop_na(self, subset=None):
        new_df = self.df.dropna(subset=subset)
        return PandasTable(new_df)
    
    def fillna(self, values=None, method=None):
        df = self.df.fillna(value=values, method=method)
        return PandasTable(df)
    
    def filter(self, predicate):
        mask = self.df.apply(lambda row: predicate(row.to_dict()), axis=1)
        return PandasTable(self.df[mask])
    
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

    def rename(self, mapping):
        new_df = self.df.rename(columns=mapping)
        return PandasTable(new_df)

    def row_iter(self):
        for _, row in self.df.iterrows():
            yield row.to_dict()
            
    def rows_count(self):
        return self.df.shape[0]
    
    def select(self, columns):
        return PandasTable(self.df[columns])

    @property
    def shape(self):
        return self.df.shape

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

    def __len__(self):
        return len(self.df)
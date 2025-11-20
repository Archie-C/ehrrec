import pandas as pd

from src.core.interfaces.groupby import GroupBy
from src.core.backend.pandas_column import PandasColumn
from src.core.interfaces.table import Table


class PandasGroupBy(GroupBy):
    def __init__(self, gb):
        # gb is a pandas.core.groupby.DataFrameGroupBy
        self.gb = gb

    def agg(self, agg_dict):
        """
        Thin wrapper around pandas GroupBy.agg.
        No index reset here – caller controls that,
        exactly like in the original pandas code.
        """
        df = self.gb.agg(agg_dict)
        from src.core.backend.pandas_table import PandasTable
        return PandasTable(df)

    def head(self, n):
        df = self.gb.head(n)
        from src.core.backend.pandas_table import PandasTable
        return PandasTable(df)

    def size(self):
        """
        Matches pandas: groupby.size().reset_index(name='size')
        This is how the old code built count tables.
        """
        df = self.gb.size().reset_index(name="size")
        from src.core.backend.pandas_table import PandasTable
        return PandasTable(df)


class PandasTable(Table):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __repr__(self):
        return repr(self.df)

    # ------------------------------------------------------------------
    # Basic type / casting
    # ------------------------------------------------------------------
    def as_type(self, dtypes: dict):
        df = self.df.astype(dtypes)
        return PandasTable(df)

    def astype(self, dtypes):
        df = self.df.astype(dtypes)
        return PandasTable(df)

    # ------------------------------------------------------------------
    # Column creation / mutation
    # ------------------------------------------------------------------
    def assign(self, **kwargs):
        """
        Mirror pandas.DataFrame.assign semantics:

        - For callable values, pass the underlying DataFrame (not Table).
        - Accept direct Series / arrays / scalars.
        """
        df = self.df

        new_kwargs = {}
        for col, val in kwargs.items():
            if callable(val):
                res = val(df)  # df is a pandas.DataFrame
            else:
                res = val

            if isinstance(res, PandasColumn):
                new_kwargs[col] = res.series
            else:
                new_kwargs[col] = res

        out = df.assign(**new_kwargs)
        return PandasTable(out)

    # ------------------------------------------------------------------
    # Row/column removal
    # ------------------------------------------------------------------
    def drop(self, columns=None, index=None, axis=1):
        df = self.df.drop(columns=columns, index=index, axis=axis)
        return PandasTable(df)

    def drop_duplicates(self, subset=None):
        df = self.df.drop_duplicates(subset=subset)
        return PandasTable(df)

    def drop_na(self, subset=None):
        df = self.df.dropna(subset=subset)
        return PandasTable(df)

    # ------------------------------------------------------------------
    # Missing values
    # ------------------------------------------------------------------
    def fillna(self, values=None, method=None):
        df = self.df.fillna(value=values, method=method)
        return PandasTable(df)

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------
    def filter(self, predicate):
        """
        Apply a predicate that receives a row *as a dict*, just like the old pipeline.
        The result must be converted to pure bool dtype.
        """
        mask = self.df.apply(lambda row: bool(predicate(row.to_dict())), axis=1)
        # mask must be boolean dtype
        mask = mask.astype(bool)

        df = self.df[mask]
        return PandasTable(df)

    # ------------------------------------------------------------------
    # Groupby
    # ------------------------------------------------------------------
    def groupby(self, by):
        gb = self.df.groupby(by)
        return PandasGroupBy(gb)

    # ------------------------------------------------------------------
    # Joins
    # ------------------------------------------------------------------
    def merge(self, right, on=None, left_on=None, right_on=None,
              how="inner", suffixes=("_x", "_y")):
        merged  = self.df.merge(
            right.df,
            how=how,
            on=on,
            left_on=left_on,
            right_on=right_on,
            suffixes=suffixes,
        )
        return PandasTable(merged)

    # ------------------------------------------------------------------
    # Index / naming
    # ------------------------------------------------------------------
    def reset_index(self, drop=False):
        df = self.df.reset_index(drop=drop)
        return PandasTable(df)

    def rename(self, mapping):
        df = self.df.rename(columns=mapping)
        return PandasTable(df)

    # ------------------------------------------------------------------
    # Iteration / size
    # ------------------------------------------------------------------
    def row_iter(self):
        """
        Yield each row as a plain dict, like the original code
        did via iterrows() in pandas.
        """
        for _, row in self.df.iterrows():
            yield row.to_dict()

    def rows_count(self):
        return self.df.shape[0]

    # ------------------------------------------------------------------
    # Column selection
    # ------------------------------------------------------------------
    def select(self, columns):
        df = self.df[columns]
        return PandasTable(df)

    @property
    def shape(self):
        return self.df.shape

    # ------------------------------------------------------------------
    # Row slicing / sorting
    # ------------------------------------------------------------------
    def slice_rows(self, start, end):
        df = self.df.iloc[start:end]
        return PandasTable(df)

    def sort_values(self, by, ascending=True):
        df = self.df.sort_values(by=by, ascending=ascending)
        return PandasTable(df)

    # ------------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------------
    def to_numpy(self):
        return self.df.to_numpy()

    def to_datetime(self, columns, format=None, errors="raise", utc=None):
        df = self.df.copy()
        for col in columns:
            df[col] = pd.to_datetime(df[col], format=format, errors=errors, utc=utc)
        return PandasTable(df)

    def to_pickle(self, path: str):
        self.df.to_pickle(path)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def head(self, n):
        df = self.df.head(n)
        return PandasTable(df)

    def __getitem__(self, col):
        """
        Return a PandasColumn wrapper around the underlying Series.
        """
        return PandasColumn(self.df[col])

    def __len__(self):
        return len(self.df)
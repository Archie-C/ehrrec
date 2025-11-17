from typing import List, Dict, Any, Callable, Optional, Union

from src.core.interfaces.groupby import GroupBy

class Table(object):
    
    def assign(self, **kwargs):
        """
        kwargs: new_col=lambda table: <column>
        """
        raise NotImplementedError
    
    def as_type(self, dtypes: Dict[str, str]):
        """
        Cast specific columns to new dtypes.
        Example:
            table.astype({"ICUSTAY_ID": "int64"})
        """
        raise NotImplementedError
    
    def columns(self) -> List[str]:
        """Return list of column names."""
        raise NotImplementedError
    
    def copy(self):
        """Deep copy the table."""
        raise NotImplementedError

    def drop(self, columns: List[str] = None, index: List[int] = None, axis: int = 1):
        """
        Drop columns or rows.

        columns: column names to drop (axis=1)
        index: row indices to drop (axis=0)
        axis: 1 = drop columns, 0 = drop rows
        """
        raise NotImplementedError

    
    def drop_duplicates(
        self,
        subset: list[str] | None = None
    ):
        raise NotImplementedError
    
    def drop_na(self):
        """Drop all rows with null values"""
        raise NotImplementedError
        
    def dtypes(self) -> Dict[str, Any]:
        """Return dict of column -> dtype."""
        raise NotImplementedError
        
    def fillna(self, method: str = None, values: Dict[str, Any] = None):
        """Fill NA with provided values or method per column."""
        raise NotImplementedError
    
    def filter(self, predicate):
        """
        predicate: function(row_dict) -> bool
        """
        raise NotImplementedError

    def groupby(self, by: list[str]) -> GroupBy:
        raise NotImplementedError

    def head(self, n):
        raise NotImplementedError
        
    def merge(
        self,
        right: "Table",
        on: list[str] | str | None = None,
        left_on: list[str] | str | None = None,
        right_on: list[str] | str | None = None,
        how: str = "inner",
        suffixes: tuple[str, str] = ("_x", "_y"),
    ):
        """Merge two tables."""
        raise NotImplementedError
    
    def rename(self, mapping: Dict[str, str]):
        """Rename columns."""
        raise NotImplementedError
        
    def reset_index(self, drop=True):
        """Reset the index and drop the old index column if needed."""
        raise NotImplementedError

    def row_iter(self):
        """Yield rows as dicts."""
        raise NotImplementedError

    def select(self, columns: List[str]):
        """Return table with only these columns."""
        raise NotImplementedError
    
    def shape(self) -> tuple[int, int]:
        """Return (rows, cols)."""
        raise NotImplementedError
    
    def slice_rows(self, start: int, end: int):
        raise NotImplementedError
    
    def sort_values(self, by, ascending=True):
        raise NotImplementedError

    def to_datetime(
        self,
        columns: list[str],
        format: str | None = None,
        errors: str = "raise",
        utc: bool | None = None,
    ):
        """
        Convert one or more columns to datetime.

        columns: list of column names
        format: optional datetime format
        errors: 'raise', 'coerce', 'ignore'
        utc: convert to UTC if backend supports it
        """
        raise NotImplementedError

    def to_numpy(self, columns):
        raise NotImplementedError

    def to_pickle(self, path: str):
        raise NotImplementedError

    def __getitem__(self, column: str):
        """Return a Column object or array-like."""
        raise NotImplementedError
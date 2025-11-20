from src.core.interfaces.loader import Loader


class CSVLoader(Loader):
    def load(self, path: str, **kwargs):

        dtype = kwargs.pop("dtype", None)

        # Pandas 3.x → cannot load category at read time
        # Convert dtype mapping "category" → "string" for read_csv
        dtype_for_read = None
        post_category_cols = []

        if dtype is not None:
            dtype_for_read = {}
            for col, dt in dtype.items():
                if dt == "category":
                    dtype_for_read[col] = "string"
                    post_category_cols.append(col)
                else:
                    dtype_for_read[col] = dt

        import pandas as pd

        if dtype_for_read is not None:
            df = pd.read_csv(path, dtype=dtype_for_read, **kwargs)
        else:
            df = pd.read_csv(path, **kwargs)

        # Convert to category after reading (Pandas 3.x safe)
        for col in post_category_cols:
            df[col] = df[col].astype("category")
        return df
class GroupBy:
    
    def agg(self, agg_dict: dict[str, str]):
        """
        Example:
            {"HADM_ID": "unique"}
        """
        raise NotImplementedError
    
    def head(self, n: int):
        raise NotImplementedError

    def size(self):
        """Return group sizes."""
        raise NotImplementedError
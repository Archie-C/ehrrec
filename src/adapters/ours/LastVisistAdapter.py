import pandas as pd
from src.adapters.BaseAdapter import BaseAdapter
from src.preprocess.BasePreprocessor import ProcessedTables


class LastVisitAdapter(BaseAdapter):
    def __init__(self):
        super().__init__()
        
    def adapt(self, data: ProcessedTables):
        train, val, test = data.tables["train"], data.tables["val"], data.tables["test"]
        
        train = self._keep_last_visit(train)
        val = self._keep_last_visit(val)
        test = self._keep_last_visit(test)
        
        vocab = data.artefacts["vocab"]
        vocab_size = (len(vocab["diag"][0]), len(vocab["proc"][0]), len(vocab["med"][0]))
        
        return train, val, test, vocab_size
        
        
    def _keep_last_visit(self, df):
        df = df.copy()
        df["ADMITTIME"] = pd.to_datetime(df["ADMITTIME"], errors="coerce")
        df = df.sort_values(["SUBJECT_ID", "ADMITTIME"]).reset_index(drop=True)
        
        data = []
        
        for _, g in df.groupby("SUBJECT_ID", sort=False):
            g = g.reset_index(drop=True)
            current_history_meds = []
            
            for _, row in g.iterrows():
                meds_ids  = row["MED_IDS"]  or []
                current_history_meds.append(meds_ids)
                if len(current_history_meds) > 1:
                    last, now = current_history_meds[-2], current_history_meds[-1]
                    data.append((last, now))
                else:
                    data.append(([], current_history_meds[-1]))
        
        return data
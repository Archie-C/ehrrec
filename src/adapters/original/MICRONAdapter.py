import pandas as pd

from typing import List, Tuple, Dict

from src.adapters.BaseAdapter import BaseAdapter
from src.preprocess.BasePreprocessor import ProcessedTables

Visit = Tuple[List[int], List[int]]
Prefix = List[Visit]
Target = List[int]
Sample = Tuple[Prefix, Target]

class MICRONAdapter(BaseAdapter):
    def __init__(self):
        super().__init__()
        
    def adapt(self, data: ProcessedTables):
        train = data.tables["train"]
        val = data.tables["val"]
        test = data.tables["test"]
        
        train_samples = self._build_samples(train)
        val_samples = self._build_samples(val)
        test_samples = self._build_samples(test)
        
        vocab = data.artefacts["vocab"]
        vocab_size = (len(vocab["diag"][0]), len(vocab["proc"][0]), len(vocab["med"][0]))
        
        return train_samples, val_samples, test_samples, vocab_size
        
    def _build_samples(self, df: pd.DataFrame):
        df = df.copy()
        df["ADMITTIME"] = pd.to_datetime(df["ADMITTIME"], errors="coerce")
        df = df.sort_values(["SUBJECT_ID", "ADMITTIME"]).reset_index(drop=True)
        
        samples: List[Sample] = []
        
        for _, g in df.groupby("SUBJECT_ID", sort=False):
            g = g.reset_index(drop=True)
            
            visits: List[Visit] = []
            meds: List[List[int]] = []
            
            for _, row in g.iterrows():
                diag_ids = row["DIAG_IDS"] or []
                proc_ids = row["PROC_IDS"] or []
                med_ids  = row["MED_IDS"]  or []

                visits.append((diag_ids, proc_ids))
                meds.append(med_ids)
            
            for t in range(len(visits)):
                if t == 0:
                    prefix: Prefix = [visits[t]]
                else:
                    prefix = [visits[t - 1], visits[t]]

                target: Target = meds[t]
                samples.append((prefix, target))

        return samples
        
        
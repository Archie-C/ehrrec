from typing import Dict, List, Optional, Tuple
import pandas as pd

from src.adapters.BaseAdapter import BaseAdapter
from src.preprocess.BasePreprocessor import ProcessedTables

Visit = List[List[int]]                       # [diag, proc, med]
TimedVisit = Tuple[pd.Timestamp, Visit]       # (admittime, visit)
PatientMap = Dict[int, List[TimedVisit]]
Sample = Tuple[List[Visit], List[int]]

class GamenetAdapter(BaseAdapter):
    def __init__(self):
        super().__init__()
    
    def adapt(self, data: ProcessedTables):
        train = data.tables["train"]
        val = data.tables["val"]
        test = data.tables["test"]
        
        train_map = self._convert_df_into_visit(train)
        val_map = self._convert_df_into_visit(val)
        test_map = self._convert_df_into_visit(test)
        
        train_samples, val_samples, test_samples = self._stitch_and_build_samples(train_map, val_map, test_map)
        
        vocab = data.artefacts["vocab"]
        vocab_size = (len(vocab["diag"][0]), len(vocab["proc"][0]), len(vocab["med"][0]))
        
        return train_samples, val_samples, test_samples, vocab_size
    
    def _convert_df_into_visit(self, df: pd.DataFrame):
        df = df.copy()
        df["ADMITTIME"] = pd.to_datetime(df["ADMITTIME"], errors="coerce")
        df = df.sort_values(["SUBJECT_ID", "ADMITTIME"]).reset_index(drop=True)

        out: PatientMap = {}
        for _, row in df.iterrows():
            sid = int(row["SUBJECT_ID"])
            t = row["ADMITTIME"]
            visit: Visit = [list(row["DIAG_IDS"]), list(row["PROC_IDS"]), list(row["MED_IDS"])]
            out.setdefault(sid, []).append((t, visit))
        return out
        
    def _stitch_and_build_samples(self, train_map, val_map, test_map):
        sids = set(train_map) | set(val_map) | set(test_map)

        train_samples: List[Sample] = []
        val_samples: List[Sample] = []
        test_samples: List[Sample] = []

        for sid in sids:
            train_visits = [v for _, v in sorted(train_map.get(sid, []), key=lambda x: x[0])]
            val_visits   = [v for _, v in sorted(val_map.get(sid, []),   key=lambda x: x[0])]
            test_visits  = [v for _, v in sorted(test_map.get(sid, []),  key=lambda x: x[0])]

            val_visit: Optional[Visit]  = val_visits[-1]  if len(val_visits)  else None
            test_visit: Optional[Visit] = test_visits[-1] if len(test_visits) else None

            # TRAIN: within-train history only
            for t in range(len(train_visits)):
                prefix = train_visits[: t + 1]
                target = train_visits[t][2]
                train_samples.append((prefix, target))

            # VAL: include train history + val visit
            if val_visit is not None:
                prefix_val = train_visits + [val_visit]
                val_samples.append((prefix_val, val_visit[2]))

            # TEST: include train history + (val if present) + test visit
            if test_visit is not None:
                prefix_test = train_visits + ([val_visit] if val_visit is not None else []) + [test_visit]
                test_samples.append((prefix_test, test_visit[2]))

        return train_samples, val_samples, test_samples
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.adapters.BaseAdapter import BaseAdapter
from src.preprocess.BasePreprocessor import ProcessedTables

Visit = List[List[int]]                       # [diag, proc, med]
TimedVisit = Tuple[pd.Timestamp, Visit]       # (admittime, visit)
PatientMap = Dict[int, List[TimedVisit]]
Sample = Tuple[List[Visit], List[int]]

class GamenetFastAdapter(BaseAdapter):
    def __init__(self):
        super().__init__()
    
    def adapt(self, data: ProcessedTables):
        train = data.tables["train"]
        val = data.tables["val"]
        test = data.tables["test"]
        
        ehr_adj = data.artefacts["ehr_adj"]
        ddi_adj = data.artefacts["ddi_adj"]
        ehr_adj = self._convert_to_sparse(ehr_adj)
        ddi_adj = self._convert_to_sparse(ddi_adj)
        
        vocab = data.artefacts["vocab"]
        vocab_size = (len(vocab["diag"][0]), len(vocab["proc"][0]), len(vocab["med"][0]))
        
        train_data = self._build_tensors_and_masks(train, vocab_size, window_size=5)
        val_data = self._build_tensors_and_masks(val, vocab_size, window_size=5)
        test_data = self._build_tensors_and_masks(test, vocab_size, window_size=5)
        
        return train_data, val_data, test_data, vocab_size, ddi_adj, ehr_adj
    
    def _convert_to_sparse(self, data):
        data = torch.tensor(data)
        data = data.to_sparse()
        return data
    
    def _build_tensors_and_masks(self, df: pd.DataFrame, vocab_size, window_size: int = 5):
        df = df.copy()
        df["ADMITTIME"] = pd.to_datetime(df["ADMITTIME"], errors="coerce")
        df = df.sort_values(["SUBJECT_ID", "ADMITTIME"]).reset_index(drop=True)
        
        data = []
        
        for _, g in df.groupby("SUBJECT_ID", sort=False):
            g = g.reset_index(drop=True)
            
            current_history_diag = []
            current_history_proc = []
            current_history_meds = []
            
            for _, row in g.iterrows():
                diag_ids = row["DIAG_IDS"] or []
                proc_ids = row["PROC_IDS"] or []
                meds_ids  = row["MED_IDS"]  or []
                
                diag_multi_hot = self._to_multi_hot(diag_ids, vocab_size[0])
                proc_multi_hot = self._to_multi_hot(proc_ids, vocab_size[1])
                meds_multi_hot = self._to_multi_hot(meds_ids, vocab_size[2])
                
                current_history_diag.append(diag_multi_hot)
                current_history_proc.append(proc_multi_hot)
                current_history_meds.append(meds_multi_hot)
                
                # window the histories
                windowed_diag = self._window(current_history_diag, window_size)
                windowed_proc = self._window(current_history_proc, window_size)
                windowed_meds = self._window(current_history_meds, window_size)
                
                # pad the histories
                padded_diag = self._create_padding(windowed_diag, vocab_size[0], window_size)
                padded_proc = self._create_padding(windowed_proc, vocab_size[1], window_size)
                padded_meds = self._create_padding(windowed_meds, vocab_size[2], window_size)
                
                item = (padded_diag, padded_proc, padded_meds)
                
                data.append(item)
        
        return data
        
        
    def _to_multi_hot(self, L: List, length: int):
        t = torch.zeros(length)
        t[L] = 1
        return t
    
    def _create_padding(self, current_history: List[torch.tensor], n_items, window_size: int = 5):
        while len(current_history) < window_size:
            padding = torch.zeros(n_items)
            current_history.insert(0, padding)
        return torch.stack(current_history)
    
    def _window(self, arr, window_size: int = 5):
        if len(arr) <= window_size:
            return arr
        return arr[-window_size:]
    
class GamenetFastDataset(Dataset):
    def __init__(self, data, device):
        self.data = data
        self.device = device 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve the data and move to the correct device
        diag_history, proc_history, med_history = self.data[idx]

        # Move data to the device
        diag_history = diag_history.to(self.device)
        proc_history = proc_history.to(self.device)
        med_history = med_history.to(self.device)

        return diag_history, proc_history, med_history
import torch
from typing import List
from torch.utils.data import Dataset

from src.adapters.BaseAdapter import BaseAdapter
from src.preprocess.BasePreprocessor import ProcessedTables


class MultiHotAdapter(BaseAdapter):
    def __init__(self):
        super().__init__()
    
    def adapt(self, data: ProcessedTables):
        train = data.tables["train"]
        val = data.tables["val"]
        test = data.tables["test"]
        
        vocab = data.artefacts["vocab"]
        vocab_size = (len(vocab["diag"][0]), len(vocab["proc"][0]), len(vocab["med"][0]))
        
        train["DIAG_MULTI"] = train["DIAG_IDS"].apply(lambda x: self._to_multi_hot(x, vocab_size[0]))
        train["PROC_MULTI"] = train["PROC_IDS"].apply(lambda x: self._to_multi_hot(x, vocab_size[1]))
        train["MEDS_MULTI"] = train["MED_IDS"].apply(lambda x: self._to_multi_hot(x, vocab_size[2]))
        
        val["DIAG_MULTI"] = val["DIAG_IDS"].apply(lambda x: self._to_multi_hot(x, vocab_size[0]))
        val["PROC_MULTI"] = val["PROC_IDS"].apply(lambda x: self._to_multi_hot(x, vocab_size[1]))
        val["MEDS_MULTI"] = val["MED_IDS"].apply(lambda x: self._to_multi_hot(x, vocab_size[2]))
        
        test["DIAG_MULTI"] = test["DIAG_IDS"].apply(lambda x: self._to_multi_hot(x, vocab_size[0]))
        test["PROC_MULTI"] = test["PROC_IDS"].apply(lambda x: self._to_multi_hot(x, vocab_size[1]))
        test["MEDS_MULTI"] = test["MED_IDS"].apply(lambda x: self._to_multi_hot(x, vocab_size[2]))
        
        
        return train, val, test, vocab_size
        
    def _to_multi_hot(self, L: List, length: int):
        t = torch.zeros(length)
        t[L] = 1
        return t
    
    
class MultiHotDataset(Dataset):
    def __init__(self, data, device):
        self.data = data
        self.device = device 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve the data and move to the correct device
        diag_history, proc_history, med_history = self.data["DIAG_MULTI"][idx], self.data["PROC_MULTI"][idx], self.data["MEDS_MULTI"][idx]

        # Move data to the device
        diag_history = diag_history.to(device=self.device, dtype=torch.float32)
        proc_history = proc_history.to(device=self.device, dtype=torch.float32)
        med_history = med_history.to(device=self.device, dtype=torch.float32)

        return diag_history, med_history
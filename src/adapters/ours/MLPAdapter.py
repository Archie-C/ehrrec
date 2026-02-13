import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from src.adapters.BaseAdapter import BaseAdapter
from src.preprocess.BasePreprocessor import ProcessedTables


class MLPAdapter(BaseAdapter):
    def __init__(self):
        super().__init__()
    
    def adapt(self, data: ProcessedTables, device, batch_size: int):
        train = data.tables["train"]
        val = data.tables["val"]
        test = data.tables["test"]
                
        vocab = data.artefacts["vocab"]
        vocab_size = (len(vocab["diag"][0]), len(vocab["proc"][0]), len(vocab["med"][0]))
        
        train_samples, val_samples, test_samples = self._convert_to_X_y(train, vocab_size), self._convert_to_X_y(val, vocab_size), self._convert_to_X_y(test, vocab_size)
        train_dataset, val_dataset, test_dataset = self._create_dataset(train_samples, device), self._create_dataset(val_samples, device), self._create_dataset(test_samples, device)
        train_loader, val_loader, test_loader = self._create_dataloader(train_dataset, batch_size), self._create_dataloader(val_dataset, batch_size), self._create_dataloader(test_dataset, batch_size)
        return train_loader, val_loader, test_loader, vocab_size
    
    def _convert_to_X_y(self, df: pd.DataFrame, vocab_size):
        df = df.copy()
        data = []
        
        for _, row in df.iterrows():
            diag = torch.zeros(vocab_size[0])
            proc = torch.zeros(vocab_size[1])
            meds = torch.zeros(vocab_size[2])
            
            diag[list(row["DIAG_IDS"])] = 1
            proc[list(row["PROC_IDS"])] = 1
            meds[list(row["MED_IDS"])] = 1
            
            X = torch.cat((diag, proc))
            data.append((X, meds))
        return data
    
    def _create_dataset(self, data, device):
        dataset = MLPDataset(data, device)
        return dataset
        
    def _create_dataloader(self, dataset, batch_size):
        return DataLoader(dataset, batch_size, shuffle=True)

class MLPDataset(Dataset):
    def __init__(self, data, device):
        self.data = data
        self.device = device 
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve the data and move to the correct device
        X, y = self.data[idx]

        # Move data to the device
        X = X.to(self.device)
        y = y.to(self.device)

        return X, y
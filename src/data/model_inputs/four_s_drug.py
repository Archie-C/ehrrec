
import logging
from typing import List
import numpy as np
import torch

from src.core.interfaces.preprocessor import Preprocessor

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm optional
    tqdm = None


class FourSDrugModelInputBuilder(Preprocessor):
    """Placeholder for additional 4SDrug-specific input shaping."""
    def __init__(self, log_level: int = logging.INFO, show_progress: bool = True, batch_size: int = 50):
        self.batch_size = batch_size
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
            self.logger.propagate = False

        self.show_progress = show_progress and (tqdm is not None)
        if show_progress and tqdm is None:
            self.logger.warning("show_progress=True but tqdm not available; progress bars disabled.")

    def run(self, *, source: str, context, train_data, val_data, test_data, voc_size=None):
        if source != "gamenet":
            raise ValueError(f"Unsupported source '{source}' for GAMENet model inputs")
        sizes = voc_size or context.vocab_sizes()
        train = self._generate_batched_data(train_data, sizes)
        train["similar_idx"] = self._build_similar_sets_idx(train["sym"])
        test = self._build_split(test_data, sizes)
        val = self._build_split(val_data, sizes)
        return train, test, val
    
    def _build_split(self, data, voc_size):
        X, y = [], []
        patient_counts: List[int] = []
        for patient in data:
            patient_counts.append(len(patient))
            for adm in patient:
                X_i = adm[0]
                y_i = self._multihot_encode(adm[2], voc_size[2])
                X.append(X_i)
                y.append(y_i)
        return {
            "sym": X,
            "drug": y,
        }
    
    def _build_similar_sets_idx(self,sym_train):
        similar_sets = [[] for _ in range(len(sym_train))]
        for i in range(len(sym_train)):
            for j in range(len(sym_train[i])):
                similar_sets[i].append(j)
                
        for idx, batch in enumerate(sym_train):
            if len(batch) <=2 or len(batch[0]) <= 2: 
                continue
            
            batch_sets = [set(sym_set) for sym_set in batch]
            for i in range(len(batch_sets)):
                max_intersection = 0
                for j in range(len(batch_sets)):
                    if i == j:
                        continue
                    if len(batch_sets[i] & batch_sets[j]) > max_intersection:
                        max_intersection = len(batch_sets[i] & batch_sets[j])
                        similar_sets[idx][i] = j
        return similar_sets
    
    def _generate_batched_data(self, train_data, voc_size):
        D, P, M = voc_size
        size_dict, drug_dict = {}, {}
        sym_sets, drug_sets = [], []
        s_set_num = 0
        for patient in train_data:
            for adm in patient:
                syms, drugs = adm[0], adm[2]
                sym_sets.append(syms)
                drug_sets.append(drugs)
                s_set_num += 1
            
        sym_count = np.zeros(D)
        for patient in train_data:
            for adm in patient:
                syms = adm[0]
                sym_count[syms] += 1
        
        for patient in train_data:
            for adm in patient:
                syms, drugs = adm[0], adm[2]
                drug_multihot = np.zeros(M)
                drug_multihot[drugs] = 1
                if size_dict.get(len(syms)):
                    size_dict[len(syms)].append(syms)
                    drug_dict[len(syms)].append(drug_multihot)
                else:
                    size_dict[len(syms)] = [syms]
                    drug_dict[len(syms)] = [drug_multihot]
                    keys, count = list(size_dict.keys()), 0
                
        keys.sort()
        new_s_set, new_d_set = [], []
        for size in keys:
            if size <= 2: 
                continue
            for (syms, drugs) in zip(size_dict[size], drug_dict[size]):
                syms = np.array(syms)
                cnt, del_nums = torch.from_numpy(sym_count[syms]), int(max(1, len(syms) * 0.2))
                if del_nums == 1:
                    del_idx = torch.multinomial(cnt, len(syms) - del_nums)
                    remained = syms[del_idx.numpy()]
                    remained = remained.tolist()
                    new_s_set.append(remained)
                    new_d_set.append(drugs)
                else:
                    for _ in range(min(del_nums, 3)):
                        del_num = np.random.randint(1, del_nums)
                        del_idx = torch.multinomial(cnt, len(syms) - del_num)
                        remained = syms[del_idx.numpy()]
                        remained = remained.tolist()
                        new_s_set.append(remained)
                        new_d_set.append(drugs)

        for (remained, drugs) in zip(new_s_set, new_d_set):
            if size_dict.get(len(remained)) is None:
                count += 1
                size_dict[len(remained)] = [remained]
                drug_dict[len(remained)] = [drugs]
            elif remained not in size_dict[len(remained)]:
                count += 1
                size_dict[len(remained)].append(remained)
                drug_dict[len(remained)].append(drugs)
        
                sym_train, drug_train = [], []
                
        keys = list(size_dict.keys())
        keys.sort()
        for size in keys:
            num_size = len(size_dict[size])
            batch_num, start_idx = num_size // self.batch_size, 0
            if num_size % self.batch_size != 0:
                batch_num += 1
            for i in range(batch_num):
                if i == batch_num:
                    syms, drugs = size_dict[size][start_idx:], drug_dict[size][start_idx:]
                else:
                    syms, drugs = size_dict[size][start_idx:start_idx + self.batch_size], drug_dict[size][start_idx:start_idx + self.batch_size]
                    start_idx += self.batch_size
                sym_train.append(syms)
                drug_train.append(drugs)
        
        return {
            "sym": sym_train,
            "drug": drug_train
        }


    def _multihot_encode(self, data, size):
        x = np.zeros(size, dtype=np.float32)
        for m in data:
            x[m] = 1.0
        return x


from typing import Dict, List, Optional, Tuple
import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS
import numpy as np
import torch
from ogb.utils import smiles2graph
from torch_geometric.data import Data

from src.adapters.BaseAdapter import BaseAdapter
from src.preprocess.BasePreprocessor import ProcessedTables

Visit = List[List[int]]                       # [diag, proc, med]
TimedVisit = Tuple[pd.Timestamp, Visit]       # (admittime, visit)
PatientMap = Dict[int, List[TimedVisit]]
Sample = Tuple[List[Visit], List[int]]

SPECIAL_TOKENS = {"<PAD>", "<UNK>"}

class MoleRecAdapter(BaseAdapter):
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
        
        molecule = data.artefacts["molecule"]
        ddi_mask_H, substructure_smiles = self._create_mask_and_substructure_smiles(molecule, vocab["med"][1])
        
        average_projection, smiles_list = self._build_projection_smiles(molecule, vocab["med"][1])
        molecule_graphs = self._graph_from_smile(smiles_list)
        substruct_graphs = self._graph_from_smile(substructure_smiles)
        
        return train_samples, val_samples, test_samples, vocab_size, average_projection, molecule_graphs, substruct_graphs, ddi_mask_H
    
    def _build_projection_smiles(self, molecule, med_idx_to_code):
        
        average_index, smiles_all = [], []
        
        for _, atc3 in med_idx_to_code.items():
            
            if atc3 in SPECIAL_TOKENS:
                average_index.append(0)
                continue
            smilesList = list(molecule.get(atc3, []))
            
            counter = 0
            for smiles in smilesList:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    smiles_all.append(smiles)
                    counter += 1
                else:
                    print("[SMILES]", smiles)
                    print("[ERROR] Invalid smiles")
            average_index.append(counter)
        
        n_col = sum(average_index)
        n_row = len(average_index)
        
        average_projection = np.zeros((n_row, n_col))
        col_counter = 0
        
        for i, item in enumerate(average_index):
            if item > 0:
                average_projection[i, col_counter: col_counter + item] = 1 / item
                col_counter += item
        print("Smiles Num:{}".format(len(smiles_all)))
        print("n_col:{}".format(n_col))
        print("n_row:{}".format(n_row))

        return torch.FloatTensor(average_projection), smiles_all
    
    def _graph_from_smile(self, smiles_list):
        edge_indexes, edge_features, node_features, last_node, batch = [], [], [], 0, []
        graphs = [smiles2graph(x) for x in smiles_list]
        for idx, graph in enumerate(graphs):
            edge_indexes.append(graph["edge_index"] + last_node)
            edge_features.append(graph["edge_feat"])
            node_features.append(graph["node_feat"])
            last_node += graph["num_nodes"]
            batch.append(np.ones(graph['num_nodes'], dtype=np.int64) * idx)
        
        result = {
            "edge_index": np.concatenate(edge_indexes, axis=-1),
            "edge_attr": np.concatenate(edge_features, axis=0),
            "batch": np.concatenate(batch, axis=0),
            "x": np.concatenate(node_features, axis=0)
        }
        
        result = {k: torch.from_numpy(v) for k, v in result.items()}
        result['num_nodes'] = last_node
        return Data(**result)
    
    def _create_mask_and_substructure_smiles(self, idx_to_smiles, med_idx_to_code):
        fraction = []
        for k, v in med_idx_to_code.items():
            tempF = set()
            if v == "<PAD>" or v == "<UNK>":
                fraction.append(tempF)
                continue
            try:
                for SMILES in idx_to_smiles[v]:
                    try:
                        m = BRICS.BRICSDecompose(Chem.MolFromSmiles(SMILES))
                        for frac in m:
                            tempF.add(frac)
                    except Exception:
                        pass
            except Exception:
                pass
            
            fraction.append(tempF)
        
        fracSet = []
        for i in fraction:
            fracSet += i
        fracSet = list(set(fracSet))
        
        ddi_matrix = torch.zeros((len(med_idx_to_code), len(fracSet)))
        for i, fracList in enumerate(fraction):
            for frac in fracList:
                ddi_matrix[i, fracSet.index(frac)] = 1
        
        return ddi_matrix, fracSet

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
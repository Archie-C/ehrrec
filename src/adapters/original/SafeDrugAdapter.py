from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS
import torch
import numpy as np

from src.adapters.BaseAdapter import BaseAdapter
from src.preprocess.BasePreprocessor import ProcessedTables

Visit = List[List[int]]                       # [diag, proc, med]
TimedVisit = Tuple[pd.Timestamp, Visit]       # (admittime, visit)
PatientMap = Dict[int, List[TimedVisit]]
Sample = Tuple[List[Visit], List[int]]

class SafeDrugAdapter(BaseAdapter):
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
        ddi_mask_H = self._create_mask(molecule, vocab["med"][1])
        
        MPNNSet, n_fingerprint, average_projection = self._buildMPNN(molecule, vocab["med"][1], 2)
        
        return train_samples, val_samples, test_samples, vocab_size, ddi_mask_H, MPNNSet, n_fingerprint, average_projection
    
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
    
    def _create_mask(self, idx_to_smiles, med_idx_to_code):
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
        
        return ddi_matrix
    
    def _buildMPNN(self, molecule, med_voc, radius=1, device="cpu:0"):

        atom_dict = defaultdict(lambda: len(atom_dict))
        bond_dict = defaultdict(lambda: len(bond_dict))
        fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
        edge_dict = defaultdict(lambda: len(edge_dict))
        MPNNSet, average_index = [], []

        print (len(med_voc.items()))
        for index, ndc in med_voc.items():
            
            if ndc == "<PAD>" or ndc == "<UNK>":
                smilesList = []
            
            else:
                smilesList = list(molecule.get(ndc, []))

            """Create each data with the above defined functions."""
            counter = 0 # counter how many drugs are under that ATC-3
            for smiles in smilesList:
                try:
                    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
                    atoms = self._create_atoms(mol, atom_dict)
                    molecular_size = len(atoms)
                    i_jbond_dict = self._create_ijbonddict(mol, bond_dict)
                    fingerprints = self._extract_fingerprints(radius, atoms, i_jbond_dict,
                                                        fingerprint_dict, edge_dict)
                    adjacency = Chem.GetAdjacencyMatrix(mol)
                    # if fingerprints.shape[0] == adjacency.shape[0]:
                    for _ in range(adjacency.shape[0] - fingerprints.shape[0]):
                        fingerprints = np.append(fingerprints, 1)
                    fingerprints = torch.LongTensor(fingerprints).to(device)
                    adjacency = torch.FloatTensor(adjacency).to(device)
                    MPNNSet.append((fingerprints, adjacency, molecular_size))
                    counter += 1
                except:
                    continue
            average_index.append(counter)

            """Transform the above each data of numpy
            to pytorch tensor on a device (i.e., CPU or GPU).
            """

        N_fingerprint = len(fingerprint_dict)

        # transform into projection matrix
        n_col = sum(average_index)
        n_row = len(average_index)

        average_projection = np.zeros((n_row, n_col))
        col_counter = 0
        for i, item in enumerate(average_index):
            if item > 0:
                average_projection[i, col_counter : col_counter + item] = 1 / item
            col_counter += item

        return MPNNSet, N_fingerprint, torch.FloatTensor(average_projection)
    
    def _extract_fingerprints(self, radius, atoms, i_jbond_dict, fingerprint_dict, edge_dict):
        """Extract the fingerprints from a molecular graph
        based on Weisfeiler-Lehman algorithm.
        """

        if (len(atoms) == 1) or (radius == 0):
            nodes = [fingerprint_dict[a] for a in atoms]

        else:
            nodes = atoms
            i_jedge_dict = i_jbond_dict

            for _ in range(radius):

                """Update each node ID considering its neighboring nodes and edges.
                The updated node IDs are the fingerprint IDs.
                """
                nodes_ = []
                for i, j_edge in i_jedge_dict.items():
                    neighbors = [(nodes[j], edge) for j, edge in j_edge]
                    fingerprint = (nodes[i], tuple(sorted(neighbors)))
                    nodes_.append(fingerprint_dict[fingerprint])

                """Also update each edge ID considering
                its two nodes on both sides.
                """
                i_jedge_dict_ = defaultdict(lambda: [])
                for i, j_edge in i_jedge_dict.items():
                    for j, edge in j_edge:
                        both_side = tuple(sorted((nodes[i], nodes[j])))
                        edge = edge_dict[(both_side, edge)]
                        i_jedge_dict_[i].append((j, edge))

                nodes = nodes_
                i_jedge_dict = i_jedge_dict_

        return np.array(nodes)

    def _create_atoms(self, mol, atom_dict):
        """Transform the atom types in a molecule (e.g., H, C, and O)
        into the indices (e.g., H=0, C=1, and O=2).
        Note that each atom index considers the aromaticity.
        """
        atoms = [a.GetSymbol() for a in mol.GetAtoms()]
        for a in mol.GetAromaticAtoms():
            i = a.GetIdx()
            atoms[i] = (atoms[i], 'aromatic')
        atoms = [atom_dict[a] for a in atoms]
        return np.array(atoms)

    def _create_ijbonddict(self, mol, bond_dict):
        """Create a dictionary, in which each key is a node ID
        and each value is the tuples of its neighboring node
        and chemical bond (e.g., single and double) IDs.
        """
        i_jbond_dict = defaultdict(lambda: [])
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            bond = bond_dict[str(b.GetBondType())]
            i_jbond_dict[i].append((j, bond))
            i_jbond_dict[j].append((i, bond))
        return i_jbond_dict
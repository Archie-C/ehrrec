import ast
import pickle
import pandas as pd

from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, Set

from src.load.BaseLoader import AbstractLoader, TablesResponse

@dataclass(frozen=True)
class MIMIC3LoaderConfig:
    root: Path = Path("data/raw/MIMIC-III").resolve()
    
    diagnoses_file: str = "DIAGNOSES_ICD.csv"
    procedures_file: str = "PROCEDURES_ICD.csv"
    medications_file: str = "PRESCRIPTIONS.csv"
    admissions_file: str = "ADMISSIONS.csv"
    
    ndc2rxcui_file: str = "ndc2RXCUI.txt"
    rxcui2atc3_file: str = "RXCUI2atc3.csv"
    
    idx2drug_file: str = "idx2drug.pkl"

    ddi_file: str = "drug-DDI_gamenet.csv"
    drug_atc_file: str = "drug-atc_gamenet.csv"
    
    ddi_mask_file: str = "ddi_mask_H.pkl"
    
    molecule_path: str = "idx2SMILES.pkl"
    
    @classmethod
    def from_config(cls, data_config: Dict):

        return cls(
            root=Path(data_config.get("data_dir", cls.root)).resolve(),
            diagnoses_file=data_config.get("diagnoses_file", cls.diagnoses_file),
            procedures_file=data_config.get("procedures_file", cls.procedures_file),
            medications_file=data_config.get("medications_file", cls.medications_file),
            admissions_file=data_config.get("admissions_file", cls.admissions_file),
            ndc2rxcui_file=data_config.get("ndc_to_rxcui_file", cls.ndc2rxcui_file),
            rxcui2atc3_file=data_config.get("rxcui_to_atc3_file", cls.rxcui2atc3_file),
            idx2drug_file=data_config.get("idx2drug_file", cls.idx2drug_file),
            ddi_file=data_config.get("ddi_file", cls.ddi_file),
            drug_atc_file=data_config.get("drug_atc_file", cls.drug_atc_file),
            ddi_mask_file=data_config.get("ddi_mask_file", cls.ddi_mask_file),
            molecule_path=data_config.get("molecule_path", cls.molecule_path),
        )


def _path(cfg: MIMIC3LoaderConfig, filename: str) -> Path:
    return cfg.root / filename

class MIMIC3Loader(AbstractLoader):
    def __init__(self, cfg: MIMIC3LoaderConfig):
        super().__init__()
        self.cfg = cfg
        
    def _load_impl(self) -> TablesResponse:
        diagnoses_df = pd.read_csv(_path(self.cfg, self.cfg.diagnoses_file))
        procedures_df = pd.read_csv(_path(self.cfg, self.cfg.procedures_file))
        medications_df = pd.read_csv(_path(self.cfg, self.cfg.medications_file), low_memory=False)
        admissions_df = pd.read_csv(_path(self.cfg, self.cfg.admissions_file))
        
        with open(_path(self.cfg, self.cfg.ndc2rxcui_file), "r") as f:
            txt = f.read().strip()
            ndc_rxcui = ast.literal_eval(txt) 
        
        rxcui_atc3 = pd.read_csv(_path(self.cfg, self.cfg.rxcui2atc3_file), dtype=str)
        
        with open(_path(self.cfg, self.cfg.idx2drug_file), "rb") as f:
            idx2drug: Dict[int, str] = pickle.load(f)
        
        ddi_path = _path(self.cfg, self.cfg.ddi_file)
        ddi_df = pd.read_csv(ddi_path)
        
        drug_atc_path = _path(self.cfg, self.cfg.drug_atc_file)
        cid2atc3 = self._read_cid_to_atc3(drug_atc_path)
        
        with open(_path(self.cfg, self.cfg.molecule_path), "rb") as f:
            molecule = pickle.load(f)

        tables = {
            "diagnoses": diagnoses_df,
            "procedures": procedures_df,
            "prescriptions": medications_df,
            "admissions": admissions_df
        }
        artefacts = {
            "ndc_rxcui": ndc_rxcui,
            "rxcui_atc3": rxcui_atc3,
            "idx2drug": idx2drug,
            "ddi": ddi_df,
            "cid2atc3": cid2atc3,
            "molecule": molecule,
        }
        meta = {
            "config": self.cfg,
            "dataset": "mimic-iii",
            "version": "1.4",
        }
        return TablesResponse(tables=tables, meta=meta, artefacts=artefacts)
    
    
    def _read_cid_to_atc3(self, path: str | Path) -> Dict[str, Set[str]]:
        """
        Parses GAMENet cid_atc file:
        CID...,ATC4,ATC4,ATC4...
        and returns CID -> set(ATC3) where ATC3 is first 4 chars.
        """
        path = Path(path)
        cid2atc3: Dict[str, Set[str]] = defaultdict(set)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",") if p.strip()]
                if len(parts) < 2:
                    continue
                cid = parts[0]
                for atc in parts[1:]:
                    cid2atc3[cid].add(atc[:4])
        return dict(cid2atc3)
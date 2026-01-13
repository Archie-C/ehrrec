from src.core.interfaces.loader import Loader
from src.data.loaders.csv_loader import CSVLoader
from src.data.loaders.text_loader import TextLoader
from src.data.raw import register_raw_loader


class MIMIC3DatasetLoader(Loader):
    def __init__(self, csv_loader=None, text_loader=None, paper="gamenet"):
        self.csv = csv_loader or CSVLoader()
        self.text = text_loader or TextLoader()
        self.paper = paper
    
    def load(self, paths, kwargs=None):
        medications_params = kwargs.get("medications", {}) if kwargs else {}
        diagnoses_params = kwargs.get("diagnoses", {}) if kwargs else {}
        procedures_params = kwargs.get("procedures", {}) if kwargs else {}
        
        result = {
            "medications": self.csv.load(paths["medications"], **medications_params),
            "diagnoses": self.csv.load(paths["diagnoses"], **diagnoses_params),
            "procedures": self.csv.load(paths["procedures"], **procedures_params),
        }
            
        if self.paper == "gamenet":
            rxnorm_to_atc_params = kwargs.get("rxnorm_to_atc", {}) if kwargs else {}
            result["rxnorm_to_atc"] = self.csv.load(paths["rxnorm_to_atc"], **rxnorm_to_atc_params)
            result["cid_to_atc"] = paths["cid_to_atc"]
            result["ndc_to_rxnorm"] = paths["ndc_to_rxnorm"]
            result["ddi"] = paths["ddi"]
            
            return result
        elif self.paper == "safedrug":
            result["RXCUI2atc4"] = self.csv.load(paths["RXCUI2atc4"])
            result["ndc2RXCUI"] = paths["ndc2RXCUI"]
            result["drug2atc"] = paths["drug2atc"]
            result["ddi"] = paths["ddi"]
            result["idx2drug"] = paths["idx2drug"]
            return result
        else:
            raise ValueError(f"Paper {self.paper} not supported")
    

@register_raw_loader("mimic3_gamenet")
def _build_mimic3_loader(cfg, **_):
    return MIMIC3DatasetLoader(paper="gamenet")

@register_raw_loader("mimic3_safedrug")
def _build_mimic3_loader(cfg, **_):
    return MIMIC3DatasetLoader(paper="safedrug")


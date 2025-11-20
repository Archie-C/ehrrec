from src.core.interfaces.loader import Loader
from src.data.loaders.csv_loader import CSVLoader
from src.data.loaders.text_loader import TextLoader
from src.data.raw import register_raw_loader


class MIMIC3DatasetLoader(Loader):
    def __init__(self, csv_loader=None, text_loader=None):
        self.csv = csv_loader or CSVLoader()
        self.text = text_loader or TextLoader()
    
    def load(self, paths, kwargs=None):
        medications_params = kwargs.get("medications", {}) if kwargs else {}
        diagnoses_params = kwargs.get("diagnoses", {}) if kwargs else {}
        procedures_params = kwargs.get("procedures", {}) if kwargs else {}
        rxnorm_to_atc_params = kwargs.get("rxnorm_to_atc", {}) if kwargs else {}
        
        return {
            "medications": self.csv.load(paths["medications"], **medications_params),
            "diagnoses": self.csv.load(paths["diagnoses"], **diagnoses_params),
            "procedures": self.csv.load(paths["procedures"], **procedures_params),
            "rxnorm_to_atc": self.csv.load(paths["rxnorm_to_atc"], **rxnorm_to_atc_params),
            "cid_to_atc": paths["cid_to_atc"],
            "ndc_to_rxnorm": paths["ndc_to_rxnorm"],
            "ddi": paths["ddi"],
        }
    

@register_raw_loader("mimic3")
def _build_mimic3_loader(cfg, **_):
    return MIMIC3DatasetLoader()

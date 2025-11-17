from src.core.interfaces.loader import Loader


class MIMIC3DatasetLoader(Loader):
    def __init__(self, csv_loader, text_loader):
        self.csv = csv_loader
        self.text= text_loader
    
    def load(self, paths, kwargs=None):
        medications_params = kwargs.get("medications", {}) if kwargs else {}
        diagnoses_params = kwargs.get("diagnoses", {}) if kwargs else {}
        procedures_params = kwargs.get("procedures", {}) if kwargs else {}
        rxnorm_to_atc_params = kwargs.get("rxnorm_to_atc", {}) if kwargs else {}
        ndc_to_rxnorm_params = kwargs.get("ndc_to_rxnorm", {}) if kwargs else {}
        ddi_params = kwargs.get("ddi", {}) if kwargs else {}
        
        return {
            "medications": self.csv.load(paths["medications"], **medications_params),
            "diagnoses": self.csv.load(paths["diagnoses"], **diagnoses_params),
            "procedures": self.csv.load(paths["procedures"], **procedures_params),
            "rxnorm_to_atc": self.csv.load(paths["rxnorm_to_atc"], **rxnorm_to_atc_params),
            "cid_to_atc": paths["cid_to_atc"],
            "ndc_to_rxnorm": self.text.load(paths["ndc_to_rxnorm"], **ndc_to_rxnorm_params),
            "ddi": self.csv.load(paths["ddi"], **ddi_params),
        }
    

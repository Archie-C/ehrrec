from core.interfaces.loader import Loader


class MIMIC3DatasetLoader(Loader):
    def __init__(self, csv_loader, text_loader):
        self.csv = csv_loader
        self.text= text_loader
    
    def load(self, paths, kwargs):
        return {
            "medications": self.csv.load(paths["medications"], kwargs["medications"]),
            "diagnoses": self.csv.load(paths["diagnoses"], kwargs["diagnoses"]),
            "procedures": self.csv.load(paths["procedures"], kwargs["procedures"]),
            "ndc_to_atc": self.csv.load(paths["ndc_to_atc"], kwargs["ndc_to_atc"]),
            "cid_to_atc": paths["cid_to_atc"],
            "ndc_to_rxnorm": self.text.load(paths["ndc_to_rxnorm"], kwargs["ndc_to_rxnorm"]),
            "ddi": self.csv.load(paths["ddi"], kwargs["ddi"]),
        }
    

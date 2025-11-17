from src.data.loaders.text_loader import TextLoader
from src.data.loaders.csv_loader import CSVLoader
from src.data.loaders.MIMIC3Datasetloader import MIMIC3DatasetLoader


def create_loader(cfg_dataset, backend):
    name = cfg_dataset.name
    
    if name == "mimic3":
        return MIMIC3DatasetLoader(
            csv_loader=CSVLoader(backend_factory=backend),
            text_loader=TextLoader()
        )
    raise ValueError(f"Unknown dataset: {name}")
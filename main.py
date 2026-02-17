from src.utils.logging import setup_logging
setup_logging()

import yaml # noqa: E402
import argparse # noqa: E402

from pathlib import Path # noqa: E402
from typing import Dict # noqa: E402

from src.utils.logging import get_logger # noqa: E402
from src.preprocess.MIMIC3Preprocessor import MIMIC3Preprocessor # noqa: E402
from src.load.MIMIC3Loader import MIMIC3Loader, MIMIC3LoaderConfig # noqa: E402

logger = get_logger("MAIN")

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config/main.yaml", help="yaml config path")

args = parser.parse_args()
config_path = args.config

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_loader(config: Dict):
    data_config = config.get("data", None)
    if data_config is None:
        raise ValueError("Unable to read the config for the loader")
    dataset = data_config.get("dataset", "mimic_iii")
    if dataset == "mimic_iii":
        loader_config = MIMIC3LoaderConfig.from_config(data_config)
        return MIMIC3Loader(cfg=loader_config)
    else:
        raise ValueError(f"Dataset: {dataset} not supported yet.")

def get_preprocessor(config: Dict):
    preprocessor_config = config.get("preprocessor", None)
    if preprocessor_config is None:
        raise ValueError("Unable to read the config for the preprocessor")
    
    preprocessor = preprocessor_config.get("dataset", "mimic_iii")
    if preprocessor == "mimic_iii":
        return MIMIC3Preprocessor()
    else:
        raise ValueError(f"Preprocessor: {preprocessor} not supported yet.")
    
def get_data_module(config: Dict):
    data_module_config = config.get("datamodule", None)
    if data_module_config is None:
        raise ValueError("Unable to read the config for the Data Module")


if __name__ == "__main__":
    config = load_config(config_path)
    logger.info(f"Loaded config from: {config_path}")
    
    loader = get_loader(config)
    data = loader.load()
    logger.info("Data loaded from files")
    
    preprocessor = get_preprocessor(config)
    processed_data = preprocessor.process(data)
    logger.info("Data processed")
    
    data_module = get_data_module(config)
    train_loader, val_loader, test_loader = data_module.load(processed_data)
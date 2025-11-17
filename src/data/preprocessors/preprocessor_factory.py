from src.data.preprocessors.GAMENet_preprocessor import GAMENetPreprocessor
import logging

def create_preprocessor(cfg_preprocessor, log_level=logging.INFO):
    name = cfg_preprocessor.name
    
    if name == "gamenet":
        return GAMENetPreprocessor(cfg_preprocessor.output_dir, cfg_preprocessor.filenames, log_level=log_level)
    
    raise ValueError(f"Unknown preprocessor: {name}")
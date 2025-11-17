from src.data.preprocessors.GAMENet_preprocessor import GAMENetPreprocessor


def create_preprocessor(cfg_preprocessor):
    name = cfg_preprocessor.name
    
    if name == "gamenet":
        return GAMENetPreprocessor(cfg_preprocessor.output_dir, cfg_preprocessor.filenames)
    
    raise ValueError(f"Unknown preprocessor: {name}")
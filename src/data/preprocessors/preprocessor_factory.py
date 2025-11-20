import logging

from src.data.preprocessors.raw.gamenet import GAMENetPreprocessor


def create_preprocessor(cfg_preprocessor, log_level=logging.INFO):
    name = cfg_preprocessor.name.lower()

    if name == "gamenet":
        return GAMENetPreprocessor(
            cfg_preprocessor.output_dir,
            cfg_preprocessor.filenames,
            log_level=log_level,
        )

    raise ValueError(f"Unknown preprocessor: {cfg_preprocessor.name}")

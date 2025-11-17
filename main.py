import hydra
from omegaconf import DictConfig

from src.data.preprocessors.preprocessor_factory import create_preprocessor
from src.data.loaders.loader_factory import create_loader
from src.core.backend.backend_factory import create_backend

@hydra.main(version_base=None, config_path='config', config_name='config')
def main(cfg: DictConfig) -> None:
    backend = create_backend(cfg.backend)
    loader = create_loader(cfg.dataset, backend)
    data = loader.load(cfg.dataset.paths, cfg.dataset.kwargs)
    preprocessor = create_preprocessor(cfg.preprocessor, log_level=cfg.logging.level)
    data = preprocessor.run(data)
    preprocessor.save(data)

if __name__ == "__main__":
    main()
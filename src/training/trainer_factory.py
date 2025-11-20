
from src.training.GAMENetTrainer import GAMENetTrainer
from src.training.KNNTrainer import KNNTrainer
from src.training.MostPopularTrainer import MostPopularTrainer


def create_trainer(cfg):
    """
    Factory that returns the correct trainer object based on cfg.trainer.name.
    """
    name = cfg.name.lower()

    if name == "gamenet":
        return GAMENetTrainer(
            epochs=cfg.epochs,
            lr=cfg.lr,
            ddi_loss=cfg.ddi_loss,
            target_ddi=cfg.target_ddi,
            ddi_decay=cfg.ddi_decay,
            save_dir=cfg.save_dir,
            log_level=cfg.log_level,
            show_progress=cfg.show_progress,
        )
    if name == "knn":
        return KNNTrainer(
            save_dir=cfg.save_dir,
            log_level=cfg.log_level,
            show_progress=cfg.show_progress,
        )
    if name == "most_popular":
        return MostPopularTrainer(
            save_dir=cfg.save_dir,
            log_level=cfg.log_level,
        )

    raise ValueError(f"Unknown trainer: {cfg.name}")

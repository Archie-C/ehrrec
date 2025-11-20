import torch

from src.models.GAMENet import GAMENet
from src.models.KNN import KNNModel, KNNConfig


def create_model(cfg_model, *, device=None):
    """
    Factory for constructing models from Hydra/OMEGACONF configs.

    Parameters
    ----------
    cfg_model: DictConfig | Mapping
        Model configuration containing at least a `name` field.
    vocab_sizes: Sequence[int]
        Vocabulary sizes [diag, proc, med] produced by preprocessing.
    ehr_adj, ddi_adj: np.ndarray
        Precomputed adjacency matrices for GAMENet.
    device: torch.device | None
        Target device to place the model on.
    """
    name = cfg_model.name
    if name == "gamenet":
        return GAMENet(
            vocab_path=cfg_model.vocab_path,
            ehr_adj_path=cfg_model.ehr_adj_path,
            ddi_adj_path=cfg_model.ddi_adj_path,
            emb_dim=cfg_model.emb_dim,
            ddi_in_memory=cfg_model.ddi_in_memory,
            dropout=cfg_model.dropout,
            device=device,
        )
    if name == "knn":
        config = KNNConfig(
            k=cfg_model.k,
            threshold=cfg_model.threshold,
            gamma=cfg_model.gamma,
            temperature=cfg_model.temperature,
            min_sim=cfg_model.min_sim,
            mode=cfg_model.mode,
        )
        return KNNModel(config=config, device=device or torch.device("cpu"))
    raise ValueError(f"Unknown model: {name}")

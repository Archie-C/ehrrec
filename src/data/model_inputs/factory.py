from src.data.model_inputs.four_s_drug import FourSDrugModelInputBuilder
from src.data.model_inputs.gamenet import GAMENetModelInputBuilder
from src.data.model_inputs.knn import KNNModelInputBuilder
from src.data.model_inputs.knn_with_pca import KNNModelInputBuilderWithPCA
from src.data.model_inputs.most_popular import MostPopularInputBuilder


def create_model_input_builder(cfg, log_level, run_mode=None):
    name = cfg.name.lower()
    if name == "gamenet":
        return GAMENetModelInputBuilder(log_level=log_level)
    if name == "knn":
        return KNNModelInputBuilder(log_level=log_level)
    if name == "knn_with_pca":
        auto_refit = run_mode in ("train", "full")
        refit_flag = cfg.get("refit", True)
        return KNNModelInputBuilderWithPCA(
            n_components=cfg.get("dim", 100),
            cache_path=cfg.get("cache_file", None),
            refit=auto_refit or refit_flag,
            log_level=log_level,
        )
    if name == "most_popular":
        return MostPopularInputBuilder(log_level=log_level)
    if name == "4sdrug":
        return FourSDrugModelInputBuilder(log_level=log_level, batch_size=cfg.batch_size)
    raise ValueError(f"Unknown model input builder: {cfg.name}")

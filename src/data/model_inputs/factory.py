from src.data.model_inputs.gamenet import GAMENetModelInputBuilder
from src.data.model_inputs.knn import KNNModelInputBuilder
from src.data.model_inputs.most_popular import MostPopularInputBuilder


def create_model_input_builder(cfg):
    name = cfg.name.lower()
    if name == "gamenet":
        return GAMENetModelInputBuilder()
    if name == "knn":
        return KNNModelInputBuilder()
    if name == "most_popular":
        return MostPopularInputBuilder()
    raise ValueError(f"Unknown model input builder: {cfg.name}")

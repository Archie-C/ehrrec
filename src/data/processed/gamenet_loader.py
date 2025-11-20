from pathlib import Path

from src.core.interfaces.loader import Loader
from src.data.context import ProcessedDataContext
from src.data.loaders.pkl_loader import PklLoader
from src.data.processed import register_processed_loader


class GAMENetProcessedLoader(Loader):
    def __init__(self, output_dir: str, filenames: dict, pkl_loader=None):
        self.output_dir = Path(output_dir)
        self.filenames = filenames
        self.pkl_loader = pkl_loader or PklLoader()

    def load(self) -> ProcessedDataContext:
        records_path = self.output_dir / self.filenames["records"]
        vocab_path = self.output_dir / self.filenames["vocab"]
        ehr_path = self.output_dir / self.filenames["ehr_adj"]
        ddi_path = self.output_dir / self.filenames["ddi_adj"]

        records = self.pkl_loader.load(records_path, override_backend="dill")
        vocab = self.pkl_loader.load(vocab_path, override_backend="dill")
        ehr_adj = self.pkl_loader.load(ehr_path, override_backend="dill")
        ddi_adj = self.pkl_loader.load(ddi_path, override_backend="dill")
        adjacency = {"ehr": ehr_adj, "ddi": ddi_adj}
        return ProcessedDataContext(
            name="gamenet",
            records=records,
            vocab=vocab,
            adjacency=adjacency,
            metadata={"output_dir": str(self.output_dir)},
        )


@register_processed_loader("gamenet")
def _build_gamenet_loader(cfg, **_):
    return GAMENetProcessedLoader(cfg.output_dir, cfg.filenames)

import pickle
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import src.load.MIMIC3Loader as m

@pytest.fixture
def cfg() -> m.MIMIC3LoaderConfig:
    return m.MIMIC3LoaderConfig()

@pytest.fixture
def mimic_files(tmp_path: Path, cfg: m.MIMIC3LoaderConfig) -> Path:
    # diagnoses / procedures / meds
    pd.DataFrame({"A": [1]}).to_csv(tmp_path / cfg.diagnoses_file, index=False)
    pd.DataFrame({"B": [2]}).to_csv(tmp_path / cfg.procedures_file, index=False)
    pd.DataFrame({"C": [3]}).to_csv(tmp_path / cfg.medications_file, index=False)

    # ndc2rxcui (TSV)
    (tmp_path / cfg.ndc2rxcui_file).write_text("NDC\tRXCUI\n0002\t123\n", encoding="utf-8")

    # rxcui2atc3 (CSV)
    pd.DataFrame({"RXCUI": ["123"], "ATC3": ["A01"]}).to_csv(tmp_path / cfg.rxcui2atc3_file, index=False)

    # idx2drug (pickle)
    with open(tmp_path / cfg.idx2drug_file, "wb") as f:
        pickle.dump({0: "drug0"}, f)

    # ddi (CSV) via template
    ddi_path = tmp_path / cfg.ddi_file_template.format(variant=cfg.variant)
    pd.DataFrame({"u": ["A"], "v": ["B"]}).to_csv(ddi_path, index=False)

    # cid_atc via template (plain text with csv-ish lines)
    cid_atc_path = tmp_path / cfg.drug_atc_file_template.format(variant=cfg.variant)
    cid_atc_path.write_text("CID1,A01B1,A01C2\nCID2,B02D3\n", encoding="utf-8")

    return tmp_path


@pytest.fixture
def loader(monkeypatch: pytest.MonkeyPatch, cfg: m.MIMIC3LoaderConfig, mimic_files: Path) -> m.MIMIC3Loader:
    # Patch _path(cfg, rel) -> tmp_path / rel
    monkeypatch.setattr(m, "_path", lambda _cfg, rel: mimic_files / rel)
    return m.MIMIC3Loader(cfg)


def test_load_impl_returns_tables_response(loader: m.MIMIC3Loader):
    resp = loader._load_impl()
    assert isinstance(resp, m.TablesResponse)


def test_tables_has_expected_keys(loader: m.MIMIC3Loader):
    resp = loader._load_impl()
    assert set(resp.tables.keys()) == {"diagnoses", "procedures", "medications"}


def test_artefacts_has_expected_keys(loader: m.MIMIC3Loader):
    resp = loader._load_impl()
    assert set(resp.artefacts.keys()) == {"ndc_rxcui", "rxcui_atc3", "idx2drug", "ddi", "cid2atc3"}


def test_meta_dataset_is_mimic_iii(loader: m.MIMIC3Loader):
    resp = loader._load_impl()
    assert resp.meta["dataset"] == "mimic-iii"


def test_meta_version_is_1_4(loader: m.MIMIC3Loader):
    resp = loader._load_impl()
    assert resp.meta["version"] == "1.4"


def test_idx2drug_loaded_from_pickle(loader: m.MIMIC3Loader):
    resp = loader._load_impl()
    assert resp.artefacts["idx2drug"][0] == "drug0"


def test_ddi_loaded_as_dataframe(loader: m.MIMIC3Loader):
    resp = loader._load_impl()
    assert isinstance(resp.artefacts["ddi"], pd.DataFrame)


def test_ndc_rxcui_loaded_as_dataframe(loader: m.MIMIC3Loader):
    resp = loader._load_impl()
    assert isinstance(resp.artefacts["ndc_rxcui"], pd.DataFrame)


def test_read_cid_to_atc3_parses_expected_mapping(loader: m.MIMIC3Loader, mimic_files: Path, cfg: m.MIMIC3LoaderConfig):
    path = mimic_files / cfg.drug_atc_file_template.format(variant=cfg.variant)
    mapping = loader._read_cid_to_atc3(path)
    assert mapping == {"CID1": {"A01B", "A01C"}, "CID2": {"B02D"}}

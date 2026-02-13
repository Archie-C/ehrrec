from __future__ import annotations

import pandas as pd

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any


SCHEMA_VERSION = "0.1.0"

@dataclass(frozen=True)
class RequiredTables:
    """
    What a preprocessor needs from the loader-produced Sample + dataset_artefacts.
    """
    # Required keys inside Sample.inputs / Sample.targets / Sample.meta / Sample.artefacts
    tables: Dict[str, Any] # e.g. {"diagnoses": df, "procedures": df, "prescriptions": df}
    meta: Dict[str, Any] = field(default_factory=dict)
    artefacts: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

@dataclass(frozen=True)
class ProcessedSpec:
    """
    What a preprocessor produces in ProcessedTables.
    """
    tables: Dict[str, Any] # e.g. {"train": pd.Dataframe}
    meta: Dict[str, Any] = field(default_factory=dict)
    artefacts: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

@dataclass(frozen=True)
class ProcessedTables:
    tables: pd.DataFrame
    artefacts: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

class AbstractPreprocessor(ABC):
    """
    A preprocessor interface that consumes:
        - Sample objects produced by a Loader
        - loader.dataset_artefacts()

    Responsibilities:
        - validate it can consume the loader's outputs (keys present)
        - fit() on training data if needed (build vocabs, compute stats)
        - transform samples into ProcessedSample objects
        - collate() ProcessedSamples into a model-ready batch dict
    """

    def __init__(self) -> None:
        self._fitted: bool = False

    @abstractmethod
    def requires(self) -> RequiredTables:
        """Declare required keys in Sample and loader.dataset_artefacts()."""
        raise NotImplementedError

    @abstractmethod
    def produces(self) -> ProcessedSpec:
        """Declare what keys this preprocessor outputs."""
        raise NotImplementedError
    
    @abstractmethod
    def process(self, loaded_data: RequiredTables) -> ProcessedTables:
        """""" 
        return NotImplementedError
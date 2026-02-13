from __future__ import annotations

from typing import Any, Dict
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

SCHEMA_VERSION = "0.1.0"

@dataclass(frozen=True)
class TablesResponse:
    tables: Dict[str, Any] # e.g. {"diagnoses": df, "procedures": df, "prescriptions": df}
    meta: Dict[str, Any] = field(default_factory=dict)
    artefacts: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

class AbstractLoader(ABC):
    def __init__(self) -> None:
        self._loaded = False
        self._resp: TablesResponse | None = None

    def load(self) -> TablesResponse:
        if not self._loaded:
            self._resp = self._load_impl()
            self._loaded = True
        return self._resp  # type: ignore[return-value]

    @abstractmethod
    def _load_impl(self) -> TablesResponse:
        raise NotImplementedError
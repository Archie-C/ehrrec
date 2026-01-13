from __future__ import annotations

from typing import Callable, Dict, TypeVar


T = TypeVar("T")


class Registry:
    """
    Minimal string-to-constructor registry.

    Keeps factory modules tidy and lets components self-register.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._registry: Dict[str, Callable[..., T]] = {}

    def register(self, key: str, builder: Callable[..., T]) -> None:
        normalized = key.lower()
        if normalized in self._registry:
            raise ValueError(f"{self._name} registry: '{key}' already registered")
        self._registry[normalized] = builder

    def build(self, key: str, *args, **kwargs) -> T:
        normalized = key.lower()
        if normalized not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(f"{self._name} registry: '{key}' not found. Available: {available}")
        return self._registry[normalized](*args, **kwargs)

    def keys(self):
        return tuple(sorted(self._registry.keys()))

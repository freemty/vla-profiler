"""
Simple key-value registry with duplicate-key protection.

Used for registering controllers, hook factories, or any named components
that should be unique within a scope.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, Tuple


class Registry:
    """
    A dict wrapper that raises on duplicate registration and provides
    helpful error messages on missing keys.

    Usage::

        reg = Registry("controllers")
        reg.register("flux", FluxController)
        ctrl_cls = reg["flux"]
    """

    def __init__(self, name: str = "registry") -> None:
        self._name = name
        self._store: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def register(self, key: str, value: Any) -> None:
        """
        Register *value* under *key*.

        Raises ``ValueError`` if *key* is already registered.
        """
        if key in self._store:
            raise ValueError(
                f"[{self._name}] Duplicate registration: '{key}' is already "
                f"registered (current value: {self._store[key]!r})"
            )
        self._store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for *key*, or *default* if not found."""
        return self._store.get(key, default)

    # ------------------------------------------------------------------
    # Dict-like access
    # ------------------------------------------------------------------

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def __getitem__(self, key: str) -> Any:
        if key not in self._store:
            available = ", ".join(sorted(self._store.keys())) or "(empty)"
            raise KeyError(
                f"[{self._name}] '{key}' not found. "
                f"Available keys: {available}"
            )
        return self._store[key]

    def keys(self) -> Iterator[str]:
        return iter(self._store.keys())

    def items(self) -> Iterator[Tuple[str, Any]]:
        return iter(self._store.items())

    # ------------------------------------------------------------------
    # Informational
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        return f"Registry(name={self._name!r}, keys={list(self._store.keys())})"

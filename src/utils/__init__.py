"""Utility modules for vlla profiling framework."""

from typing import Any


def layer_sort_key(name: str) -> int:
    """Extract numeric suffix from layer name for sorting (e.g., 'layer_7' -> 7)."""
    parts = name.split("_")
    return int(parts[-1]) if parts[-1].isdigit() else 0


def to_plain(obj: Any) -> Any:
    """Convert OmegaConf DictConfig/ListConfig to plain Python types. Pass-through for others."""
    if hasattr(obj, "_iter_ex"):
        from omegaconf import OmegaConf
        return OmegaConf.to_container(obj, resolve=True)
    return obj

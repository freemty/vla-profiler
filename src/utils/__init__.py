"""Utility modules for vlla profiling framework."""


def layer_sort_key(name: str) -> int:
    """Extract numeric suffix from layer name for sorting (e.g., 'layer_7' -> 7)."""
    parts = name.split("_")
    return int(parts[-1]) if parts[-1].isdigit() else 0

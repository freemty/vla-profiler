"""Shared profiling aggregation helper for standalone scripts.

Produces Hydra-PhaseTimer-compatible JSON with full quantile support.
Used by: profile_fastwam.py, profile_lingbot_va.py
"""

from __future__ import annotations

import statistics
from typing import Dict, List


def compute_phase_stats(measurements: List[float]) -> Dict:
    """Compute full statistics for one phase.

    Returns keys aligned with Hydra PhaseTimer output:
    mean_ms, median_ms, std_ms, min_ms, max_ms, p10_ms, p90_ms, p99_ms,
    cv (coefficient of variation, %), n, all_ms (raw iteration data).
    """
    if not measurements:
        return {"n": 0, "all_ms": []}

    n = len(measurements)
    mean_ms = statistics.mean(measurements)
    std_ms = statistics.stdev(measurements) if n > 1 else 0.0
    median_ms = statistics.median(measurements)
    sorted_ms = sorted(measurements)

    def _percentile(p: float) -> float:
        if n == 1:
            return sorted_ms[0]
        k = (n - 1) * p
        f = int(k)
        c = min(f + 1, n - 1)
        return sorted_ms[f] + (sorted_ms[c] - sorted_ms[f]) * (k - f)

    return {
        "mean_ms": round(mean_ms, 3),
        "median_ms": round(median_ms, 3),
        "std_ms": round(std_ms, 3),
        "min_ms": round(sorted_ms[0], 3),
        "max_ms": round(sorted_ms[-1], 3),
        "p10_ms": round(_percentile(0.10), 3),
        "p90_ms": round(_percentile(0.90), 3),
        "p99_ms": round(_percentile(0.99), 3),
        "cv_pct": round((std_ms / mean_ms * 100) if mean_ms > 0 else 0.0, 2),
        "n": n,
        "all_ms": [round(x, 3) for x in measurements],
    }


def print_phase_summary(phase: str, stats: Dict, label_width: int = 16) -> None:
    """Print a single-line phase summary to stdout."""
    if stats.get("n", 0) == 0:
        return
    print(
        f"  {phase:>{label_width}s}: "
        f"mean={stats['mean_ms']:7.2f}ms "
        f"median={stats['median_ms']:7.2f}ms "
        f"p10/p90={stats['p10_ms']:.2f}/{stats['p90_ms']:.2f} "
        f"cv={stats['cv_pct']:.1f}%"
    )


def stable_window_stats(measurements: List[float], drop_first: int = 0) -> Dict:
    """Return stats computed only on runs [drop_first:].

    Useful to quarantine GPU power-warmup bimodal contamination.
    Pass drop_first > 0 only when you have prior evidence of bimodality.
    """
    if drop_first <= 0 or drop_first >= len(measurements):
        return compute_phase_stats(measurements)
    trimmed = measurements[drop_first:]
    stats = compute_phase_stats(trimmed)
    stats["_drop_first"] = drop_first
    return stats

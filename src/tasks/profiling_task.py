"""
EPD (Encode-Prefill-Decode) profiling task.

Reads timing data from the controller's PhaseTimer and writes
structured timing results to JSON.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

from src.tasks import TASK_REGISTRY


logger = logging.getLogger(__name__)


def task_epd_profiling(
    controller: Any,
    save_dir: str,
    task_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extract encode/prefill/decode timing from controller and save to JSON.

    Checks for:
    1. controller.timer — single-run timing from PhaseTimer
    2. controller._aggregated_timing — multi-run benchmark stats
       (mean/std set by run_tasks.py)

    Outputs:
        {save_dir}/epd_profiling/epd_timing.json
    """
    output_dir = os.path.join(save_dir, "epd_profiling")
    os.makedirs(output_dir, exist_ok=True)

    timing_data: Dict[str, Any] = {}

    # Check for aggregated multi-run timing (set by run_tasks.py)
    aggregated = getattr(controller, "_aggregated_timing", None)
    if aggregated is not None:
        timing_data["aggregated"] = aggregated
        num_runs = aggregated.get("num_runs", 0)
        logger.info("Using aggregated timing from %d runs", num_runs)
        for phase in ("encode", "prefill", "decode"):
            phase_stats = aggregated.get(phase)
            if phase_stats is None:
                continue
            cv = phase_stats.get("cv", 0.0)
            cv_warn = " [UNSTABLE cv>5%%]" if cv > 0.05 else ""
            logger.info(
                "  %s: median=%.2fms mean=%.2fms std=%.2fms "
                "p10=%.2f p90=%.2f cv=%.3f%s",
                phase,
                phase_stats.get("median_ms", 0.0),
                phase_stats.get("mean_ms", 0.0),
                phase_stats.get("std_ms", 0.0),
                phase_stats.get("p10_ms", 0.0),
                phase_stats.get("p90_ms", 0.0),
                cv,
                cv_warn,
            )

    # Single-run timing from PhaseTimer
    timer = getattr(controller, "timer", None)
    if timer is not None:
        summary = timer.summary()
        single_run: Dict[str, Any] = {}

        for phase in ("encode", "prefill", "decode"):
            if phase in summary:
                single_run[phase] = {
                    "elapsed_ms": summary[phase],
                }

        decode_steps = timer.decode_step_count
        if decode_steps > 0 and "decode" in summary:
            single_run["decode"]["step_count"] = decode_steps
            single_run["decode"]["ms_per_token"] = (
                summary["decode"] / decode_steps
            )

        # Total latency
        total_ms = sum(summary.values())
        single_run["total_ms"] = total_ms

        timing_data["single_run"] = single_run

    # Write output
    output_path = os.path.join(output_dir, "epd_timing.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(timing_data, f, indent=2, ensure_ascii=False)

    logger.info("EPD timing saved to %s", output_path)
    return timing_data


TASK_REGISTRY.register("epd_profiling", task_epd_profiling)

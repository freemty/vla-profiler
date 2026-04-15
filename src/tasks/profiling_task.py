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
        # Log all phase stats dynamically (not hardcoded to E/P/D)
        for key, val in aggregated.items():
            if isinstance(val, dict) and "mean_ms" in val:
                cv = val.get("cv", 0.0)
                cv_warn = " [UNSTABLE cv>5%%]" if cv > 0.05 else ""
                logger.info(
                    "  %s: mean=%.2fms std=%.2fms%s",
                    key,
                    val.get("mean_ms", 0.0),
                    val.get("std_ms", 0.0),
                    cv_warn,
                )

    # Single-run timing from PhaseTimer
    timer = getattr(controller, "timer", None)
    if timer is not None:
        summary = timer.summary()
        single_run: Dict[str, Any] = {}

        decode_steps = timer.decode_step_count
        # Dynamically iterate all recorded phases
        for phase, elapsed in summary.items():
            phase_entry: Dict[str, Any] = {"elapsed_ms": elapsed}
            if phase == "decode" and decode_steps > 0:
                phase_entry = {
                    **phase_entry,
                    "step_count": decode_steps,
                    "ms_per_token": elapsed / decode_steps,
                }
            single_run[phase] = phase_entry

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

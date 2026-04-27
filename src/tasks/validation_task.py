"""
Timing cross-validation task.

Runs a single inference pass with both PhaseTimer (CUDA Events) and
torch.profiler (record_function) active simultaneously. Compares the
two independent measurements to verify PhaseTimer trustworthiness.

Also computes the gap between sum(E+P+D) and end-to-end wall clock
to surface any unmeasured time (projection layers, sampling, etc.).
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.tasks import TASK_REGISTRY


logger = logging.getLogger(__name__)

# Verdict thresholds (deviation percentage)
_PASS_THRESHOLD = 5.0
_WARN_THRESHOLD = 15.0


def _verdict(deviation_pct: float) -> str:
    """Return PASS / WARN / FAIL based on deviation percentage."""
    abs_dev = abs(deviation_pct)
    if abs_dev < _PASS_THRESHOLD:
        return "PASS"
    if abs_dev < _WARN_THRESHOLD:
        return "WARN"
    return "FAIL"


def _extract_phase_cuda_time(
    key_averages: Any,
    phase_prefix: str = "PHASE_",
) -> Dict[str, float]:
    """
    Extract cuda_time_total for PHASE_* record_function entries.

    Returns dict mapping phase name -> cuda_time in milliseconds.
    """
    result: Dict[str, float] = {}
    for evt in key_averages:
        if evt.key.startswith(phase_prefix):
            phase_name = evt.key[len(phase_prefix):]
            # device_time_total (PyTorch ≥2.9) or cuda_time_total (older), in μs
            us = getattr(evt, "device_time_total", None) or getattr(evt, "cuda_time_total", 0)
            cuda_ms = us / 1000.0
            # Accumulate if same phase appears multiple times (decode)
            existing = result.get(phase_name, 0.0)
            result[phase_name] = existing + cuda_ms
    return result


class _ProfilerPhaseTracker:
    """
    Manages record_function contexts for phase tracking via torch.profiler.

    Registered as module hooks on the same boundaries as PhaseTimer,
    so the two measurements cover identical code regions.
    """

    def __init__(self) -> None:
        self._active_rf: Optional[Any] = None
        self._hooks: List[torch.utils.hooks.RemovableHook] = []

    def register_hooks(
        self,
        vision_encoder: nn.Module,
        layer_blocks: List[nn.Module],
    ) -> None:
        """Register record_function hooks on vision encoder + LLM layers."""
        first_layer = layer_blocks[0]
        last_layer = layer_blocks[-1]
        tracker = self

        # --- Vision encoder ---
        def _encode_pre(module: nn.Module, inputs: Any) -> None:
            rf = torch.autograd.profiler.record_function("PHASE_encode")
            rf.__enter__()
            tracker._active_rf = rf

        def _encode_post(module: nn.Module, inputs: Any, output: Any) -> None:
            if tracker._active_rf is not None:
                tracker._active_rf.__exit__(None, None, None)
                tracker._active_rf = None

        self._hooks = [
            *self._hooks,
            vision_encoder.register_forward_pre_hook(_encode_pre),
            vision_encoder.register_forward_hook(_encode_post),
        ]

        # --- LLM first/last layer ---
        def _llm_first_pre(module: nn.Module, inputs: Any) -> None:
            hidden = inputs[0] if isinstance(inputs, tuple) else inputs
            seq_len = hidden.shape[1] if hidden.dim() >= 2 else 1
            phase = "prefill" if seq_len > 1 else "decode"
            rf = torch.autograd.profiler.record_function(f"PHASE_{phase}")
            rf.__enter__()
            tracker._active_rf = rf

        def _llm_last_post(module: nn.Module, inputs: Any, output: Any) -> None:
            if tracker._active_rf is not None:
                tracker._active_rf.__exit__(None, None, None)
                tracker._active_rf = None

        self._hooks = [
            *self._hooks,
            first_layer.register_forward_pre_hook(_llm_first_pre),
            last_layer.register_forward_hook(_llm_last_post),
        ]

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []


def run_validation_inference(
    controller: Any,
    cfg: Any,
    inp: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run a single inference with both PhaseTimer and torch.profiler active.

    Returns a validation report dict with phase comparisons and gap analysis.
    """
    # Reset PhaseTimer for a clean measurement
    controller.timer.reset()

    # Set up profiler phase tracker on same boundaries
    tracker = _ProfilerPhaseTracker()
    tracker.register_hooks(
        vision_encoder=controller.get_vision_encoder(),
        layer_blocks=controller.get_layer_blocks(),
    )

    # Run inference under torch.profiler
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        wall_start = time.perf_counter()
        torch.cuda.synchronize()
        controller.model_inference(controller.pipeline, cfg, inp)
        torch.cuda.synchronize()
        wall_end = time.perf_counter()

    tracker.remove_hooks()

    wall_clock_ms = (wall_end - wall_start) * 1000.0

    # Extract PhaseTimer measurements
    phase_timer_summary = controller.timer.summary()

    # Extract torch.profiler measurements
    key_averages = prof.key_averages()
    profiler_phases = _extract_phase_cuda_time(key_averages)

    # Build comparison
    phases_report: Dict[str, Any] = {}
    all_phases = sorted(set(phase_timer_summary.keys()) | set(profiler_phases.keys()))

    for phase in all_phases:
        pt_ms = phase_timer_summary.get(phase, 0.0)
        pr_ms = profiler_phases.get(phase, 0.0)
        deviation_pct = ((pt_ms - pr_ms) / pr_ms * 100.0) if pr_ms > 0 else 0.0

        phases_report[phase] = {
            "phase_timer_ms": round(pt_ms, 2),
            "profiler_cuda_ms": round(pr_ms, 2),
            "deviation_pct": round(deviation_pct, 1),
            "verdict": _verdict(deviation_pct),
        }

    # Gap analysis: wall clock vs sum of phases
    sum_phases_ms = sum(phase_timer_summary.values())
    gap_ms = wall_clock_ms - sum_phases_ms
    gap_pct = (gap_ms / wall_clock_ms * 100.0) if wall_clock_ms > 0 else 0.0

    gap_report = {
        "wall_clock_ms": round(wall_clock_ms, 2),
        "sum_phases_ms": round(sum_phases_ms, 2),
        "gap_ms": round(gap_ms, 2),
        "gap_pct": round(gap_pct, 1),
    }

    # Overall verdict: worst of all phase verdicts
    verdicts = [p["verdict"] for p in phases_report.values()]
    if "FAIL" in verdicts:
        overall = "FAIL"
    elif "WARN" in verdicts:
        overall = "WARN"
    else:
        overall = "PASS"

    return {
        "input_name": inp.get("name", "unnamed"),
        "phases": phases_report,
        "gap": gap_report,
        "overall_verdict": overall,
    }


def task_timing_validation(
    controller: Any,
    save_dir: str,
    task_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Read validation results from controller and write to JSON.

    The actual validation run is triggered by run_tasks.py, which stores
    the result on controller._validation_report.
    """
    output_dir = os.path.join(save_dir, "timing_validation")
    os.makedirs(output_dir, exist_ok=True)

    report = getattr(controller, "_validation_report", None)
    if report is None:
        logger.warning("No validation report found — skipping")
        return {}

    # Log summary
    overall = report.get("overall_verdict", "UNKNOWN")
    logger.info("Timing validation: %s", overall)
    for phase, data in report.get("phases", {}).items():
        logger.info(
            "  %s: PhaseTimer=%.1fms  Profiler=%.1fms  dev=%.1f%%  %s",
            phase,
            data["phase_timer_ms"],
            data["profiler_cuda_ms"],
            data["deviation_pct"],
            data["verdict"],
        )
    gap = report.get("gap", {})
    logger.info(
        "  gap: %.1fms (%.1f%% of wall clock %.1fms)",
        gap.get("gap_ms", 0),
        gap.get("gap_pct", 0),
        gap.get("wall_clock_ms", 0),
    )

    # Write output
    output_path = os.path.join(output_dir, "validation_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("Validation report saved to %s", output_path)
    return report


TASK_REGISTRY.register("timing_validation", task_timing_validation)

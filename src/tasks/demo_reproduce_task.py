"""
Demo reproduction task — verify model inference matches official demo behavior.

For each model type, checks:
- VLM (Qwen/OpenVLA-AR): non-empty text output, token count, format
- VLA (ACT/LingBot/Pi-Zero): action shape, denoise steps, value range
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

import torch

from src.tasks import TASK_REGISTRY


logger = logging.getLogger(__name__)


def _check_vlm_text(result: Any, expected: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Validate VLM text generation output."""
    checks = []

    if isinstance(result, list):
        output_text = result
    elif isinstance(result, dict):
        output_text = result.get("output_text", result.get("decoded_text", []))
    else:
        output_text = []

    non_empty = len(output_text) > 0 and all(len(t.strip()) > 0 for t in output_text)
    checks.append({
        "name": "output_non_empty",
        "passed": non_empty,
        "actual": output_text[:1] if output_text else [],
        "expected": "non-empty text list",
    })

    min_tokens = expected.get("min_output_tokens", 5)
    if output_text:
        word_count = len(output_text[0].split())
        checks.append({
            "name": "min_output_length",
            "passed": word_count >= min_tokens,
            "actual": word_count,
            "expected": f">= {min_tokens} words",
        })

    keywords = expected.get("keywords", [])
    if keywords and output_text:
        text_lower = output_text[0].lower()
        found = [kw for kw in keywords if kw.lower() in text_lower]
        checks.append({
            "name": "keyword_presence",
            "passed": len(found) > 0,
            "actual": found,
            "expected": f"at least 1 of {keywords}",
        })

    return checks


def _check_openvla_action(result: Dict[str, Any], expected: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Validate OpenVLA discrete action token output."""
    checks = []

    action_ids = result.get("action_token_ids", [])
    expected_len = expected.get("action_token_count", 7)
    checks.append({
        "name": "action_token_count",
        "passed": len(action_ids) == expected_len,
        "actual": len(action_ids),
        "expected": expected_len,
    })

    if action_ids:
        in_range = all(0 <= t <= 255 for t in action_ids)
        checks.append({
            "name": "action_token_range",
            "passed": in_range,
            "actual": f"min={min(action_ids)}, max={max(action_ids)}",
            "expected": "[0, 255]",
        })

    return checks


def _check_vla_action(result: Dict[str, Any], expected: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Validate VLA continuous action output (ACT/LingBot/Pi-Zero)."""
    checks = []

    action_shape = result.get("action_shape", [])
    if isinstance(action_shape, str):
        checks.append({
            "name": "action_shape_valid",
            "passed": False,
            "actual": action_shape,
            "expected": "tensor shape list",
        })
        return checks

    expected_shape = expected.get("action_shape")
    if expected_shape:
        shape_match = list(action_shape) == list(expected_shape)
        checks.append({
            "name": "action_shape",
            "passed": shape_match,
            "actual": action_shape,
            "expected": expected_shape,
        })

    expected_ndim = expected.get("action_ndim")
    if expected_ndim:
        checks.append({
            "name": "action_ndim",
            "passed": len(action_shape) == expected_ndim,
            "actual": len(action_shape),
            "expected": expected_ndim,
        })

    expected_denoise = expected.get("denoise_steps")
    actual_denoise = result.get("denoise_steps", result.get("num_denoise_steps"))
    if expected_denoise and actual_denoise is not None:
        checks.append({
            "name": "denoise_steps",
            "passed": actual_denoise == expected_denoise,
            "actual": actual_denoise,
            "expected": expected_denoise,
        })

    actions = result.get("actions")
    if actions is not None and torch.is_tensor(actions):
        has_nan = torch.isnan(actions).any().item()
        has_inf = torch.isinf(actions).any().item()
        checks.append({
            "name": "action_no_nan_inf",
            "passed": not has_nan and not has_inf,
            "actual": f"nan={has_nan}, inf={has_inf}",
            "expected": "no NaN or Inf",
        })

        clip_value = expected.get("action_clip_value")
        if clip_value:
            in_range = (actions.abs() <= clip_value + 1e-3).all().item()
            checks.append({
                "name": "action_value_range",
                "passed": in_range,
                "actual": f"max_abs={actions.abs().max().item():.4f}",
                "expected": f"<= {clip_value}",
            })

    return checks


def task_demo_reproduce(
    controller: Any,
    save_dir: str,
    task_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Verify model inference output matches expected demo behavior.

    Reads controller._last_result (set by run_tasks.py demo mode)
    and runs model-specific validation checks.

    task_config keys:
        model_type: "vlm" | "openvla" | "vla"
        expected: dict with model-specific expected values
    """
    output_dir = os.path.join(save_dir, "demo_reproduce")
    os.makedirs(output_dir, exist_ok=True)

    result = getattr(controller, "_last_result", None)
    if result is None:
        report = {
            "status": "ERROR",
            "message": "No inference result found (controller._last_result is None)",
            "checks": [],
        }
        _save_report(output_dir, report)
        return report

    model_type = task_config.get("model_type", "vla")
    expected = task_config.get("expected", {})

    if model_type == "vlm":
        checks = _check_vlm_text(result, expected)
    elif model_type == "openvla":
        checks = _check_openvla_action(result, expected)
    else:
        checks = _check_vla_action(result, expected)

    passed_all = all(c["passed"] for c in checks)
    report = {
        "status": "PASS" if passed_all else "FAIL",
        "model_type": model_type,
        "num_checks": len(checks),
        "num_passed": sum(1 for c in checks if c["passed"]),
        "checks": checks,
    }

    _save_report(output_dir, report)

    status_icon = "PASS" if passed_all else "FAIL"
    logger.info(
        "Demo reproduce: %s (%d/%d checks passed)",
        status_icon, report["num_passed"], report["num_checks"],
    )
    for c in checks:
        icon = "ok" if c["passed"] else "FAIL"
        logger.info("  [%s] %s: actual=%s", icon, c["name"], c["actual"])

    return report


def _save_report(output_dir: str, report: Dict[str, Any]) -> None:
    save_path = os.path.join(output_dir, "demo_report.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("Demo report saved to %s", save_path)


TASK_REGISTRY.register("demo_reproduce", task_demo_reproduce)

"""
Smoke tests for framework wiring — no GPU or model weights required.
"""

import sys
import os

# Ensure src/core/probe_core is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "core"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_controller_registry_has_qwen():
    from src.controllers import CONTROLLER_REGISTRY
    assert "qwen_vl" in CONTROLLER_REGISTRY


def test_task_registry_has_all_tasks():
    from src.tasks import TASK_REGISTRY
    expected = ["epd_profiling", "visual_text_attention", "sink_detection", "per_layer_stats", "attention_overlay"]
    for task_name in expected:
        assert task_name in TASK_REGISTRY, f"Missing task: {task_name}"


def test_hook_mode_profiling_exists():
    from src.core.probe_core import HookMode
    assert HookMode.PROFILING.value == "profiling"


def test_base_controller_has_phase_support():
    import inspect
    from src.core.probe_core import BaseController
    sig = inspect.signature(BaseController.__init__)
    assert "store_phases" in sig.parameters


def test_registry_unknown_key_raises():
    from src.core.probe_core import Registry
    r = Registry("test")
    try:
        r["nonexistent"]
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


def test_registry_duplicate_raises():
    from src.core.probe_core import Registry
    r = Registry("test")
    r.register("a", 1)
    try:
        r.register("a", 2)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_qwen_controller_has_interpretability_mixin():
    from src.controllers.qwen_vl_controller import QwenVLController
    from src.interpretability.base_mixin import BaseInterpretabilityMixin

    assert issubclass(QwenVLController, BaseInterpretabilityMixin)

# VLM Profiling & Attention Analysis Framework — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable VLM inference profiling + attention analysis framework, starting with Qwen2.5-VL 7B, with a shared core extracted from rope2sink.

**Architecture:** Three-layer inheritance (probe_core.BaseController → BaseVLMController → QwenVLController). Two independent modes: profiling (CUDA event timing for E/P/D) and analysis (QKV tensor collection for attention pattern study). Shared core consumed as git submodule.

**Tech Stack:** Python 3.10+, PyTorch (CUDA events, hooks API), transformers (Qwen2.5-VL), Hydra/OmegaConf (config), einops, easydict

**Spec:** `docs/superpowers/specs/2026-04-14-vlm-profiling-framework-design.md`

**Reference codebase:** `/Users/sum_young/code/projects/rope2sink/` — BaseController at `src/controllers/base_controller.py`, QwenVLController at `src/controllers/qwenvl_controller.py`, run_tasks at `src/run_tasks.py`

---

## File Map

### New repo: `model-probe-core` (created at `/Users/sum_young/code/projects/model-probe-core/`)

| File | Responsibility |
|------|---------------|
| `probe_core/__init__.py` | Public API exports |
| `probe_core/hooks.py` | HookManager class: path resolution, hook registration helpers |
| `probe_core/state.py` | intervene_internal(), StoreManager mixin for step_store/global_store |
| `probe_core/controller.py` | HookMode enum, BaseController (imports from hooks.py and state.py) |
| `probe_core/registry.py` | Registry class for controllers and tasks |

### In vlla project: `/Users/sum_young/code/projects/vlla/`

| File | Responsibility |
|------|---------------|
| `src/core/` | Git submodule → model-probe-core |
| `src/utils/__init__.py` | Package init |
| `src/utils/timing.py` | PhaseTimer: CUDA event wrapper for E/P/D timing |
| `src/controllers/__init__.py` | CONTROLLER_REGISTRY |
| `src/controllers/base_vlm_controller.py` | BaseVLMController: VLM-specific hook registration, phase management |
| `src/controllers/qwen_vl_controller.py` | QwenVLController: Qwen2.5-VL model loading, inference, input prep |
| `src/tasks/__init__.py` | TASK_REGISTRY |
| `src/tasks/profiling_task.py` | task_epd_profiling: timing aggregation and output |
| `src/tasks/attention_task.py` | task_visual_text_attn, task_sink_detection, task_per_layer_stats |
| `src/run_tasks.py` | Hydra entry point |
| `configs/base.yaml` | Shared defaults |
| `configs/qwen_vl_7b/profiling.yaml` | Profiling experiment config |
| `configs/qwen_vl_7b/attention.yaml` | Attention analysis experiment config |
| `tests/test_timing.py` | PhaseTimer unit tests |
| `tests/test_registry.py` | Registry unit tests |
| `tests/test_controller_init.py` | Controller initialization smoke test |

---

## Task 1: Create model-probe-core repo and extract hooks.py

**Files:**
- Create: `/Users/sum_young/code/projects/model-probe-core/probe_core/__init__.py`
- Create: `/Users/sum_young/code/projects/model-probe-core/probe_core/hooks.py`

- [ ] **Step 1: Create the repo and directory structure**

```bash
cd /Users/sum_young/code/projects
mkdir -p model-probe-core/probe_core
cd model-probe-core
git init
```

- [ ] **Step 2: Write `probe_core/hooks.py`**

Extract HookManager from rope2sink's `base_controller.py:124-305`. This is the path resolution and hook registration utility class. Key methods to extract:
- `resolve_path(block, path)` — nested module access via dot path
- `resolve_absolute(model, abs_path)` — absolute path from model root
- `register_variable_action_abs()` — register pre/post hooks with apply functions
- `register_log_action_abs()` — register logging hooks that capture tensors
- `register_extractor_log_action_abs()` — register hooks with custom extractor functions
- `patch_method_action_abs()` — monkey-patch a module method
- `make_pre_apply_hook()` / `make_post_apply_hook()` — hook factory functions

```python
"""
Model-agnostic hook management utilities.
Extracted from rope2sink BaseController.HookManager.
"""

import logging
from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class HookManager:
    """Utilities for registering and managing PyTorch forward hooks on arbitrary modules."""

    @staticmethod
    def make_pre_apply_hook(apply_fn: Callable[[torch.Tensor], torch.Tensor]):
        def pre_hook(module, inputs):
            if not inputs:
                return inputs
            return apply_fn(inputs)
        return pre_hook

    @staticmethod
    def make_post_apply_hook(apply_fn: Callable[[torch.Tensor], torch.Tensor]):
        def post_hook(module, inputs, output):
            return apply_fn(output)
        return post_hook

    @classmethod
    def resolve_path(cls, block: nn.Module, path: str) -> nn.Module:
        """Resolve nested attribute/index path like 'ffn.net[0].proj'."""
        if path == "":
            return block
        current = block
        for segment in path.split('.'):
            if not segment:
                continue
            if '[' in segment and segment.endswith(']'):
                attr = segment[: segment.index('[')]
                idx_str = segment[segment.index('[') + 1 : -1]
                current = getattr(current, attr)
                current = current[int(idx_str)]
            else:
                current = getattr(current, segment)
        return current

    @classmethod
    def resolve_absolute(cls, model: nn.Module, abs_path: str) -> nn.Module:
        """Resolve an absolute path starting from model root."""
        current: Any = model
        for segment in abs_path.split('.'):
            if not segment:
                continue
            if '[' in segment and segment.endswith(']'):
                attr = segment[: segment.index('[')]
                idx_str = segment[segment.index('[') + 1 : -1]
                current = getattr(current, attr)
                current = current[int(idx_str)]
            else:
                current = getattr(current, segment)
        return current

    @classmethod
    def register_variable_action_abs(
        cls,
        controller: Any,
        model: nn.Module,
        *,
        abs_path: str,
        point: str,
        apply_stage: str,
        func: Callable[[torch.Tensor], torch.Tensor],
        collection: str = "intervention_hooks",
        name_suffix: str = "",
    ):
        module = cls.resolve_absolute(model, abs_path)
        key = f"{abs_path}:{point}{(':' + name_suffix) if name_suffix else ''}"
        if apply_stage == "pre" or point == "input":
            hook = module.register_forward_pre_hook(cls.make_pre_apply_hook(func))
        else:
            hook = module.register_forward_hook(cls.make_post_apply_hook(func))
        getattr(controller, collection)[key] = hook
        return hook

    @classmethod
    def register_log_action_abs(
        cls,
        controller: Any,
        model: nn.Module,
        *,
        abs_path: str,
        point: str,
        key: str,
        index: Optional[int] = None,
        log_func: Optional[Callable] = None,
        collection: str = "analysis_hooks",
        name_suffix: str = "",
    ):
        store_key = key

        def log_identity(t: torch.Tensor) -> torch.Tensor:
            if controller.should_store(store_key):
                if isinstance(t, tuple):
                    t_ = t[index]
                else:
                    t_ = t
                if t_ is None:
                    return t
                out = t_.detach().to("cpu", non_blocking=True)
                if store_key not in controller.step_store:
                    controller.step_store[store_key] = [out]
                else:
                    controller.step_store[store_key].append(out)
            return t

        if log_func is None:
            log_func = log_identity
        apply_stage = "pre" if point == "input" else "post"
        return cls.register_variable_action_abs(
            controller=controller,
            model=model,
            abs_path=abs_path,
            point=point,
            apply_stage=apply_stage,
            func=log_identity,
            collection=collection,
            name_suffix=name_suffix or f"log_{key}",
        )

    @classmethod
    def register_extractor_log_action_abs(
        cls,
        controller: Any,
        model: nn.Module,
        *,
        abs_path: str,
        key: str,
        extractor: Callable[[nn.Module, Tuple[Any, ...], Any], torch.Tensor],
        collection: str = "analysis_hooks",
        name_suffix: str = "",
    ):
        module = cls.resolve_absolute(model, abs_path)
        store_key = key

        def hook_fn(mod, inputs, output):
            try:
                if controller.should_store(store_key):
                    data = extractor(mod, inputs, output)
                    out = data.detach().to("cpu", non_blocking=True)
                    if store_key not in controller.step_store:
                        controller.step_store[store_key] = [out]
                    else:
                        controller.step_store[store_key].append(out)
            except Exception:
                pass
            return output

        hook = module.register_forward_hook(hook_fn)
        key_name = name_suffix or f"log_extractor_{key}"
        getattr(controller, collection)[f"{abs_path}:extractor:{key_name}"] = hook
        return hook

    @classmethod
    def patch_method(
        cls,
        controller: Any,
        model: nn.Module,
        *,
        abs_path: str,
        method_name: str,
        wrapper_builder: Callable[[Callable[..., Any]], Callable[..., Any]],
        collection_originals: str = "intervention_original_processors",
    ):
        module = cls.resolve_absolute(model, abs_path)
        original = getattr(module, method_name)
        wrapped = wrapper_builder(original)
        setattr(module, method_name, wrapped)
        getattr(controller, collection_originals)[module] = original
        return original
```

- [ ] **Step 3: Write `probe_core/__init__.py`**

```python
from probe_core.hooks import HookManager

__all__ = ["HookManager"]
```

- [ ] **Step 4: Commit**

```bash
cd /Users/sum_young/code/projects/model-probe-core
git add -A
git commit -m "feat: extract HookManager from rope2sink"
```

---

## Task 2: Create state.py and registry.py in model-probe-core

**Files:**
- Create: `/Users/sum_young/code/projects/model-probe-core/probe_core/state.py`
- Create: `/Users/sum_young/code/projects/model-probe-core/probe_core/registry.py`

- [ ] **Step 1: Write `probe_core/state.py`**

Extract `intervene_internal()` and the store management mixin from rope2sink's `base_controller.py:27-116` and `546-627`.

```python
"""
Tensor state management: step_store / global_store lifecycle and intervene_internal.
"""

from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn


def intervene_internal(
    module: nn.Module,
    input: Any,
    output: torch.Tensor,
    *,
    action: str,
    controller: Optional[Any] = None,
    layer_idx: Optional[int] = None,
    func: Optional[Callable] = None,
    new_value: Optional[torch.Tensor] = None,
    key: Optional[str] = None,
) -> torch.Tensor:
    """
    Generic intervention on internal activations (forward hook compatible).

    Supports actions:
    - "log": record output into controller.step_store
    - "replace": replace output with new_value
    - "apply": apply func to output
    """
    if not torch.is_tensor(output):
        return output

    if action == "log":
        if controller is not None and layer_idx is not None and key is not None:
            store_key = f"{layer_idx}_{key}"
            log_output = output.detach().to("cpu", non_blocking=True)
            if store_key not in controller.step_store:
                controller.step_store[store_key] = [log_output]
            else:
                controller.step_store[store_key].append(log_output)
        return output

    if action == "replace":
        if new_value is None:
            raise ValueError("new_value is required for action 'replace'")
        output[0] = new_value.to(device=output.device, dtype=output.dtype)
        return output

    if action == "apply":
        if func is None or not callable(func):
            raise ValueError("func callable is required for action 'apply'")
        output[0] = func(output[0])
        return output

    raise ValueError(f"Unknown action: {action}")


class StoreMixin:
    """Mixin providing step_store -> global_store tensor lifecycle management."""

    def reset_state(self):
        self.global_store: Dict[str, List[torch.Tensor]] = {}
        self.step_store: Dict[str, List[torch.Tensor]] = {}
        self.cur_phase: str = ""

    def flush_step_store(self):
        """Merge step_store into global_store, then clear step_store."""
        for key in self.step_store:
            if key not in self.global_store:
                self.global_store[key] = self.step_store[key]
            else:
                self.global_store[key] += self.step_store[key]
        self.step_store = {}

    def process_log_dict(self, log_dict: Dict[str, torch.Tensor], layer_idx: Optional[int] = None):
        for key, value in log_dict.items():
            store_key = f"{layer_idx}_{key}" if layer_idx is not None else key
            log_value = value.detach().to("cpu", non_blocking=True)
            if store_key not in self.step_store:
                self.step_store[store_key] = [log_value]
            else:
                self.step_store[store_key].append(log_value)
```

- [ ] **Step 2: Write `probe_core/registry.py`**

```python
"""
Generic registry pattern for controllers and tasks.
"""

from typing import Any, Callable, Dict


class Registry:
    """Dict-based registry with name-based lookup and optional validation."""

    def __init__(self, name: str):
        self.name = name
        self._entries: Dict[str, Any] = {}

    def register(self, key: str, value: Any):
        if key in self._entries:
            raise ValueError(f"{self.name} registry: '{key}' already registered")
        self._entries[key] = value
        return value

    def get(self, key: str) -> Any:
        if key not in self._entries:
            available = ", ".join(self._entries.keys())
            raise KeyError(f"{self.name} registry: '{key}' not found. Available: {available}")
        return self._entries[key]

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def keys(self):
        return self._entries.keys()

    def items(self):
        return self._entries.items()
```

- [ ] **Step 3: Update `probe_core/__init__.py`**

```python
from probe_core.hooks import HookManager
from probe_core.state import StoreMixin, intervene_internal
from probe_core.registry import Registry

__all__ = ["HookManager", "StoreMixin", "intervene_internal", "Registry"]
```

- [ ] **Step 4: Commit**

```bash
cd /Users/sum_young/code/projects/model-probe-core
git add -A
git commit -m "feat: add StoreMixin, intervene_internal, Registry"
```

---

## Task 3: Create BaseController in model-probe-core

**Files:**
- Create: `/Users/sum_young/code/projects/model-probe-core/probe_core/controller.py`
- Modify: `/Users/sum_young/code/projects/model-probe-core/probe_core/__init__.py`

- [ ] **Step 1: Write `probe_core/controller.py`**

The model-agnostic base controller combining HookManager + StoreMixin. Adapted from rope2sink `base_controller.py:119-712`, removing all DiT/diffusers-specific logic.

```python
"""
Model-agnostic base controller: hook lifecycle, state management, phase filtering.
"""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from probe_core.hooks import HookManager
from probe_core.state import StoreMixin


class HookMode(Enum):
    ANALYSIS = "analysis"
    INTERVENE = "intervene"
    PROFILING = "profiling"
    BOTH = "both"
    NONE = "none"


class BaseController(StoreMixin, ABC):
    """
    Abstract base controller for collecting and analyzing model intermediate results.
    Subclasses implement model-specific hook registration and inference.
    """

    HookManager = HookManager

    def __init__(
        self,
        model_name: str,
        pipeline: Any = None,
        logger: Optional[logging.Logger] = None,
        store_type: Optional[Union[str, List[str]]] = None,
        store_layers: Optional[Union[int, List[int]]] = None,
        store_phases: Optional[List[str]] = None,
        hook_mode: Optional[str] = "analysis",
    ):
        self.model_name = model_name
        if logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)
            logging.basicConfig(level=logging.INFO)
        else:
            self.logger = logger

        self.pipeline = pipeline

        # Parse hook mode
        mode_str = str(hook_mode).lower() if hook_mode else "analysis"
        mode_map = {
            "analysis": HookMode.ANALYSIS,
            "intervene": HookMode.INTERVENE,
            "profiling": HookMode.PROFILING,
            "both": HookMode.BOTH,
            "none": HookMode.NONE,
        }
        if mode_str not in mode_map:
            raise ValueError(f"Unknown hook_mode: {hook_mode}")
        self.hook_mode = mode_map[mode_str]

        # Store type filtering
        self.store_type: List[str] = []
        if store_type:
            self.store_type = [store_type] if isinstance(store_type, str) else [str(x) for x in store_type]

        # Layer filtering
        self.store_layers: List[int] = store_layers if store_layers is not None else [-1]
        if isinstance(self.store_layers, int):
            self.store_layers = [self.store_layers]

        # Phase filtering (replaces store_time_ids)
        self.store_phases: List[str] = store_phases or []

        # Initialize state
        self.reset_state()
        self.reset_hooks()

    @property
    def analysis_mode(self) -> bool:
        return self.hook_mode in (HookMode.ANALYSIS, HookMode.BOTH)

    @property
    def intervention_mode(self) -> bool:
        return self.hook_mode in (HookMode.INTERVENE, HookMode.BOTH)

    @property
    def profiling_mode(self) -> bool:
        return self.hook_mode == HookMode.PROFILING

    def should_store(self, store_key: str) -> bool:
        """Check whether to store data for a given key based on layer and phase filters."""
        # Layer filter
        parts = store_key.split("_", 1)
        if len(parts) >= 2 and parts[0].isdigit():
            layer_idx = int(parts[0])
            if -1 not in self.store_layers and layer_idx not in self.store_layers:
                return False

        # Phase filter
        if self.store_phases and self.cur_phase and self.cur_phase not in self.store_phases:
            return False

        return True

    def reset_hooks(self):
        self.analysis_hooks: Dict[str, Any] = {}
        self.intervention_hooks: Dict[str, Any] = {}
        self.analysis_original_processors: Dict[Any, Any] = {}
        self.analysis_original_forwards: Dict[Any, Any] = {}
        self.intervention_original_processors: Dict[Any, Any] = {}
        self.intervention_original_forwards: Dict[Any, Any] = {}

    def register_hooks(self):
        """Register hooks based on current mode. Subclasses implement the specific methods."""
        self.remove_hooks()
        self.logger.info(f"Registering hooks in {self.hook_mode.value} mode")
        if self.profiling_mode and hasattr(self, "register_profiling_hooks"):
            self.register_profiling_hooks()
        if self.intervention_mode and hasattr(self, "register_intervention_hooks"):
            self.register_intervention_hooks()
        if self.analysis_mode and hasattr(self, "register_analysis_hooks"):
            self.register_analysis_hooks()

    def remove_hooks(self):
        for hook in self.analysis_hooks.values():
            hook.remove()
        for hook in self.intervention_hooks.values():
            hook.remove()
        self.analysis_hooks = {}
        self.intervention_hooks = {}
        # Restore patched processors/forwards
        for module, original in self.analysis_original_processors.items():
            module.processor = original
        self.analysis_original_processors = {}
        for module, original in self.analysis_original_forwards.items():
            module.forward = original
        self.analysis_original_forwards = {}
        for module, original in self.intervention_original_processors.items():
            module.processor = original
        self.intervention_original_processors = {}
        for module, original in self.intervention_original_forwards.items():
            module.forward = original
        self.intervention_original_forwards = {}

    @abstractmethod
    def prepare_inputs(self, cfg) -> list:
        ...

    @abstractmethod
    def model_inference(self, pipeline, cfg, inputs) -> Any:
        ...

    def save_results(self, inputs, results, cfg) -> str:
        """Override in subclass to persist outputs. Returns save directory."""
        raise NotImplementedError

    def postprocess(self, **kwargs):
        """Override in subclass for model-specific tensor postprocessing."""
        pass
```

- [ ] **Step 2: Update `probe_core/__init__.py`**

```python
from probe_core.hooks import HookManager
from probe_core.state import StoreMixin, intervene_internal
from probe_core.registry import Registry
from probe_core.controller import BaseController, HookMode

__all__ = [
    "HookManager",
    "StoreMixin",
    "intervene_internal",
    "Registry",
    "BaseController",
    "HookMode",
]
```

- [ ] **Step 3: Commit**

```bash
cd /Users/sum_young/code/projects/model-probe-core
git add -A
git commit -m "feat: add BaseController with phase-based filtering"
```

---

## Task 4: Add model-probe-core as submodule to vlla and scaffold project

**Files:**
- Create: `/Users/sum_young/code/projects/vlla/src/__init__.py`
- Create: `/Users/sum_young/code/projects/vlla/src/utils/__init__.py`
- Create: `/Users/sum_young/code/projects/vlla/src/controllers/__init__.py`
- Create: `/Users/sum_young/code/projects/vlla/src/tasks/__init__.py`

- [ ] **Step 1: Initialize vlla as a git repo if needed and add submodule**

```bash
cd /Users/sum_young/code/projects/vlla
git init  # skip if already a repo
git submodule add ../model-probe-core src/core
```

- [ ] **Step 2: Create package scaffolding**

```python
# src/__init__.py
```

```python
# src/utils/__init__.py
```

```python
# src/controllers/__init__.py
from src.core.probe_core import Registry

CONTROLLER_REGISTRY = Registry("controller")
```

```python
# src/tasks/__init__.py
from src.core.probe_core import Registry

TASK_REGISTRY = Registry("task")
```

- [ ] **Step 3: Commit**

```bash
cd /Users/sum_young/code/projects/vlla
git add -A
git commit -m "feat: scaffold vlla src with model-probe-core submodule"
```

---

## Task 5: Implement PhaseTimer

**Files:**
- Create: `/Users/sum_young/code/projects/vlla/src/utils/timing.py`
- Create: `/Users/sum_young/code/projects/vlla/tests/test_timing.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_timing.py
import pytest
import torch
from src.utils.timing import PhaseTimer


class TestPhaseTimer:
    def test_mark_and_elapsed_cpu(self):
        """PhaseTimer records elapsed time for a phase (CPU fallback)."""
        timer = PhaseTimer()
        timer.mark_start("test_phase")
        # Do trivial work
        _ = sum(range(10000))
        timer.mark_end("test_phase")
        elapsed = timer.elapsed_ms("test_phase")
        assert elapsed >= 0.0

    def test_multiple_phases(self):
        """PhaseTimer tracks multiple independent phases."""
        timer = PhaseTimer()
        timer.mark_start("a")
        timer.mark_end("a")
        timer.mark_start("b")
        timer.mark_end("b")
        assert timer.elapsed_ms("a") >= 0.0
        assert timer.elapsed_ms("b") >= 0.0

    def test_accumulate_decode_steps(self):
        """PhaseTimer accumulates multiple decode steps."""
        timer = PhaseTimer()
        for _ in range(5):
            timer.mark_start("decode")
            timer.mark_end("decode")
        assert timer.decode_step_count == 5
        assert timer.elapsed_ms("decode") >= 0.0

    def test_elapsed_unknown_phase_raises(self):
        """Requesting elapsed time for unknown phase raises KeyError."""
        timer = PhaseTimer()
        with pytest.raises(KeyError):
            timer.elapsed_ms("nonexistent")

    def test_reset(self):
        """Reset clears all recorded events."""
        timer = PhaseTimer()
        timer.mark_start("x")
        timer.mark_end("x")
        timer.reset()
        with pytest.raises(KeyError):
            timer.elapsed_ms("x")

    def test_summary(self):
        """summary() returns a dict with all recorded phases."""
        timer = PhaseTimer()
        timer.mark_start("encode")
        timer.mark_end("encode")
        timer.mark_start("prefill")
        timer.mark_end("prefill")
        s = timer.summary()
        assert "encode" in s
        assert "prefill" in s
        assert isinstance(s["encode"], float)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/sum_young/code/projects/vlla && python -m pytest tests/test_timing.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.utils.timing'`

- [ ] **Step 3: Write the implementation**

```python
# src/utils/timing.py
"""
PhaseTimer: CUDA event-based timing for E/P/D phase measurement.
Falls back to time.perf_counter when CUDA is not available.
"""

import time
from typing import Dict, List, Optional, Tuple

import torch


class PhaseTimer:
    """Precise GPU timing using CUDA events, with CPU fallback."""

    def __init__(self):
        self._use_cuda = torch.cuda.is_available()
        self._events: Dict[str, List[Tuple]] = {}  # phase -> [(start, end), ...]
        self.decode_step_count: int = 0

    def mark_start(self, phase: str):
        if self._use_cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
        else:
            event = time.perf_counter()
        if phase not in self._events:
            self._events[phase] = []
        self._events[phase].append((event, None))

    def mark_end(self, phase: str):
        if phase not in self._events or not self._events[phase]:
            raise KeyError(f"No start event for phase '{phase}'")

        start, _ = self._events[phase][-1]

        if self._use_cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
        else:
            event = time.perf_counter()

        self._events[phase][-1] = (start, event)

        if phase == "decode":
            self.decode_step_count += 1

    def elapsed_ms(self, phase: str) -> float:
        if phase not in self._events:
            raise KeyError(f"No events recorded for phase '{phase}'")

        if self._use_cuda:
            torch.cuda.synchronize()

        total = 0.0
        for start, end in self._events[phase]:
            if end is None:
                continue
            if self._use_cuda:
                total += start.elapsed_time(end)  # ms
            else:
                total += (end - start) * 1000.0  # s -> ms
        return total

    def summary(self) -> Dict[str, float]:
        result = {}
        for phase in self._events:
            result[phase] = self.elapsed_ms(phase)
        return result

    def reset(self):
        self._events.clear()
        self.decode_step_count = 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/sum_young/code/projects/vlla && python -m pytest tests/test_timing.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/sum_young/code/projects/vlla
git add src/utils/timing.py tests/test_timing.py
git commit -m "feat: add PhaseTimer with CUDA event timing"
```

---

## Task 6: Implement BaseVLMController

**Files:**
- Create: `/Users/sum_young/code/projects/vlla/src/controllers/base_vlm_controller.py`
- Modify: `/Users/sum_young/code/projects/vlla/src/controllers/__init__.py`

- [ ] **Step 1: Write `base_vlm_controller.py`**

```python
# src/controllers/base_vlm_controller.py
"""
Base controller for autoregressive VLM models.
Implements profiling and analysis hook registration using E/P/D phase model.
"""

import logging
from abc import abstractmethod
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn

from src.core.probe_core import BaseController
from src.utils.timing import PhaseTimer


class BaseVLMController(BaseController):
    """
    VLM-specific base controller. Subclasses provide model-specific accessors.
    Provides:
    - Profiling hooks for E/P/D timing via PhaseTimer
    - Analysis hooks for QKV tensor collection at specified layers
    """

    PHASES = ["encode", "prefill", "decode"]

    def __init__(
        self,
        model_name: str,
        pipeline: Any = None,
        logger: Optional[logging.Logger] = None,
        store_type: Optional[Union[str, List[str]]] = None,
        store_layers: Optional[Union[int, List[int]]] = None,
        store_phases: Optional[List[str]] = None,
        hook_mode: Optional[str] = "analysis",
        generation_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            pipeline=pipeline,
            logger=logger,
            store_type=store_type,
            store_layers=store_layers,
            store_phases=store_phases,
            hook_mode=hook_mode,
        )
        self.generation_kwargs = generation_kwargs or {}
        self.timer = PhaseTimer()

        # Resolve -1 in store_layers to all layer indices
        layer_blocks = self.get_layer_blocks()
        self.num_layers = len(layer_blocks)
        if self.store_layers == [-1]:
            self.store_layers = list(range(self.num_layers))

    # ---- Subclass must implement ----

    @abstractmethod
    def get_vision_encoder(self) -> nn.Module:
        ...

    @abstractmethod
    def get_language_model(self) -> nn.Module:
        ...

    @abstractmethod
    def get_layer_blocks(self) -> List[nn.Module]:
        ...

    @staticmethod
    @abstractmethod
    def init_pipeline(cfg) -> Any:
        ...

    # ---- Profiling hooks ----

    def register_profiling_hooks(self):
        """Insert CUDA event markers at E/P/D boundaries. Lightweight, no tensor copies."""
        self.timer.reset()
        vision_encoder = self.get_vision_encoder()
        layer_blocks = self.get_layer_blocks()
        first_layer = layer_blocks[0]
        last_layer = layer_blocks[-1]

        # E: vision encoder pre/post
        def _encode_pre(module, inputs):
            self.timer.mark_start("encode")

        def _encode_post(module, inputs, output):
            self.timer.mark_end("encode")

        h1 = vision_encoder.register_forward_pre_hook(_encode_pre)
        h2 = vision_encoder.register_forward_hook(_encode_post)
        self.analysis_hooks["profiling_encode_pre"] = h1
        self.analysis_hooks["profiling_encode_post"] = h2

        # P/D: first layer pre-hook determines phase by seq_len
        def _layer_first_pre(module, inputs):
            hidden = inputs[0] if isinstance(inputs, tuple) else inputs
            seq_len = hidden.shape[1] if hidden.dim() >= 2 else 1
            phase = "prefill" if seq_len > 1 else "decode"
            self.cur_phase = phase
            self.timer.mark_start(phase)

        def _layer_last_post(module, inputs, output):
            self.timer.mark_end(self.cur_phase)

        h3 = first_layer.register_forward_pre_hook(_layer_first_pre)
        h4 = last_layer.register_forward_hook(_layer_last_post)
        self.analysis_hooks["profiling_layer_first_pre"] = h3
        self.analysis_hooks["profiling_layer_last_post"] = h4

    # ---- Analysis hooks ----

    def register_analysis_hooks(self):
        """Register hooks to capture QKV tensors at specified layers.
        Subclasses can override for model-specific attention module paths."""
        layer_blocks = self.get_layer_blocks()
        for layer_idx in self.store_layers:
            if layer_idx >= len(layer_blocks):
                continue
            block = layer_blocks[layer_idx]
            self._register_layer_analysis_hooks(block, layer_idx)

    def _register_layer_analysis_hooks(self, block: nn.Module, layer_idx: int):
        """Register analysis hooks for one layer. Subclass override for model-specific paths."""
        pass

    def postprocess(self, **kwargs):
        """Flush step_store into global_store."""
        self.flush_step_store()
```

- [ ] **Step 2: Update `src/controllers/__init__.py`**

```python
from src.core.probe_core import Registry

CONTROLLER_REGISTRY = Registry("controller")
```

(QwenVLController will register itself in the next task.)

- [ ] **Step 3: Commit**

```bash
cd /Users/sum_young/code/projects/vlla
git add src/controllers/base_vlm_controller.py src/controllers/__init__.py
git commit -m "feat: add BaseVLMController with profiling and analysis hooks"
```

---

## Task 7: Implement QwenVLController

**Files:**
- Create: `/Users/sum_young/code/projects/vlla/src/controllers/qwen_vl_controller.py`
- Modify: `/Users/sum_young/code/projects/vlla/src/controllers/__init__.py`

- [ ] **Step 1: Write `qwen_vl_controller.py`**

Migrated from rope2sink's `src/controllers/qwenvl_controller.py`, adapted to the new inheritance chain.

```python
# src/controllers/qwen_vl_controller.py
"""
Qwen2.5-VL specific controller.
Provides model loading, input preparation, and inference for Qwen2.5-VL models.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from easydict import EasyDict as edict

from src.controllers.base_vlm_controller import BaseVLMController
from src.controllers import CONTROLLER_REGISTRY


class QwenVLController(BaseVLMController):

    def __init__(
        self,
        model_name: str,
        pipeline: Any = None,
        logger: Optional[logging.Logger] = None,
        store_type: Optional[Union[str, List[str]]] = None,
        store_layers: Optional[Union[int, List[int]]] = None,
        store_phases: Optional[List[str]] = None,
        hook_mode: Optional[str] = "analysis",
        generation_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        if pipeline is None:
            pipeline = self.init_pipeline(
                edict({"model_name": model_name, **(generation_kwargs or {})})
            )
        super().__init__(
            model_name=model_name,
            pipeline=pipeline,
            logger=logger,
            store_type=store_type,
            store_layers=store_layers,
            store_phases=store_phases,
            hook_mode=hook_mode,
            generation_kwargs=generation_kwargs,
            **kwargs,
        )
        model = self.pipeline.model
        self.num_heads = model.config.num_attention_heads
        self.hidden_size = model.config.hidden_size

    # ---- Abstract method implementations ----

    def get_vision_encoder(self) -> nn.Module:
        return self.pipeline.model.visual

    def get_language_model(self) -> nn.Module:
        return self.pipeline.model.model

    def get_layer_blocks(self) -> List[nn.Module]:
        return list(self.pipeline.model.model.layers)

    @staticmethod
    def init_pipeline(cfg) -> Any:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        model_path = cfg.model_name
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        min_pixels = getattr(cfg, "min_pixels", 256 * 28 * 28)
        max_pixels = getattr(cfg, "max_pixels", 1280 * 28 * 28)
        processor = AutoProcessor.from_pretrained(
            model_path, min_pixels=min_pixels, max_pixels=max_pixels
        )
        return edict(model=model, processor=processor)

    def prepare_inputs(self, cfg) -> list:
        inputs_list = []
        raw_inputs = cfg.get("inputs", [])
        for item in raw_inputs:
            name = item.get("name", "unnamed")
            messages = item.get("messages", [])
            inputs_list.append({"name": name, "messages": messages})
        return inputs_list

    @torch.no_grad()
    def model_inference(self, pipeline, cfg, inputs) -> Any:
        from qwen_vl_utils import process_vision_info

        processor = pipeline.processor
        model = pipeline.model
        messages = inputs.get("messages")
        if not isinstance(messages, list):
            messages = [messages]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        processed = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        processed = processed.to(model.device)

        max_new_tokens = getattr(cfg, "max_new_tokens", 128)
        generated_ids = model.generate(**processed, max_new_tokens=max_new_tokens)

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(processed.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return {"text": output_text, "input_ids": processed.input_ids}

    def save_results(self, inputs, results, cfg) -> str:
        name = inputs.get("name", "unnamed")
        save_dir = os.path.join(cfg.get("output_path", "./outputs"), name)
        os.makedirs(save_dir, exist_ok=True)
        result_path = os.path.join(save_dir, "result.json")
        with open(result_path, "w") as f:
            json.dump({
                "name": name,
                "output_text": results.get("text", []),
                "num_input_tokens": results["input_ids"].shape[1] if "input_ids" in results else 0,
            }, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Results saved to {result_path}")
        return save_dir

    def _register_layer_analysis_hooks(self, block: nn.Module, layer_idx: int):
        """Register QKV capture hooks on Qwen2.5-VL self-attention layers."""
        self_attn = block.self_attn

        if "self_q" in self.store_type or "self_k" in self.store_type:
            self.HookManager.register_extractor_log_action_abs(
                controller=self,
                model=self.pipeline.model,
                abs_path=f"model.layers[{layer_idx}].self_attn",
                key=f"{layer_idx}_qk_states",
                extractor=self._extract_qk_states,
                collection="analysis_hooks",
                name_suffix=f"layer{layer_idx}_qk",
            )

    @staticmethod
    def _extract_qk_states(module, inputs, output):
        """Extract Q and K states from Qwen2.5-VL self_attn output.
        Note: exact extraction depends on model version; may need adjustment."""
        if hasattr(module, '_last_q') and hasattr(module, '_last_k'):
            return torch.stack([module._last_q, module._last_k], dim=0)
        if isinstance(output, tuple) and len(output) >= 2:
            return output[0]
        return output[0] if isinstance(output, tuple) else output


CONTROLLER_REGISTRY.register("QwenVLController", QwenVLController)
```

- [ ] **Step 2: Update `src/controllers/__init__.py` to import the controller**

```python
from src.core.probe_core import Registry

CONTROLLER_REGISTRY = Registry("controller")

# Import triggers registration
from src.controllers.qwen_vl_controller import QwenVLController  # noqa: F401
```

- [ ] **Step 3: Commit**

```bash
cd /Users/sum_young/code/projects/vlla
git add src/controllers/qwen_vl_controller.py src/controllers/__init__.py
git commit -m "feat: add QwenVLController with model loading and inference"
```

---

## Task 8: Implement profiling task

**Files:**
- Create: `/Users/sum_young/code/projects/vlla/src/tasks/profiling_task.py`
- Modify: `/Users/sum_young/code/projects/vlla/src/tasks/__init__.py`

- [ ] **Step 1: Write `profiling_task.py`**

```python
# src/tasks/profiling_task.py
"""
E/P/D profiling task: collects timing data from PhaseTimer and outputs JSON summary.
"""

import json
import os
from typing import Any, Dict

from src.tasks import TASK_REGISTRY


def task_epd_profiling(controller: Any, save_dir: str, task_config: Dict) -> None:
    """
    Aggregate PhaseTimer results and write timing JSON.

    Expected controller state: controller.timer has recorded encode/prefill/decode events.
    """
    task_dir = os.path.join(save_dir, "epd_profiling")
    os.makedirs(task_dir, exist_ok=True)

    timer = controller.timer
    timing = {}
    for phase in ["encode", "prefill", "decode"]:
        try:
            timing[phase] = round(timer.elapsed_ms(phase), 3)
        except KeyError:
            timing[phase] = None

    decode_steps = timer.decode_step_count
    if timing.get("decode") is not None and decode_steps > 0:
        timing["decode_per_token"] = round(timing["decode"] / decode_steps, 3)
    else:
        timing["decode_per_token"] = None

    include_per_token = task_config.get("include_per_token_decode", True)
    if not include_per_token:
        timing.pop("decode_per_token", None)

    result = {
        "model": controller.model_name,
        "timing_ms": timing,
        "decode_steps": decode_steps,
    }

    output_path = os.path.join(task_dir, "epd_timing.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    controller.logger.info(f"Profiling results saved to {output_path}")


TASK_REGISTRY.register("epd_profiling", task_epd_profiling)
```

- [ ] **Step 2: Update `src/tasks/__init__.py`**

```python
from src.core.probe_core import Registry

TASK_REGISTRY = Registry("task")

# Import triggers registration
from src.tasks.profiling_task import task_epd_profiling  # noqa: F401
```

- [ ] **Step 3: Commit**

```bash
cd /Users/sum_young/code/projects/vlla
git add src/tasks/profiling_task.py src/tasks/__init__.py
git commit -m "feat: add epd_profiling task"
```

---

## Task 9: Implement attention analysis tasks

**Files:**
- Create: `/Users/sum_young/code/projects/vlla/src/tasks/attention_task.py`
- Modify: `/Users/sum_young/code/projects/vlla/src/tasks/__init__.py`

- [ ] **Step 1: Write `attention_task.py`**

```python
# src/tasks/attention_task.py
"""
Attention analysis tasks for VLM prefill characterization.
- visual_text_attention: sparsity and concentration of cross-modal attention
- sink_detection: identify tokens receiving disproportionate attention
- per_layer_stats: layer-wise attention pattern statistics
"""

import json
import os
from typing import Any, Dict, List

import torch

from src.tasks import TASK_REGISTRY


def _get_task_dir(save_dir: str, task_name: str) -> str:
    task_dir = os.path.join(save_dir, task_name)
    os.makedirs(task_dir, exist_ok=True)
    return task_dir


def _compute_attention_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Compute scaled dot-product attention scores (no softmax).
    q, k: [seq_len, num_heads, head_dim] or [batch, seq_len, num_heads, head_dim]
    Returns: [num_heads, seq_len, seq_len] attention scores.
    """
    if q.dim() == 4:
        q, k = q[0], k[0]
    # [seq, heads, dim] -> [heads, seq, dim]
    q = q.permute(1, 0, 2).float()
    k = k.permute(1, 0, 2).float()
    scale = q.shape[-1] ** 0.5
    scores = torch.bmm(q, k.transpose(1, 2)) / scale
    return scores


def _softmax_attention(scores: torch.Tensor) -> torch.Tensor:
    """Apply softmax along last dim."""
    return torch.softmax(scores, dim=-1)


def task_visual_text_attn(controller: Any, save_dir: str, task_config: Dict) -> None:
    """
    Analyze visual-to-text and text-to-visual attention patterns.
    Requires QK tensors in global_store with keys like '{layer_idx}_qk_states'.
    """
    task_dir = _get_task_dir(save_dir, "visual_text_attention")
    num_visual = task_config.get("num_visual_tokens", None)

    results_per_layer = {}
    for key, tensors in controller.global_store.items():
        if not key.endswith("_qk_states"):
            continue
        layer_idx = key.split("_")[0]

        for t in tensors:
            if t.dim() >= 3 and t.shape[0] == 2:
                q, k = t[0], t[1]
            else:
                controller.logger.warning(f"Unexpected tensor shape for {key}: {t.shape}")
                continue

            scores = _compute_attention_scores(q, k)
            attn = _softmax_attention(scores)  # [heads, seq, seq]
            seq_len = attn.shape[-1]

            if num_visual is None:
                num_visual = seq_len // 2

            # text-to-visual: rows=text tokens, cols=visual tokens
            text_to_vis = attn[:, num_visual:, :num_visual]  # [heads, text_len, vis_len]
            vis_to_text = attn[:, :num_visual, num_visual:]

            # Sparsity: Gini coefficient per head
            def gini(x):
                x_sorted, _ = torch.sort(x.flatten())
                n = x_sorted.shape[0]
                index = torch.arange(1, n + 1, dtype=x_sorted.dtype)
                return (2 * (index * x_sorted).sum() / (n * x_sorted.sum()) - (n + 1) / n).item()

            # Top-k concentration
            k_val = task_config.get("top_k", 10)
            text_to_vis_flat = text_to_vis.mean(dim=0)  # [text_len, vis_len]
            topk_vals, _ = torch.topk(text_to_vis_flat, min(k_val, text_to_vis_flat.shape[-1]), dim=-1)
            topk_concentration = (topk_vals.sum() / text_to_vis_flat.sum()).item()

            results_per_layer[str(layer_idx)] = {
                "text_to_visual_gini": gini(text_to_vis),
                "visual_to_text_gini": gini(vis_to_text),
                f"text_to_visual_top{k_val}_concentration": round(topk_concentration, 4),
                "num_visual_tokens": num_visual,
                "seq_len": seq_len,
            }

    output_path = os.path.join(task_dir, "sparsity_stats.json")
    with open(output_path, "w") as f:
        json.dump(results_per_layer, f, indent=2)
    controller.logger.info(f"Visual-text attention stats saved to {output_path}")


def task_sink_detection(controller: Any, save_dir: str, task_config: Dict) -> None:
    """
    Detect attention sink tokens: tokens that receive disproportionate attention.
    """
    task_dir = _get_task_dir(save_dir, "sink_detection")
    num_sink = task_config.get("num_sink_tokens", 12)
    num_visual = task_config.get("num_visual_tokens", None)

    all_layers = {}
    for key, tensors in controller.global_store.items():
        if not key.endswith("_qk_states"):
            continue
        layer_idx = key.split("_")[0]

        for t in tensors:
            if t.dim() >= 3 and t.shape[0] == 2:
                q, k = t[0], t[1]
            else:
                continue

            scores = _compute_attention_scores(q, k)
            attn = _softmax_attention(scores)
            # Attention received per token: sum over query dimension, mean over heads
            attn_received = attn.mean(dim=0).sum(dim=0)  # [seq_len]
            seq_len = attn_received.shape[0]

            if num_visual is None:
                num_visual = seq_len // 2

            topk_vals, topk_idx = torch.topk(attn_received, min(num_sink, seq_len))
            sink_info = []
            for val, idx in zip(topk_vals.tolist(), topk_idx.tolist()):
                region = "visual" if idx < num_visual else "text"
                if idx == num_visual - 1 or idx == num_visual:
                    region = "boundary"
                sink_info.append({
                    "token_idx": idx,
                    "attention_received": round(val, 4),
                    "region": region,
                })

            all_layers[str(layer_idx)] = {
                "sinks": sink_info,
                "total_attention": round(attn_received.sum().item(), 4),
                "sink_share": round(topk_vals.sum().item() / attn_received.sum().item(), 4),
            }

    output_path = os.path.join(task_dir, "sink_tokens.json")
    with open(output_path, "w") as f:
        json.dump(all_layers, f, indent=2)
    controller.logger.info(f"Sink detection results saved to {output_path}")


def task_per_layer_stats(controller: Any, save_dir: str, task_config: Dict) -> None:
    """
    Per-layer attention statistics: entropy, sparsity, mean attention.
    """
    task_dir = _get_task_dir(save_dir, "per_layer_stats")

    stats = {}
    for key, tensors in controller.global_store.items():
        if not key.endswith("_qk_states"):
            continue
        layer_idx = key.split("_")[0]

        for t in tensors:
            if t.dim() >= 3 and t.shape[0] == 2:
                q, k = t[0], t[1]
            else:
                continue

            scores = _compute_attention_scores(q, k)
            attn = _softmax_attention(scores)  # [heads, seq, seq]

            # Entropy per head (mean over queries)
            eps = 1e-10
            entropy = -(attn * (attn + eps).log()).sum(dim=-1).mean(dim=-1)  # [heads]

            stats[str(layer_idx)] = {
                "mean_entropy": round(entropy.mean().item(), 4),
                "min_entropy": round(entropy.min().item(), 4),
                "max_entropy": round(entropy.max().item(), 4),
                "mean_attn_value": round(attn.mean().item(), 6),
                "num_heads": attn.shape[0],
                "seq_len": attn.shape[-1],
            }

    output_path = os.path.join(task_dir, "layer_stats.json")
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    controller.logger.info(f"Per-layer stats saved to {output_path}")


TASK_REGISTRY.register("visual_text_attention", task_visual_text_attn)
TASK_REGISTRY.register("sink_detection", task_sink_detection)
TASK_REGISTRY.register("per_layer_stats", task_per_layer_stats)
```

- [ ] **Step 2: Update `src/tasks/__init__.py`**

```python
from src.core.probe_core import Registry

TASK_REGISTRY = Registry("task")

# Import triggers registration
from src.tasks.profiling_task import task_epd_profiling  # noqa: F401
from src.tasks.attention_task import (  # noqa: F401
    task_visual_text_attn,
    task_sink_detection,
    task_per_layer_stats,
)
```

- [ ] **Step 3: Commit**

```bash
cd /Users/sum_young/code/projects/vlla
git add src/tasks/attention_task.py src/tasks/__init__.py
git commit -m "feat: add attention analysis tasks (visual_text, sink, per_layer)"
```

---

## Task 10: Implement run_tasks.py and Hydra configs

**Files:**
- Create: `/Users/sum_young/code/projects/vlla/src/run_tasks.py`
- Create: `/Users/sum_young/code/projects/vlla/configs/base.yaml`
- Create: `/Users/sum_young/code/projects/vlla/configs/qwen_vl_7b/profiling.yaml`
- Create: `/Users/sum_young/code/projects/vlla/configs/qwen_vl_7b/attention.yaml`

- [ ] **Step 1: Write `src/run_tasks.py`**

```python
# src/run_tasks.py
"""
Unified entry point for VLM profiling and analysis tasks.
Usage: python src/run_tasks.py --config-name qwen_vl_7b/profiling
"""

import logging
import os

import hydra
from easydict import EasyDict as edict
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.controllers import CONTROLLER_REGISTRY
from src.tasks import TASK_REGISTRY


def main(cfg: DictConfig):
    config = edict(OmegaConf.to_container(cfg, resolve=True))

    base_output_path = config.get("base_output_path", "./outputs")
    model_name = config.get("model_name", "unknown")
    output_root = os.path.join(base_output_path, model_name)
    os.makedirs(output_root, exist_ok=True)
    config.output_path = output_root

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    logger.info(f"Model: {config.model_name}")
    logger.info(f"Tasks: {config.tasks}")
    logger.info(f"Output: {config.output_path}")

    # Initialize controller
    controller_name = config.controller_name
    ControllerClass = CONTROLLER_REGISTRY[controller_name]
    pipeline = ControllerClass.init_pipeline(config)
    controller = ControllerClass(
        pipeline=pipeline,
        logger=logger,
        model_name=config.model_name,
        **config.controller_config,
    )

    controller.register_hooks()

    # Process each input
    inputs_list = controller.prepare_inputs(config)
    tasks = config.get("tasks", [])
    tasks_config = config.get("task_config", {})

    num_runs = config.get("num_benchmark_runs", 1)
    num_warmup = config.get("num_warmup_runs", 0)

    for input_idx, inputs in tqdm(
        enumerate(inputs_list), total=len(inputs_list), desc="Processing inputs"
    ):
        # Warmup runs (profiling mode)
        if controller.profiling_mode and num_warmup > 0:
            logger.info(f"Running {num_warmup} warmup iterations...")
            for _ in range(num_warmup):
                controller.timer.reset()
                controller.model_inference(pipeline=pipeline, cfg=config, inputs=inputs)
                controller.reset_state()

        # Benchmark runs (profiling mode: multiple runs for mean/std)
        if controller.profiling_mode and num_runs > 1:
            import statistics
            all_timings = {phase: [] for phase in ["encode", "prefill", "decode"]}
            for run_idx in range(num_runs):
                controller.timer.reset()
                result = controller.model_inference(pipeline=pipeline, cfg=config, inputs=inputs)
                for phase in all_timings:
                    try:
                        all_timings[phase].append(controller.timer.elapsed_ms(phase))
                    except KeyError:
                        pass
                controller.reset_state()

            # Store aggregated timing on controller for task consumption
            controller.timer.reset()
            controller._aggregated_timing = {
                phase: {
                    "mean": round(statistics.mean(vals), 3) if vals else None,
                    "std": round(statistics.stdev(vals), 3) if len(vals) > 1 else 0.0,
                }
                for phase, vals in all_timings.items()
            }
            controller._aggregated_timing["num_runs"] = num_runs
            save_dir = os.path.join(output_root, inputs.get("name", f"input_{input_idx}"))
            os.makedirs(save_dir, exist_ok=True)
        else:
            # Single run (analysis mode or single profiling run)
            result = controller.model_inference(pipeline=pipeline, cfg=config, inputs=inputs)
            save_dir = controller.save_results(inputs, result, config)
            controller.postprocess()

        # Execute tasks
        for task_name in tasks:
            if task_name in TASK_REGISTRY:
                logger.info(f"Executing task: {task_name}")
                task_specific_config = tasks_config.get(task_name, {})
                try:
                    TASK_REGISTRY[task_name](
                        controller=controller,
                        save_dir=save_dir,
                        task_config=task_specific_config,
                    )
                except Exception as e:
                    logger.error(f"Task {task_name} failed: {e}")
                    raise
            else:
                logger.warning(f"Unknown task: {task_name}")

        controller.reset_state()

    controller.remove_hooks()
    logger.info("All tasks completed.")


@hydra.main(
    config_path="../configs",
    config_name="qwen_vl_7b/profiling",
    version_base=None,
)
def hydra_main(cfg: DictConfig):
    main(cfg)


if __name__ == "__main__":
    hydra_main()
```

- [ ] **Step 2: Write `configs/base.yaml`**

```yaml
debug: false
device: "cuda:0"
seed: 42
base_output_path: ./outputs
num_warmup_runs: 3
num_benchmark_runs: 10
max_new_tokens: 128
```

- [ ] **Step 3: Write `configs/qwen_vl_7b/profiling.yaml`**

```yaml
defaults:
  - ../base

model_name: "Qwen/Qwen2.5-VL-7B-Instruct"
controller_name: "QwenVLController"
controller_config:
  mode: profiling
  store_phases: ["encode", "prefill", "decode"]

tasks: ["epd_profiling"]
task_config:
  epd_profiling:
    output_format: "json"
    include_per_token_decode: true

inputs:
  - name: "text_only"
    messages:
      - role: "user"
        content:
          - type: "text"
            text: "What is the capital of France?"
  - name: "single_image"
    messages:
      - role: "user"
        content:
          - type: "image"
            image: "assets/demo.jpg"
          - type: "text"
            text: "Describe this image in detail."
  - name: "multi_image"
    messages:
      - role: "user"
        content:
          - type: "image"
            image: "assets/img1.jpg"
          - type: "image"
            image: "assets/img2.jpg"
          - type: "text"
            text: "Compare these two images."
```

- [ ] **Step 4: Write `configs/qwen_vl_7b/attention.yaml`**

```yaml
defaults:
  - ../base

model_name: "Qwen/Qwen2.5-VL-7B-Instruct"
controller_name: "QwenVLController"
controller_config:
  mode: analysis
  store_phases: ["prefill"]
  store_layers: [0, 7, 14, 21, 27]
  store_type: ["self_q", "self_k"]

num_benchmark_runs: 1
num_warmup_runs: 0

tasks: ["visual_text_attention", "sink_detection", "per_layer_stats"]
task_config:
  visual_text_attention:
    top_k: 10
  sink_detection:
    num_sink_tokens: 12
```

- [ ] **Step 5: Commit**

```bash
cd /Users/sum_young/code/projects/vlla
git add src/run_tasks.py configs/
git commit -m "feat: add Hydra entry point and experiment configs"
```

---

## Task 11: Integration smoke test

**Files:**
- Create: `/Users/sum_young/code/projects/vlla/tests/test_controller_init.py`

- [ ] **Step 1: Write smoke test**

This test verifies the import chain and registry wiring work, without requiring a GPU or model weights.

```python
# tests/test_controller_init.py
"""
Smoke tests for framework wiring — no GPU or model weights required.
"""

import pytest
from src.core.probe_core import BaseController, HookMode, Registry
from src.controllers import CONTROLLER_REGISTRY
from src.tasks import TASK_REGISTRY


class TestRegistryWiring:
    def test_controller_registry_has_qwen(self):
        assert "QwenVLController" in CONTROLLER_REGISTRY

    def test_task_registry_has_all_tasks(self):
        expected = ["epd_profiling", "visual_text_attention", "sink_detection", "per_layer_stats"]
        for task_name in expected:
            assert task_name in TASK_REGISTRY, f"Missing task: {task_name}"

    def test_hook_mode_profiling_exists(self):
        assert HookMode.PROFILING.value == "profiling"

    def test_base_controller_has_phase_support(self):
        """BaseController accepts store_phases parameter."""
        # Cannot instantiate (abstract), but verify the __init__ signature
        import inspect
        sig = inspect.signature(BaseController.__init__)
        assert "store_phases" in sig.parameters

    def test_registry_unknown_key_raises(self):
        r = Registry("test")
        with pytest.raises(KeyError):
            r["nonexistent"]

    def test_registry_duplicate_raises(self):
        r = Registry("test")
        r.register("a", 1)
        with pytest.raises(ValueError):
            r.register("a", 2)
```

- [ ] **Step 2: Run tests**

Run: `cd /Users/sum_young/code/projects/vlla && python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
cd /Users/sum_young/code/projects/vlla
git add tests/test_controller_init.py
git commit -m "test: add integration smoke tests for registry and framework wiring"
```

---

## Task 12: Create test assets and validate end-to-end on GPU

**Files:**
- Create: `/Users/sum_young/code/projects/vlla/assets/demo.jpg` (any small image)

- [ ] **Step 1: Create a test image**

```bash
cd /Users/sum_young/code/projects/vlla
mkdir -p assets
# Create a minimal test image (solid color, 224x224)
python3 -c "
from PIL import Image
img = Image.new('RGB', (224, 224), color='blue')
img.save('assets/demo.jpg')
img.save('assets/img1.jpg')
Image.new('RGB', (224, 224), color='red').save('assets/img2.jpg')
"
```

- [ ] **Step 2: Run profiling end-to-end on GPU**

```bash
cd /Users/sum_young/code/projects/vlla
python src/run_tasks.py --config-name qwen_vl_7b/profiling
```

Expected: JSON files in `outputs/Qwen2.5-VL-7B-Instruct/{text_only,single_image,multi_image}/epd_profiling/epd_timing.json` with timing data.

- [ ] **Step 3: Run attention analysis end-to-end on GPU**

```bash
cd /Users/sum_young/code/projects/vlla
python src/run_tasks.py --config-name qwen_vl_7b/attention
```

Expected: JSON files in `outputs/Qwen2.5-VL-7B-Instruct/single_image/{visual_text_attention,sink_detection,per_layer_stats}/` with analysis results.

- [ ] **Step 4: Inspect outputs and verify data makes sense**

```bash
cat outputs/Qwen2.5-VL-7B-Instruct/text_only/epd_profiling/epd_timing.json
cat outputs/Qwen2.5-VL-7B-Instruct/single_image/sink_detection/sink_tokens.json
```

Verify: encode time is 0 or near-0 for text_only, non-trivial for single_image. Sink tokens should show concentration at specific positions.

- [ ] **Step 5: Commit test assets**

```bash
cd /Users/sum_young/code/projects/vlla
git add assets/
git commit -m "chore: add test image assets for E2E validation"
```

---

## Summary

| Task | Description | Estimated Steps |
|------|------------|----------------|
| 1 | model-probe-core: hooks.py | 4 |
| 2 | model-probe-core: state.py + registry.py | 4 |
| 3 | model-probe-core: controller.py | 3 |
| 4 | vlla: submodule + scaffold | 3 |
| 5 | vlla: PhaseTimer (TDD) | 5 |
| 6 | vlla: BaseVLMController | 3 |
| 7 | vlla: QwenVLController | 3 |
| 8 | vlla: profiling task | 3 |
| 9 | vlla: attention tasks | 3 |
| 10 | vlla: run_tasks.py + configs | 5 |
| 11 | vlla: smoke tests | 3 |
| 12 | vlla: E2E GPU validation | 5 |
| **Total** | | **44 steps** |

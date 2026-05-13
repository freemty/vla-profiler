"""
Base controller combining HookManager + StoreMixin.

Provides the abstract skeleton that model-specific controllers
(e.g., FluxController, QwenVLController) inherit from.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from src.core.probe_core.hooks import HookManager
from src.core.probe_core.state import StoreMixin


class HookMode(Enum):
    """Operational mode for the controller's hook system."""

    ANALYSIS = "analysis"
    INTERVENE = "intervene"
    PROFILING = "profiling"
    BOTH = "both"  # analysis + intervene
    NONE = "none"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_HOOK_MODE_MAP: Dict[str, HookMode] = {
    "analysis": HookMode.ANALYSIS,
    "intervene": HookMode.INTERVENE,
    "profiling": HookMode.PROFILING,
    "both": HookMode.BOTH,
    "none": HookMode.NONE,
}


def _parse_string_list(
    value: Optional[Union[str, List[str]]],
) -> List[str]:
    """Normalize *value* into a lowercase string list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value.lower()]
    return [str(x).lower() for x in value]


def _parse_int_list(
    value: Optional[Union[int, List[int]]],
    *,
    default_sentinel: int = -1,
) -> List[int]:
    """Normalize *value* into an int list; ``-1`` means 'all'."""
    if value is None:
        return [default_sentinel]
    if isinstance(value, int):
        return [value]
    return list(value)


# ------------------------------------------------------------------
# BaseController
# ------------------------------------------------------------------

class BaseController(StoreMixin, ABC):
    """
    Abstract base hook controller for collecting / modifying model
    intermediate activations.

    Subclasses **must** implement:
        - ``prepare_inputs()``
        - ``model_inference()``

    Subclasses **may** override:
        - ``register_analysis_hooks()``
        - ``register_intervention_hooks()``
        - ``register_profiling_hooks()``
        - ``save_results()``
        - ``postprocess()``
    """

    # Expose HookManager as a class attribute so subclasses can use
    # ``self.HookManager.resolve_absolute(...)`` etc.
    HookManager = HookManager

    def __init__(
        self,
        model_name: str,
        pipeline: Any = None,
        logger: Optional[logging.Logger] = None,
        store_type: Optional[Union[str, List[str]]] = None,
        store_layers: Optional[Union[int, List[int]]] = None,
        store_phases: Optional[Union[str, List[str]]] = None,
        hook_mode: Optional[str] = "analysis",
    ) -> None:
        # Logger
        if logger is None:
            self.logger = logging.getLogger(f"probe_core.{self.__class__.__name__}")
            if not self.logger.handlers:
                logging.basicConfig(level=logging.INFO)
        else:
            self.logger = logger

        self.model_name = model_name
        self.pipeline = pipeline

        # Hook mode
        mode_str = str(hook_mode).lower() if hook_mode is not None else "analysis"
        if mode_str not in _HOOK_MODE_MAP:
            raise ValueError(
                f"Unsupported hook_mode '{mode_str}'. "
                f"Expected one of: {list(_HOOK_MODE_MAP.keys())}"
            )
        self.hook_mode: HookMode = _HOOK_MODE_MAP[mode_str]

        # Storage config
        self.store_type: List[str] = _parse_string_list(store_type) if self.analysis_mode else []
        self.store_layers: List[int] = _parse_int_list(store_layers)
        self.store_phases: List[str] = _parse_string_list(store_phases) if store_phases else ["-1"]

        # State (from StoreMixin)
        self.reset_state()

        # Current phase tracking (set externally, e.g., "encode", "prefill", "decode")
        self.cur_phase: str = ""

        # Hook storage
        self._reset_hooks()

        self.logger.info(
            "Initialized %s (mode=%s, layers=%s, phases=%s)",
            self.__class__.__name__,
            self.hook_mode.value,
            self.store_layers,
            self.store_phases,
        )

    # ------------------------------------------------------------------
    # Mode properties
    # ------------------------------------------------------------------

    @property
    def analysis_mode(self) -> bool:
        return self.hook_mode in (HookMode.ANALYSIS, HookMode.BOTH)

    @property
    def intervention_mode(self) -> bool:
        return self.hook_mode in (HookMode.INTERVENE, HookMode.BOTH)

    @property
    def profiling_mode(self) -> bool:
        return self.hook_mode == HookMode.PROFILING

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def should_store(self, store_key: str) -> bool:
        """
        Determine whether data for *store_key* should be captured.

        Checks both layer index (parsed from the key prefix) and the
        current phase against configured filters.
        """
        # Layer filter
        layer_idx = int(store_key.split("_")[0])
        if -1 not in self.store_layers and layer_idx not in self.store_layers:
            return False

        # Phase filter ("-1" sentinel means "all phases")
        if "-1" not in self.store_phases and self.cur_phase not in self.store_phases:
            return False

        return True

    def set_phase(self, phase: str) -> None:
        """Update the current inference phase (e.g., ``"prefill"``)."""
        self.cur_phase = phase

    # ------------------------------------------------------------------
    # Hook lifecycle
    # ------------------------------------------------------------------

    def _reset_hooks(self) -> None:
        """Initialize / clear all hook dictionaries."""
        self.analysis_hooks: Dict[str, Any] = {}
        self.intervention_hooks: Dict[str, Any] = {}
        self.analysis_original_processors: Dict[Any, Any] = {}
        self.analysis_original_forwards: Dict[Any, Any] = {}
        self.intervention_original_processors: Dict[Any, Any] = {}
        self.intervention_original_forwards: Dict[Any, Any] = {}

    def register_hooks(self) -> None:
        """
        Register hooks based on the current ``hook_mode``.

        Dispatches to ``register_profiling_hooks()``,
        ``register_intervention_hooks()``, and/or
        ``register_analysis_hooks()`` as appropriate.

        Intervention hooks are registered *before* analysis hooks so that
        logging captures post-intervention values.
        """
        self.remove_hooks()
        self.logger.info("Registering hooks in %s mode", self.hook_mode.value)

        if self.profiling_mode:
            self.register_profiling_hooks()
            return

        # Intervention first, then analysis (log captures post-intervention)
        if self.intervention_mode:
            self.logger.info("Registering intervention hooks...")
            self.register_intervention_hooks()

        if self.analysis_mode:
            self.logger.info("Registering analysis hooks...")
            self.register_analysis_hooks()

    def remove_hooks(self) -> None:
        """Remove all registered hooks and restore patched methods."""
        for hook in self.analysis_hooks.values():
            hook.remove()
        for hook in self.intervention_hooks.values():
            hook.remove()
        self.analysis_hooks = {}
        self.intervention_hooks = {}

        self._restore_processors("analysis")
        self._restore_processors("intervention")
        self._restore_forwards("analysis")
        self._restore_forwards("intervention")

    def _restore_processors(self, mode: str) -> None:
        attr = f"{mode}_original_processors"
        for module, processor in getattr(self, attr, {}).items():
            module.processor = processor
        setattr(self, attr, {})

    def _restore_forwards(self, mode: str) -> None:
        attr = f"{mode}_original_forwards"
        for block, forward in getattr(self, attr, {}).items():
            block.forward = forward
        setattr(self, attr, {})

    # ------------------------------------------------------------------
    # Hook mode management
    # ------------------------------------------------------------------

    def set_hook_mode(self, mode: Union[str, HookMode]) -> None:
        """Change hook mode and re-register hooks."""
        if isinstance(mode, HookMode):
            self.hook_mode = mode
        else:
            mode_str = str(mode).lower()
            if mode_str not in _HOOK_MODE_MAP:
                raise ValueError(f"Unsupported hook mode: {mode}")
            self.hook_mode = _HOOK_MODE_MAP[mode_str]
        self.register_hooks()

    # ------------------------------------------------------------------
    # Subclass hook registration points (optional overrides)
    # ------------------------------------------------------------------

    def register_profiling_hooks(self) -> None:
        """Override to register profiling-specific hooks."""

    def register_analysis_hooks(self) -> None:
        """Override to register analysis hooks."""

    def register_intervention_hooks(self) -> None:
        """Override to register intervention hooks."""

    # ------------------------------------------------------------------
    # Abstract methods -- subclass MUST implement
    # ------------------------------------------------------------------

    @abstractmethod
    def prepare_inputs(self, pipeline: Any, cfg: Any, inputs: Any) -> Any:
        """Prepare inputs for model inference."""
        ...

    @abstractmethod
    def model_inference(self, pipeline: Any, cfg: Any, inputs: Any) -> Any:
        """Run model inference."""
        ...

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    def save_results(self, inputs: Any, results: Any, cfg: Any) -> None:
        """Save / persist inference results. No-op by default."""

    def postprocess(self, results: Any) -> Any:
        """Post-process inference results. Identity by default."""
        return results

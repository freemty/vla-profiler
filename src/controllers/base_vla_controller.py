"""
Base VLA controller for Flow/Diffusion/Single-forward action models.

Phase model: E/C/A (Encode → Context → Action)
- E: Vision encoder forward pass
- C: VLM context encoding (optional — only for VLA with VLM backbone)
- A: Action generation (flow denoise loop, DiT denoise, or single forward)

Unlike BaseVLMController (E/P/D for autoregressive models), this controller
handles iterative action generation where each denoise step is a separate
forward pass through the action head.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any, List, Optional, Union

import torch.nn as nn

from src.core.probe_core import BaseController
from src.utils.timing import PhaseTimer


logger = logging.getLogger(__name__)


class BaseVLAController(BaseController):
    """
    Abstract VLA controller with E/C/A phase awareness.

    Subclasses must implement:
        - get_vision_encoder() -> nn.Module
        - get_action_head() -> nn.Module
        - get_denoise_steps() -> int
        - init_pipeline(cfg) -> Any
        - prepare_inputs(cfg) -> list
        - model_inference(pipeline, cfg, inputs) -> Any

    Subclasses may implement:
        - get_language_model() -> nn.Module  (only if VLA has VLM backbone)
        - get_layer_blocks() -> List[nn.Module]  (for analysis hooks)
        - has_context_phase() -> bool  (default: True)
    """

    PHASES = ("encode", "context", "action")

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
        super().__init__(
            model_name=model_name,
            pipeline=pipeline,
            logger=logger,
            store_type=store_type,
            store_layers=store_layers,
            store_phases=store_phases,
            hook_mode=hook_mode,
        )
        self.timer = PhaseTimer()
        self._resolve_store_layers()

    # ---- Subclass MUST implement ----

    @abstractmethod
    def get_vision_encoder(self) -> nn.Module:
        """Return the vision encoder module (ResNet, ViT, etc.)."""
        ...

    @abstractmethod
    def get_action_head(self) -> nn.Module:
        """Return the action generation module (flow head, DiT, CVAE decoder)."""
        ...

    @abstractmethod
    def get_denoise_steps(self) -> int:
        """Return expected number of denoise/forward steps for action generation.
        1 for single-forward models (ACT), N for flow/diffusion models."""
        ...

    @abstractmethod
    def init_pipeline(self, cfg: Any) -> Any:
        """Initialize and return the model pipeline from config."""
        ...

    # ---- Subclass MAY implement ----

    def get_language_model(self) -> Optional[nn.Module]:
        """Return VLM backbone if present. None for pure vision-action models (ACT)."""
        return None

    def get_layer_blocks(self) -> List[nn.Module]:
        """Return transformer layer blocks for analysis hooks. Empty if N/A."""
        return []

    def has_context_phase(self) -> bool:
        """Whether this VLA has a context encoding phase (VLM prefill).
        False for pure vision-action models like ACT."""
        return self.get_language_model() is not None

    # ---- Layer resolution ----

    def _resolve_store_layers(self) -> None:
        """Resolve store_layers sentinel -1 to all layer indices."""
        if self.pipeline is None:
            return

        try:
            layer_blocks = self.get_layer_blocks()
        except (AttributeError, TypeError):
            return

        num_layers = len(layer_blocks)
        if not num_layers:
            return

        if -1 in self.store_layers:
            self.store_layers = list(range(num_layers))
        else:
            self.store_layers = [
                idx if idx >= 0 else num_layers + idx
                for idx in self.store_layers
            ]

    # ---- Profiling hooks: E/C/A ----

    def register_profiling_hooks(self) -> None:
        """
        Register timing hooks for E/C/A phases.

        E: vision encoder pre/post hooks
        C: language model first/last layer hooks (if VLM backbone exists)
        A: action head pre/post hooks (accumulates across denoise steps)
        """
        timer = self.timer
        vision_encoder = self.get_vision_encoder()
        action_head = self.get_action_head()

        # --- E: Vision encoder timing ---
        def _encode_pre(module: nn.Module, inputs: Any) -> None:
            timer.mark_start("encode")

        def _encode_post(module: nn.Module, inputs: Any, output: Any) -> None:
            timer.mark_end("encode")

        self.analysis_hooks["profiling:encode_pre"] = (
            vision_encoder.register_forward_pre_hook(_encode_pre)
        )
        self.analysis_hooks["profiling:encode_post"] = (
            vision_encoder.register_forward_hook(_encode_post)
        )

        # --- C: Context encoding timing (optional) ---
        if self.has_context_phase():
            lm = self.get_language_model()
            layer_blocks = self.get_layer_blocks()

            if layer_blocks:
                first_layer = layer_blocks[0]
                last_layer = layer_blocks[-1]

                def _context_pre(module: nn.Module, inputs: Any) -> None:
                    timer.mark_start("context")

                def _context_post(module: nn.Module, inputs: Any, output: Any) -> None:
                    timer.mark_end("context")

                self.analysis_hooks["profiling:context_pre"] = (
                    first_layer.register_forward_pre_hook(_context_pre)
                )
                self.analysis_hooks["profiling:context_post"] = (
                    last_layer.register_forward_hook(_context_post)
                )

        # --- A: Action head timing (accumulates across denoise steps) ---
        def _action_pre(module: nn.Module, inputs: Any) -> None:
            timer.mark_start("action")

        def _action_post(module: nn.Module, inputs: Any, output: Any) -> None:
            timer.mark_end("action")

        self.analysis_hooks["profiling:action_pre"] = (
            action_head.register_forward_pre_hook(_action_pre)
        )
        self.analysis_hooks["profiling:action_post"] = (
            action_head.register_forward_hook(_action_post)
        )

        denoise_steps = self.get_denoise_steps()
        has_ctx = "yes" if self.has_context_phase() else "no"
        self.logger.info(
            "Registered VLA profiling hooks (context=%s, expected_denoise_steps=%d)",
            has_ctx,
            denoise_steps,
        )

    # ---- Analysis hooks ----

    def register_analysis_hooks(self) -> None:
        """Register analysis hooks on configured store_layers."""
        layer_blocks = self.get_layer_blocks()
        for layer_idx in self.store_layers:
            if 0 <= layer_idx < len(layer_blocks):
                block = layer_blocks[layer_idx]
                self._register_layer_analysis_hooks(block, layer_idx)

        self.logger.info(
            "Registered analysis hooks on %d layers", len(self.store_layers)
        )

    def _register_layer_analysis_hooks(
        self, block: nn.Module, layer_idx: int
    ) -> None:
        """Register layer-specific analysis hooks. Override in subclass."""

    # ---- Postprocess ----

    def postprocess(self, results: Any = None) -> Any:
        """Flush step_store into global_store after inference."""
        self.flush_step_store()
        return results

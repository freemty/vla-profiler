"""
Base VLM controller extending probe_core.BaseController.

Provides VLM-specific phase timing (encode/prefill/decode), hook
registration patterns for vision encoders and language model layers,
and an abstract interface for model-specific implementations.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

import torch.nn as nn

from src.core.probe_core import BaseController, HookManager
from src.utils.timing import PhaseTimer


logger = logging.getLogger(__name__)


class BaseVLMController(BaseController):
    """
    Abstract VLM controller with encode/prefill/decode phase awareness.

    Subclasses must implement:
        - get_vision_encoder() -> nn.Module
        - get_language_model() -> nn.Module
        - get_layer_blocks() -> List[nn.Module]
        - init_pipeline(cfg) -> Any
        - prepare_inputs(cfg) -> list
        - model_inference(pipeline, cfg, inputs) -> Any

    Subclasses may override:
        - _register_layer_analysis_hooks(block, layer_idx)
    """

    PHASES = ("encode", "prefill", "decode")

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
        self._aggregated_timing: Optional[Dict[str, Any]] = None
        self._resolve_store_layers()

    # ------------------------------------------------------------------
    # Abstract methods — subclass MUST implement
    # ------------------------------------------------------------------

    @abstractmethod
    def get_vision_encoder(self) -> nn.Module:
        """Return the vision encoder module."""
        ...

    @abstractmethod
    def get_language_model(self) -> nn.Module:
        """Return the language model backbone (without the head)."""
        ...

    @abstractmethod
    def get_layer_blocks(self) -> List[nn.Module]:
        """Return an ordered list of transformer layer blocks."""
        ...

    @abstractmethod
    def init_pipeline(self, cfg: Any) -> Any:
        """Initialize and return the model pipeline from config."""
        ...

    # ------------------------------------------------------------------
    # Layer resolution
    # ------------------------------------------------------------------

    def _resolve_store_layers(self) -> None:
        """
        Resolve store_layers sentinel -1 to all layer indices.

        Called during __init__ and can be called again after pipeline
        initialization when layer count becomes known.
        """
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
            # Resolve negative indices
            self.store_layers = [
                idx if idx >= 0 else num_layers + idx
                for idx in self.store_layers
            ]

    # ------------------------------------------------------------------
    # Profiling hooks
    # ------------------------------------------------------------------

    def register_profiling_hooks(self) -> None:
        """
        Register timing hooks on vision encoder and LLM layers.

        Vision encoder: pre/post hooks for encode phase timing.
        LLM first/last layer: hooks for prefill/decode phase timing,
        distinguished by seq_len > 1 (prefill) vs seq_len == 1 (decode).
        """
        timer = self.timer
        vision_encoder = self.get_vision_encoder()
        layer_blocks = self.get_layer_blocks()

        if not layer_blocks:
            self.logger.warning("No layer blocks found; skipping LLM timing hooks")
            return

        first_layer = layer_blocks[0]
        last_layer = layer_blocks[-1]

        # --- Vision encoder timing ---
        def _encode_pre_hook(module: nn.Module, inputs: Any) -> None:
            timer.mark_start("encode")

        def _encode_post_hook(module: nn.Module, inputs: Any, output: Any) -> None:
            timer.mark_end("encode")

        self.analysis_hooks["profiling:encode_pre"] = (
            vision_encoder.register_forward_pre_hook(_encode_pre_hook)
        )
        self.analysis_hooks["profiling:encode_post"] = (
            vision_encoder.register_forward_hook(_encode_post_hook)
        )

        # --- LLM phase timing ---
        def _llm_first_pre_hook(module: nn.Module, inputs: Any) -> None:
            """Start prefill or decode timer based on sequence length."""
            hidden = inputs[0] if isinstance(inputs, tuple) else inputs
            seq_len = hidden.shape[1] if hidden.dim() >= 2 else 1
            phase = "prefill" if seq_len > 1 else "decode"
            timer.mark_start(phase)

        def _llm_last_post_hook(module: nn.Module, inputs: Any, output: Any) -> None:
            """End the currently active phase timer."""
            hidden = inputs[0] if isinstance(inputs, tuple) else inputs
            seq_len = hidden.shape[1] if hidden.dim() >= 2 else 1
            phase = "prefill" if seq_len > 1 else "decode"
            timer.mark_end(phase)

        self.analysis_hooks["profiling:llm_first_pre"] = (
            first_layer.register_forward_pre_hook(_llm_first_pre_hook)
        )
        self.analysis_hooks["profiling:llm_last_post"] = (
            last_layer.register_forward_hook(_llm_last_post_hook)
        )

        self.logger.info(
            "Registered profiling hooks on vision encoder + LLM layers [0, %d]",
            len(layer_blocks) - 1,
        )

    # ------------------------------------------------------------------
    # Analysis hooks
    # ------------------------------------------------------------------

    def register_analysis_hooks(self) -> None:
        """
        Register analysis hooks on configured store_layers.

        Iterates store_layers, calls _register_layer_analysis_hooks
        for each layer block.
        """
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
        """
        Register layer-specific analysis hooks. Override in subclass.

        Default is a no-op.
        """

    def _register_capture_hook(
        self,
        proj_module: nn.Module,
        layer_idx: int,
        qkv_type: str,
    ) -> None:
        """Register a forward hook to capture Q/K/V projection output.

        Shared by all VLM controllers — avoids duplicating the closure
        in each subclass. Uses process_log_dict for step_store insertion.
        """
        store_key = f"{layer_idx}_{qkv_type}_states"
        controller = self

        def _capture(module: nn.Module, inputs: Any, output: Any) -> Any:
            if controller.should_store(store_key):
                controller.process_log_dict(
                    {f"{qkv_type}_states": output.detach().to("cpu", non_blocking=True)},
                    layer_idx=layer_idx,
                )
            return output

        hook = proj_module.register_forward_hook(_capture)
        self.analysis_hooks[f"analysis:layer{layer_idx}_{qkv_type}"] = hook

    # ------------------------------------------------------------------
    # Postprocess
    # ------------------------------------------------------------------

    def postprocess(self, results: Any = None) -> Any:
        """Flush step_store into global_store after inference."""
        self.flush_step_store()
        return results

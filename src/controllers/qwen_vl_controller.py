"""
QwenVL controller for Qwen2.5-VL model profiling and analysis.

Implements model-specific hooks and inference pipeline for the
Qwen2.5-VL family of vision-language models.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from src.controllers import CONTROLLER_REGISTRY
from src.controllers.base_vlm_controller import BaseVLMController
from src.core.probe_core import HookManager


logger = logging.getLogger(__name__)


class QwenVLController(BaseVLMController):
    """
    Hook controller for Qwen2.5-VL vision-language model.

    Supports profiling (encode/prefill/decode timing) and analysis
    (QKV attention state capture) modes.
    """

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

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def get_vision_encoder(self) -> nn.Module:
        """Return the Qwen2.5-VL vision encoder (ViT)."""
        return self.pipeline.model.visual

    def get_language_model(self) -> nn.Module:
        """Return the Qwen2.5-VL language model backbone."""
        return self.pipeline.model.model

    def get_layer_blocks(self) -> List[nn.Module]:
        """Return ordered list of LLM transformer layers."""
        return list(self.pipeline.model.model.layers)

    @staticmethod
    def init_pipeline(cfg: Any) -> Any:
        """
        Initialize Qwen2.5-VL pipeline from config.

        Loads model with bfloat16 precision and flash_attention_2.
        Returns an edict with model + processor.
        """
        from easydict import EasyDict as edict
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
            model_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

        return edict(model=model, processor=processor)

    def prepare_inputs(self, cfg: Any) -> List[Dict[str, Any]]:
        """
        Prepare input list from config.

        Each input entry has 'name' and 'messages' keys.
        Reads from cfg.inputs which is a list of {name, messages} dicts.
        """
        raw_inputs = getattr(cfg, "inputs", [])
        inputs = []

        for entry in raw_inputs:
            name = entry.get("name", "unnamed")
            messages = entry.get("messages", [])
            inputs = [
                *inputs,
                {"name": name, "messages": messages},
            ]

        return inputs

    @torch.no_grad()
    def model_inference(
        self, pipeline: Any, cfg: Any, inputs: Dict[str, Any]
    ) -> List[str]:
        """
        Run Qwen2.5-VL inference on a single input.

        Applies chat template, processes vision info, and generates text.
        """
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            from src.utils.qwen_vl_utils import process_vision_info

        processor = pipeline.processor
        model = pipeline.model
        messages = inputs.get("messages", [])

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        model_inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        model_inputs = model_inputs.to(model.device)

        max_new_tokens = getattr(cfg, "max_new_tokens", 256)
        generated_ids = model.generate(
            **model_inputs, max_new_tokens=max_new_tokens
        )

        # Trim input tokens from output
        trimmed_ids = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_text

    def save_results(
        self,
        inputs: Dict[str, Any],
        results: List[str],
        cfg: Any,
    ) -> str:
        """Save inference results to JSON file."""
        name = inputs.get("name", "unnamed")
        save_dir = os.path.join(cfg.output_path, name)
        os.makedirs(save_dir, exist_ok=True)

        output_data = {
            "name": name,
            "messages": inputs.get("messages", []),
            "output_text": results,
        }

        save_path = os.path.join(save_dir, f"{name}_result.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        self.logger.info("Results saved to %s", save_path)
        return save_dir

    # ------------------------------------------------------------------
    # Analysis hooks: QKV capture
    # ------------------------------------------------------------------

    def _register_layer_analysis_hooks(
        self, block: nn.Module, layer_idx: int
    ) -> None:
        """
        Register QKV capture hooks on self_attn for a given layer.

        Captures Q and K projections based on store_type config.
        """
        self_attn = block.self_attn

        if "self_q" in self.store_type:
            self._register_qk_hook(
                self_attn.q_proj, layer_idx, "q", "self_q"
            )

        if "self_k" in self.store_type:
            self._register_qk_hook(
                self_attn.k_proj, layer_idx, "k", "self_k"
            )

        if "self_v" in self.store_type:
            self._register_qk_hook(
                self_attn.v_proj, layer_idx, "v", "self_v"
            )

    def _register_qk_hook(
        self,
        proj_module: nn.Module,
        layer_idx: int,
        qkv_type: str,
        store_type_key: str,
    ) -> None:
        """Register a forward hook to capture Q/K/V projection output."""
        store_key = f"{layer_idx}_{qkv_type}_states"
        controller = self

        def _capture_hook(
            module: nn.Module, inputs: Any, output: torch.Tensor
        ) -> torch.Tensor:
            if controller.should_store(store_key):
                detached = output.detach().to("cpu", non_blocking=True)
                if store_key not in controller.step_store:
                    controller.step_store[store_key] = [detached]
                else:
                    controller.step_store[store_key] = [
                        *controller.step_store[store_key],
                        detached,
                    ]
            return output

        hook = proj_module.register_forward_hook(_capture_hook)
        hook_key = f"analysis:layer{layer_idx}_{qkv_type}"
        self.analysis_hooks[hook_key] = hook


CONTROLLER_REGISTRY.register("qwen_vl", QwenVLController)

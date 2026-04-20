"""
LingBot-VLA controller for profiling and analysis.

LingBot-VLA is a 4B autoregressive VLA based on Qwen2.5-VL-3B-Instruct.
Actions are tokenized and generated autoregressively (up to 75 action dims).
E/P/D phases apply directly — identical to VLM profiling.
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


logger = logging.getLogger(__name__)


class LingBotVLAController(BaseVLMController):
    """
    Hook controller for LingBot-VLA (4B, Qwen2.5-VL-3B based).

    Architecture: Qwen2VisionTransformer (visual) -> Qwen2VLModel (LLM)
    Action output: up to 75 discrete action tokens (autoregressive)
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

    # ---- Abstract method implementations ----

    def get_vision_encoder(self) -> nn.Module:
        """Return the Qwen2 vision transformer (model.visual)."""
        return self.pipeline.model.visual

    def get_language_model(self) -> nn.Module:
        """Return the Qwen2VL language model backbone (model.model)."""
        return self.pipeline.model.model

    def get_layer_blocks(self) -> List[nn.Module]:
        """Return Qwen2VL transformer layers."""
        return list(self.pipeline.model.model.layers)

    @staticmethod
    def init_pipeline(cfg: Any) -> Any:
        """
        Load LingBot-VLA via AutoModelForCausalLM with Qwen2.5-VL architecture.

        Uses flash_attention_2 for efficient inference.
        """
        from easydict import EasyDict as edict
        from transformers import AutoModelForCausalLM, AutoProcessor

        model_path = cfg.model_name

        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto",
        )

        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        return edict(model=model, processor=processor)

    def prepare_inputs(self, cfg: Any) -> List[Dict[str, Any]]:
        """
        Prepare input list from config.

        LingBot-VLA inputs: image + instruction text in Qwen chat format.
        """
        from omegaconf import OmegaConf

        raw_inputs = getattr(cfg, "inputs", [])
        inputs = []

        for entry in raw_inputs:
            if hasattr(entry, "_iter_ex"):
                entry = OmegaConf.to_container(entry, resolve=True)
            name = entry.get("name", "unnamed")
            messages = entry.get("messages", [])
            if isinstance(messages, list) and messages and hasattr(messages[0], "_iter_ex"):
                messages = [OmegaConf.to_container(m, resolve=True) for m in messages]
            inputs = [*inputs, {"name": name, "messages": messages}]

        return inputs

    @torch.no_grad()
    def model_inference(
        self, pipeline: Any, cfg: Any, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run LingBot-VLA inference.

        Processes image + instruction via Qwen2.5-VL processor,
        generates action tokens autoregressively.
        """
        from io import BytesIO

        import requests
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        processor = pipeline.processor
        model = pipeline.model
        messages = inputs.get("messages", [])

        # Build Qwen chat messages with image + text
        chat_messages = _build_chat_messages(messages)

        # Apply chat template
        text = processor.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True
        )

        # Process vision info (handles image loading/resizing)
        image_inputs, video_inputs = process_vision_info(chat_messages)

        # Tokenize
        model_inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        # Generate action tokens
        max_new_tokens = getattr(cfg, "max_new_tokens", 256)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        # Decode output
        input_len = model_inputs["input_ids"].shape[1]
        action_token_ids = generated_ids[0, input_len:]
        decoded_text = processor.batch_decode(
            generated_ids[:, input_len:],
            skip_special_tokens=True,
        )

        return {
            "action_token_ids": action_token_ids.cpu().tolist(),
            "decoded_text": decoded_text,
            "input_len": input_len,
            "output_len": len(action_token_ids),
        }

    def save_results(
        self,
        inputs: Dict[str, Any],
        results: Dict[str, Any],
        cfg: Any,
    ) -> str:
        """Save inference results to JSON."""
        name = inputs.get("name", "unnamed")
        base_output = getattr(cfg, "output_path", getattr(cfg, "base_output_path", "./output"))
        save_dir = os.path.join(base_output, name)
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"{name}_result.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "name": name,
                    "action_token_ids": results.get("action_token_ids", []),
                    "decoded_text": results.get("decoded_text", []),
                    "input_len": results.get("input_len", 0),
                    "output_len": results.get("output_len", 0),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        self.logger.info("Results saved to %s", save_path)
        return save_dir

    # ---- Analysis hooks ----

    def _register_layer_analysis_hooks(
        self, block: nn.Module, layer_idx: int
    ) -> None:
        """Register QKV capture hooks on Qwen2VL self-attention layers."""
        self_attn = block.self_attn

        if "self_q" in self.store_type:
            self._register_capture_hook(self_attn.q_proj, layer_idx, "q")

        if "self_k" in self.store_type:
            self._register_capture_hook(self_attn.k_proj, layer_idx, "k")

        if "self_v" in self.store_type:
            self._register_capture_hook(self_attn.v_proj, layer_idx, "v")


def _build_chat_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert config messages to Qwen chat format."""
    chat_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content_items = msg.get("content", [])
        qwen_content = []
        for item in content_items:
            item_type = item.get("type", "")
            if item_type == "image":
                qwen_content = [*qwen_content, {"type": "image", "image": item.get("image", "")}]
            elif item_type == "text":
                qwen_content = [*qwen_content, {"type": "text", "text": item.get("text", "")}]
        chat_messages = [*chat_messages, {"role": role, "content": qwen_content}]
    return chat_messages


CONTROLLER_REGISTRY.register("lingbot_vla", LingBotVLAController)

"""
OpenVLA controller for profiling and analysis.

OpenVLA is an autoregressive VLA based on Prismatic VLM (Llama-2 7B + DINOv2 + SigLIP).
Actions are tokenized as 256 discrete bins, 7 tokens for 7-DoF.
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


class OpenVLAController(BaseVLMController):
    """
    Hook controller for OpenVLA (openvla-7b).

    Architecture: DINOv2+SigLIP vision backbone -> MLP projector -> Llama-2 7B
    Action output: 7 discrete tokens (256 bins each) for 7-DoF control
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
        """Return the Prismatic dual vision backbone (DINOv2 + SigLIP)."""
        return self.pipeline.model.vision_backbone

    def get_language_model(self) -> nn.Module:
        """Return the Llama-2 language model."""
        return self.pipeline.model.llm_backbone.llm

    def get_layer_blocks(self) -> List[nn.Module]:
        """Return Llama-2 transformer layers (32 layers for 7B)."""
        return list(self.pipeline.model.llm_backbone.llm.model.layers)

    @staticmethod
    def init_pipeline(cfg: Any) -> Any:
        """
        Load OpenVLA via AutoModelForCausalLM with trust_remote_code.

        Uses sdpa attention (flash_attention_2 may not be installed).
        """
        from easydict import EasyDict as edict
        from transformers import AutoModelForCausalLM, AutoProcessor

        model_path = cfg.model_name

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
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

        OpenVLA inputs: image (PIL or path) + instruction text.
        Config format same as Qwen — messages with role/content.
        """
        from omegaconf import OmegaConf

        raw_inputs = getattr(cfg, "inputs", [])
        inputs = []

        for entry in raw_inputs:
            if hasattr(entry, '_iter_ex'):
                entry = OmegaConf.to_container(entry, resolve=True)
            name = entry.get("name", "unnamed")
            messages = entry.get("messages", [])
            if isinstance(messages, list) and messages and hasattr(messages[0], '_iter_ex'):
                messages = [OmegaConf.to_container(m, resolve=True) for m in messages]
            inputs = [*inputs, {"name": name, "messages": messages}]

        return inputs

    @torch.no_grad()
    def model_inference(
        self, pipeline: Any, cfg: Any, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run OpenVLA inference.

        Processes image + instruction, generates action tokens (7 for 7-DoF).
        """
        from PIL import Image
        import requests
        from io import BytesIO

        processor = pipeline.processor
        model = pipeline.model
        messages = inputs.get("messages", [])

        # Extract image and text from messages
        image = None
        instruction = ""
        for msg in messages:
            if msg.get("role") != "user":
                continue
            for content in msg.get("content", []):
                if content.get("type") == "image":
                    img_path = content.get("image", "")
                    if img_path.startswith("http"):
                        response = requests.get(img_path, timeout=10)
                        image = Image.open(BytesIO(response.content)).convert("RGB")
                    else:
                        image = Image.open(img_path).convert("RGB")
                elif content.get("type") == "text":
                    instruction = content.get("text", "")

        if image is None:
            raise ValueError("OpenVLA requires an image input")

        # Build prompt (OpenVLA format)
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"

        # Process inputs
        model_inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)

        # Generate action tokens
        max_new_tokens = getattr(cfg, "max_new_tokens", 512)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        # Decode
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
            json.dump({
                "name": name,
                "action_token_ids": results.get("action_token_ids", []),
                "decoded_text": results.get("decoded_text", []),
                "input_len": results.get("input_len", 0),
                "output_len": results.get("output_len", 0),
            }, f, ensure_ascii=False, indent=2)

        self.logger.info("Results saved to %s", save_path)
        return save_dir

    # ---- Analysis hooks ----

    def _register_layer_analysis_hooks(
        self, block: nn.Module, layer_idx: int
    ) -> None:
        """Register QKV capture hooks on Llama-2 self-attention layers."""
        self_attn = block.self_attn

        if "self_q" in self.store_type:
            self._register_proj_hook(self_attn.q_proj, layer_idx, "q")

        if "self_k" in self.store_type:
            self._register_proj_hook(self_attn.k_proj, layer_idx, "k")

        if "self_v" in self.store_type:
            self._register_proj_hook(self_attn.v_proj, layer_idx, "v")

    def _register_proj_hook(
        self, proj_module: nn.Module, layer_idx: int, qkv_type: str
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
        self.analysis_hooks[f"analysis:layer{layer_idx}_{qkv_type}"] = hook


CONTROLLER_REGISTRY.register("openvla", OpenVLAController)

"""
StarVLA-OFT controller for E/C/A profiling.

StarVLA is a modular VLA training framework (HKUST). StarVLA-OFT uses
Qwen3-VL-4B backbone + parallel MLP action head (OFT recipe from OpenVLA-OFT).

The paper reports 96.6% LIBERO accuracy but zero inference latency numbers.
This controller fills that gap.

Architecture: Qwen3-VL-4B (ViT + Qwen3 LLM) → parallel MLP action head
Phase model: E/C/A
  - E: Qwen3 ViT vision encoder
  - C: Qwen3 LLM prefill
  - A: Parallel MLP forward (single pass, no denoise)

Paper: arXiv:2604.05014
Code: github.com/StarVLA/StarVLA
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from src.controllers import CONTROLLER_REGISTRY
from src.controllers.base_vla_controller import BaseVLAController
from src.controllers.openvla_oft_controller import _get_or_create_oft_head


logger = logging.getLogger(__name__)


class StarVLAOFTController(BaseVLAController):
    """
    Hook controller for StarVLA-OFT.

    Architecture: Qwen3-VL-4B (ViT + LLM) + parallel MLP action head (OFT).
    Single forward through MLP → action chunk. No denoise loop.
    """

    def register_profiling_hooks(self) -> None:
        self.logger.info(
            "StarVLA-OFT: skipping hook-based profiling — using manual E/C/A "
            "timing in model_inference"
        )

    def get_vision_encoder(self) -> nn.Module:
        return self.pipeline.vision_encoder

    def get_action_head(self) -> nn.Module:
        return self.pipeline.action_head

    def get_denoise_steps(self) -> int:
        return 1

    def get_language_model(self) -> Optional[nn.Module]:
        return self.pipeline.language_model

    def get_layer_blocks(self) -> List[nn.Module]:
        lm = self.get_language_model()
        if lm is None:
            return []
        if hasattr(lm, "layers"):
            return list(lm.layers)
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return list(lm.model.layers)
        return []

    def has_context_phase(self) -> bool:
        return True

    @staticmethod
    def init_pipeline(cfg: Any) -> Any:
        """
        Initialize StarVLA-OFT.

        Two modes:
        1. StarVLA repo available: load via StarVLA's modular API
        2. Fallback: load Qwen3-VL-4B directly + random OFT head (timing only)
        """
        from easydict import EasyDict as edict

        device = getattr(cfg, "device", "cuda:0")
        action_dim = getattr(cfg, "action_dim", 7)
        chunk_size = getattr(cfg, "chunk_size", 5)
        model_path = getattr(cfg, "model_name", "")
        backbone_path = getattr(cfg, "backbone_path", "Qwen/Qwen3-VL-4B")

        # Try StarVLA import first
        starvla_available = _try_import_starvla(cfg)

        if starvla_available:
            return _init_via_starvla(
                cfg, model_path, backbone_path, device, action_dim, chunk_size,
            )

        return _init_via_qwen3(
            backbone_path, device, action_dim, chunk_size,
        )

    def prepare_inputs(self, cfg: Any) -> List[Dict[str, Any]]:
        from src.utils import to_plain

        raw_inputs = getattr(cfg, "inputs", [])
        inputs = []
        device = self.pipeline.device

        for entry in raw_inputs:
            entry = to_plain(entry)
            name = entry.get("name", "unnamed")
            image_shape = entry.get("image_shape", [3, 224, 224])

            # Qwen2.5-VL ViT expects pre-patchified tensors + grid_thw
            patch_size = 14
            temporal = 2
            img_h, img_w = image_shape[1], image_shape[2]
            grid_h = img_h // patch_size
            grid_w = img_w // patch_size
            num_patches = grid_h * grid_w
            patch_dim = image_shape[0] * temporal * patch_size * patch_size

            pixel_values = torch.randn(
                num_patches, patch_dim, device=device, dtype=torch.bfloat16,
            )
            grid_thw = torch.tensor(
                [[1, grid_h, grid_w]], device=device, dtype=torch.long,
            )

            seq_len = entry.get("seq_len", 50)
            input_ids = torch.randint(
                1, 152064, (1, seq_len), device=device, dtype=torch.long,
            )

            inputs = [*inputs, {
                "name": name,
                "pixel_values": pixel_values,
                "grid_thw": grid_thw,
                "input_ids": input_ids,
            }]

        return inputs

    @torch.no_grad()
    def model_inference(
        self, pipeline: Any, cfg: Any, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run StarVLA-OFT inference with manual E/C/A phase timing.

        E: Qwen ViT vision encoder (pre-patchified input)
        C: Qwen LLM prefill
        A: Parallel MLP action head (single forward)
        """
        vision_encoder = pipeline.vision_encoder
        language_model = pipeline.language_model
        action_head = pipeline.action_head
        embed_tokens = pipeline.embed_tokens
        device = pipeline.device

        pixel_values = inputs["pixel_values"]
        grid_thw = inputs["grid_thw"]
        input_ids = inputs["input_ids"]

        # --- Phase E: Vision encoder ---
        self.timer.mark_start("encode")
        torch.cuda.synchronize()

        visual_features = vision_encoder(pixel_values, grid_thw=grid_thw)

        torch.cuda.synchronize()
        self.timer.mark_end("encode")

        # --- Phase C: LLM prefill ---
        self.timer.mark_start("context")
        torch.cuda.synchronize()

        text_embeds = embed_tokens(input_ids)
        # visual_features shape: (num_patches, hidden) → (1, num_patches, hidden)
        if visual_features.dim() == 2:
            visual_features = visual_features.unsqueeze(0)
        combined_embeds = torch.cat([visual_features, text_embeds], dim=1)
        combined_mask = torch.ones(
            combined_embeds.shape[:2], device=device, dtype=torch.long,
        )

        llm_output = language_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
        )
        hidden_states = llm_output.last_hidden_state

        torch.cuda.synchronize()
        self.timer.mark_end("context")

        # --- Phase A: OFT parallel MLP action head ---
        self.timer.mark_start("action")
        torch.cuda.synchronize()

        last_hidden = hidden_states[:, -1, :]
        actions = action_head(last_hidden)

        torch.cuda.synchronize()
        self.timer.mark_end("action")

        return {
            "actions": actions,
            "action_shape": list(actions.shape),
            "denoise_steps": 1,
        }

    def save_results(
        self,
        inputs: Dict[str, Any],
        results: Dict[str, Any],
        cfg: Any,
    ) -> str:
        name = inputs.get("name", "unnamed")
        base_output = getattr(
            cfg, "output_path", getattr(cfg, "base_output_path", "./output")
        )
        save_dir = os.path.join(base_output, name)
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"{name}_result.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({
                "name": name,
                "action_shape": results.get("action_shape", []),
                "denoise_steps": results.get("denoise_steps", 0),
            }, f, ensure_ascii=False, indent=2)

        self.logger.info("Results saved to %s", save_path)
        return save_dir


def _try_import_starvla(cfg: Any) -> bool:
    """Check if StarVLA repo is available."""
    vendor_path = getattr(cfg, "starvla_path", None)
    if vendor_path and os.path.isdir(vendor_path):
        abs_path = os.path.abspath(vendor_path)
        if abs_path not in sys.path:
            sys.path.insert(0, abs_path)

    try:
        import starvla  # noqa: F401
        return True
    except ImportError:
        return False


def _init_via_starvla(
    cfg: Any,
    model_path: str,
    backbone_path: str,
    device: str,
    action_dim: int,
    chunk_size: int,
) -> Any:
    """Initialize via StarVLA's modular API."""
    from easydict import EasyDict as edict

    try:
        from starvla.models import build_model
        model = build_model(
            backbone=backbone_path,
            action_head="oft",
            action_dim=action_dim,
            chunk_size=chunk_size,
        )
        if model_path:
            state_dict = torch.load(model_path, weights_only=False, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]
            model.load_state_dict(state_dict, strict=False)

        model = model.to(device, dtype=torch.bfloat16)
        model.eval()

        vision_encoder = model.backbone.visual if hasattr(model.backbone, "visual") else model.backbone
        language_model = model.backbone.model if hasattr(model.backbone, "model") else model.backbone
        action_head = model.action_head
        embed_tokens = language_model.embed_tokens if hasattr(language_model, "embed_tokens") else language_model.model.embed_tokens

        hidden_size = action_head.in_features if hasattr(action_head, "in_features") else 2048

        logger.info("StarVLA loaded via StarVLA API")
        return edict(
            model=model,
            vision_encoder=vision_encoder,
            language_model=language_model,
            action_head=action_head,
            embed_tokens=embed_tokens,
            visual_projector=getattr(model, "projector", None),
            hidden_size=hidden_size,
            device=device,
        )
    except (ImportError, FileNotFoundError, OSError) as e:
        logger.warning("StarVLA API init failed (%s), falling back to Qwen3", e)
        return _init_via_qwen3(backbone_path, device, action_dim, chunk_size)


def _init_via_qwen3(
    backbone_path: str,
    device: str,
    action_dim: int,
    chunk_size: int,
) -> Any:
    """Fallback: load Qwen3-VL directly + random OFT head."""
    from easydict import EasyDict as edict
    from transformers import Qwen2_5_VLForConditionalGeneration

    logger.info("Loading %s directly (StarVLA not available)", backbone_path)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        backbone_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)
    model.eval()

    hidden_size = model.config.hidden_size
    action_head = _get_or_create_oft_head(
        model, hidden_size, action_dim, chunk_size, device,
    )

    lm = model.model
    if hasattr(lm, "language_model"):
        lm = lm.language_model

    return edict(
        model=model,
        vision_encoder=model.visual,
        language_model=lm,
        action_head=action_head,
        embed_tokens=lm.embed_tokens,
        visual_projector=None,
        hidden_size=hidden_size,
        device=device,
    )


CONTROLLER_REGISTRY.register("starvla_oft", StarVLAOFTController)

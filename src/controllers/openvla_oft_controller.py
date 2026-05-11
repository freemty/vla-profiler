"""
OpenVLA-OFT controller for E/C/A profiling.

OpenVLA-OFT replaces OpenVLA's AR action decode with a parallel MLP head
(OFT = Optimized Fine-Tuning). This makes it a VLA, not a VLM — one forward
pass through the MLP produces the entire action chunk (no decode loop).

Architecture: Prismatic VLM (DINOv2 + SigLIP → MLP projector → Llama-2 7B)
              → parallel MLP action head (L1 regression)
Phase model: E/C/A
  - E: Dual vision encoder (DINOv2 + SigLIP)
  - C: Llama-2 7B prefill (text + visual tokens)
  - A: Parallel MLP forward (single pass, no denoise)

Paper: arXiv:2502.19645
Code: github.com/openvla/openvla (OFT branch)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from src.controllers import CONTROLLER_REGISTRY
from src.controllers.base_vla_controller import BaseVLAController


logger = logging.getLogger(__name__)


class OpenVLAOFTController(BaseVLAController):
    """
    Hook controller for OpenVLA-OFT.

    Architecture: Prismatic (DINOv2+SigLIP → Llama-2 7B) + parallel MLP action head.
    Single forward through MLP → action chunk. No denoise loop.
    """

    def register_profiling_hooks(self) -> None:
        self.logger.info(
            "OpenVLA-OFT: skipping hook-based profiling — using manual E/C/A "
            "timing in model_inference"
        )

    def get_vision_encoder(self) -> nn.Module:
        return self.pipeline.model.vision_backbone

    def get_action_head(self) -> nn.Module:
        return self.pipeline.action_head

    def get_denoise_steps(self) -> int:
        return 1

    def get_language_model(self) -> Optional[nn.Module]:
        return self.pipeline.model.language_model

    def get_layer_blocks(self) -> List[nn.Module]:
        lm = self.pipeline.model.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return list(lm.model.layers)
        if hasattr(lm, "layers"):
            return list(lm.layers)
        return []

    def has_context_phase(self) -> bool:
        return True

    @staticmethod
    def init_pipeline(cfg: Any) -> Any:
        """
        Load OpenVLA-OFT model.

        Supports two modes:
        - Full checkpoint with OFT head (model_name points to OFT checkpoint)
        - Random weights for timing (model_name empty or base openvla model)
        """
        from easydict import EasyDict as edict
        from transformers import AutoModelForVision2Seq, AutoProcessor

        model_path = getattr(cfg, "model_name", "") or "openvla/openvla-7b"
        device = getattr(cfg, "device", "cuda:0")
        action_dim = getattr(cfg, "action_dim", 7)
        chunk_size = getattr(cfg, "chunk_size", 5)

        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config._attn_implementation = "eager"

        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        model = model.to(device)
        model.eval()

        # OFT action head: parallel MLP projecting hidden states → action chunk
        # If the loaded model already has an OFT head, use it;
        # otherwise create a random one for timing purposes
        hidden_size = model.config.text_config.hidden_size
        action_head = _get_or_create_oft_head(
            model, hidden_size, action_dim, chunk_size, device,
        )

        logger.info(
            "OpenVLA-OFT initialized: device=%s, hidden=%d, action_dim=%d, "
            "chunk_size=%d",
            device, hidden_size, action_dim, chunk_size,
        )

        return edict(
            model=model,
            processor=processor,
            action_head=action_head,
            device=device,
            hidden_size=hidden_size,
            action_dim=action_dim,
            chunk_size=chunk_size,
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

            # Prismatic dual encoder: DINOv2 (3ch) + SigLIP (3ch) = 6 channels
            pixel_values = torch.randn(
                1, 6, image_shape[1], image_shape[2],
                device=device, dtype=torch.bfloat16,
            )

            # Synthetic text tokens (matching Prismatic's expected input)
            seq_len = entry.get("seq_len", 300)
            input_ids = torch.randint(
                1, 32000, (1, seq_len), device=device, dtype=torch.long,
            )
            attention_mask = torch.ones_like(input_ids)

            inputs = [*inputs, {
                "name": name,
                "pixel_values": pixel_values,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }]

        return inputs

    @torch.no_grad()
    def model_inference(
        self, pipeline: Any, cfg: Any, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run OpenVLA-OFT inference with manual E/C/A phase timing.

        E: Dual vision encoder (DINOv2 + SigLIP)
        C: Llama-2 7B prefill
        A: Parallel MLP action head (single forward)
        """
        model = pipeline.model
        action_head = pipeline.action_head
        device = pipeline.device

        pixel_values = inputs["pixel_values"]
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # --- Phase E: Vision encoder ---
        self.timer.mark_start("encode")
        torch.cuda.synchronize()

        visual_features = model.vision_backbone(pixel_values)
        projected_features = model.projector(visual_features)

        torch.cuda.synchronize()
        self.timer.mark_end("encode")

        # --- Phase C: LLM prefill ---
        self.timer.mark_start("context")
        torch.cuda.synchronize()

        lm = model.language_model
        lm_core = lm.model if hasattr(lm, "model") else lm
        text_embeds = lm_core.embed_tokens(input_ids)
        combined_embeds = torch.cat([projected_features, text_embeds], dim=1)
        combined_mask = torch.ones(
            combined_embeds.shape[:2],
            device=device,
            dtype=attention_mask.dtype,
        )

        llm_output = lm_core(
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


def _get_or_create_oft_head(
    model: nn.Module,
    hidden_size: int,
    action_dim: int,
    chunk_size: int,
    device: str,
) -> nn.Module:
    """
    Return the OFT action head if it exists on the model, otherwise create one.

    OFT head: Linear(hidden_size → action_dim * chunk_size) — single MLP.
    """
    # Check for existing OFT head (various attribute names used in the wild)
    for attr in ("action_head", "oft_head", "action_projection"):
        if hasattr(model, attr):
            head = getattr(model, attr)
            logger.info("Found existing OFT head at model.%s", attr)
            return head.to(device)

    # Create random MLP head for timing
    head = nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.GELU(),
        nn.Linear(hidden_size, action_dim * chunk_size),
    ).to(device, dtype=torch.bfloat16)
    logger.info(
        "Created random OFT head: %d → %d (action_dim=%d × chunk_size=%d)",
        hidden_size, action_dim * chunk_size, action_dim, chunk_size,
    )
    return head


CONTROLLER_REGISTRY.register("openvla_oft", OpenVLAOFTController)

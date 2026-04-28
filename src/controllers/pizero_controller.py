"""
Pi-Zero (π₀) controller via open-pi-zero (allenzren/open-pi-zero).

Architecture: PaliGemma (SigLIP ViT-So400m/14 + Gemma 2B) + Gemma 300M Action Expert
Phase model: E/C/A
  - E: SigLIP vision encoder + multi_modal_projector
  - C: Joint model prefill (VLM + proprio), caches KV
  - A: Flow matching denoise (Action Expert, N Euler steps)

Backend: vendor/open_pi_zero (vendored from github.com/allenzren/open-pi-zero)

IMPORTANT (setup coordination):
Upstream's package is laid out at `vendor/open_pi_zero/src/` and all internal
imports use `from src.model...`. That would shadow *our* project's own `src/`
package when this controller is imported. To avoid that collision, setup_pizero.sh
MUST perform a one-time transformation at install time:
  1. Rename `vendor/open_pi_zero/src` → `vendor/open_pi_zero/pizero_src`
  2. sed-replace `from src.` → `from pizero_src.` in every .py under
     `vendor/open_pi_zero/pizero_src/` (and any sibling scripts that import from it).
After that rename, `from pizero_src.model.vla.pizero import PiZero` works cleanly
without clobbering our own top-level `src` package.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.controllers import CONTROLLER_REGISTRY
from src.controllers.base_vla_controller import BaseVLAController


logger = logging.getLogger(__name__)

VENDOR_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "vendor", "open_pi_zero"
)


def _ensure_vendor_on_path() -> None:
    """
    Ensure the renamed vendored upstream package (`pizero_src`) is importable.

    Expects setup_pizero.sh to have renamed upstream's `src/` → `pizero_src/`
    and rewritten its internal `from src.` imports to `from pizero_src.`.
    We add `vendor/open_pi_zero/` itself to sys.path so that
    `from pizero_src.model.vla.pizero import PiZero` resolves.
    """
    abs_path = os.path.abspath(VENDOR_PATH)
    expected_module = os.path.join(abs_path, "pizero_src", "model", "vla", "pizero.py")
    if not os.path.exists(expected_module):
        raise ImportError(
            f"Vendored pizero_src not found at {abs_path}. "
            "Run scripts/setup_pizero.sh (clones allenzren/open-pi-zero, "
            "renames src/ → pizero_src/, and rewrites internal imports)."
        )
    if abs_path not in sys.path:
        sys.path.insert(0, abs_path)


class PiZeroController(BaseVLAController):
    """
    Hook controller for Pi-Zero via open-pi-zero.

    Dual-stream Transformer: PaliGemma processes image+text,
    Action Expert processes noisy actions. Expert attends to PaliGemma KV
    but not vice versa — enabling KV cache reuse across denoise steps.
    """

    DEFAULT_DENOISE_STEPS = 10

    def register_profiling_hooks(self) -> None:
        self.logger.info(
            "PiZero: skipping hook-based profiling — using manual E/C/A "
            "timing in model_inference (dual-stream architecture requires "
            "explicit phase decomposition)"
        )

    def __init__(
        self,
        model_name: str,
        pipeline: Any = None,
        logger: Optional[logging.Logger] = None,
        store_type: Optional[Union[str, List[str]]] = None,
        store_layers: Optional[Union[int, List[int]]] = None,
        store_phases: Optional[Union[str, List[str]]] = None,
        hook_mode: Optional[str] = "analysis",
        denoise_steps: int = 10,
    ) -> None:
        self._denoise_steps = denoise_steps
        super().__init__(
            model_name=model_name,
            pipeline=pipeline,
            logger=logger,
            store_type=store_type,
            store_layers=store_layers,
            store_phases=store_phases,
            hook_mode=hook_mode,
        )

    def get_vision_encoder(self) -> nn.Module:
        return self.pipeline.model.vision_tower

    def get_action_head(self) -> nn.Module:
        return self.pipeline.model.joint_model.mixtures["action"]

    def get_denoise_steps(self) -> int:
        if self.pipeline is not None:
            return self.pipeline.model.num_inference_steps
        return self._denoise_steps

    def get_language_model(self) -> Optional[nn.Module]:
        return self.pipeline.model.joint_model.mixtures["vlm"]

    def get_layer_blocks(self) -> List[nn.Module]:
        vlm = self.get_language_model()
        if vlm is None:
            return []
        if hasattr(vlm, "layers"):
            return list(vlm.layers)
        return []

    def has_context_phase(self) -> bool:
        return True

    @staticmethod
    def init_pipeline(cfg: Any) -> Any:
        """
        Initialize Pi-Zero via open-pi-zero vendored code.

        Loads model config from YAML, builds PiZero model,
        optionally loads pretrained weights.
        """
        from easydict import EasyDict as edict

        _ensure_vendor_on_path()

        # cuDNN mismatch: torch cu121 vs system cuDNN 9.19 on some hosts.
        # Only disable if cuDNN init actually fails on a probe conv.
        _cudnn_before = torch.backends.cudnn.enabled
        if torch.cuda.is_available():
            try:
                _probe = torch.randn(1, 3, 4, 4, device="cuda")
                torch.nn.functional.conv2d(_probe, torch.randn(1, 3, 3, 3, device="cuda"))
                del _probe
            except RuntimeError:
                torch.backends.cudnn.enabled = False
                logger.warning("cuDNN probe failed, disabling cuDNN for this session")

        from pizero_src.model.vla.pizero import PiZero

        model_path = getattr(cfg, "model_name", "")
        denoise_steps = getattr(cfg, "denoise_steps", 10)
        device = getattr(cfg, "device", "cuda:0")
        use_bf16 = getattr(cfg, "use_bf16", True)

        model_config_path = getattr(cfg, "model_config", None)
        if model_config_path and os.path.exists(model_config_path):
            model_cfg = OmegaConf.load(model_config_path)
        else:
            model_cfg = _default_pizero_config(model_path)
            if not model_path:
                logger.warning(
                    "No model_name specified — using random weights. "
                    "Set model_name to a checkpoint directory for real profiling."
                )

        if denoise_steps:
            model_cfg.num_inference_steps = denoise_steps

        model = PiZero(model_cfg)
        model.tie_action_proprio_weights()

        if model_path:
            if os.path.isfile(model_path) and model_path.endswith(".pt"):
                data = torch.load(model_path, weights_only=False, map_location="cpu")
                state_dict = data["model"] if "model" in data else data
                state_dict = {
                    k.replace("_orig_mod.", ""): v
                    for k, v in state_dict.items()
                }
                model.load_state_dict(state_dict, strict=True)
                logger.info("Loaded .pt checkpoint from %s", model_path)
            elif os.path.isdir(model_path):
                model_cfg.pretrained_model_path = model_path
                model.load_pretrained_weights()
                logger.info("Loaded PaliGemma base weights from %s", model_path)
            else:
                raise ValueError(
                    f"model_name '{model_path}' is neither a .pt file nor a directory."
                )

        dtype = torch.bfloat16 if use_bf16 else torch.float32
        model = model.to(device).to(dtype)
        model.eval()

        logger.info(
            "PiZero initialized: device=%s, dtype=%s, denoise_steps=%d",
            device, dtype, denoise_steps,
        )

        return edict(
            model=model,
            config=model_cfg,
            dtype=dtype,
            device=device,
        )

    def prepare_inputs(self, cfg: Any) -> List[Dict[str, Any]]:
        """
        Prepare synthetic inputs for Pi-Zero profiling.

        Pi-Zero expects: pixel_values, input_ids, attention_mask, proprios.
        For profiling we use random tensors matching expected shapes.
        """
        from src.utils import to_plain

        raw_inputs = getattr(cfg, "inputs", [])
        inputs = []
        device = self.pipeline.device
        dtype = self.pipeline.dtype
        model_cfg = self.pipeline.config

        max_image_text_tokens = model_cfg.max_image_text_tokens
        num_image_tokens = model_cfg.vision.config.num_image_tokens
        cond_steps = model_cfg.cond_steps
        action_dim = model_cfg.action_dim
        proprio_dim = model_cfg.proprio_dim

        image_size = model_cfg.vision.config.image_size

        for entry in raw_inputs:
            entry = to_plain(entry)
            name = entry.get("name", "unnamed")
            image_shape = entry.get("image_shape", [3, 224, 224])

            bsz = 1
            # SigLIP expects fixed image_size; resize to match vision tower
            pixel_values = torch.randn(
                bsz, image_shape[0], image_size, image_size,
                device=device, dtype=dtype,
            )

            text_len = max_image_text_tokens - num_image_tokens
            input_ids = torch.full(
                (bsz, max_image_text_tokens),
                model_cfg.pad_token_id,
                dtype=torch.long,
                device=device,
            )
            input_ids[:, :num_image_tokens] = model_cfg.image_token_index
            input_ids[:, num_image_tokens:num_image_tokens + 5] = 2

            attention_mask = torch.zeros(
                bsz, max_image_text_tokens, dtype=torch.long, device=device
            )
            attention_mask[:, :num_image_tokens + 5] = 1

            proprios = torch.randn(
                bsz, cond_steps, proprio_dim, device=device, dtype=dtype
            )

            inputs = [*inputs, {
                "name": name,
                "pixel_values": pixel_values,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "proprios": proprios,
            }]

        return inputs

    @torch.no_grad()
    def model_inference(
        self, pipeline: Any, cfg: Any, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run Pi-Zero inference with manual E/C/A phase timing.

        Mirrors infer_action() internals but wraps each phase with
        PhaseTimer marks so profiling_task can report per-phase latency.
        """
        model = pipeline.model
        dtype = pipeline.dtype
        device = pipeline.device

        pixel_values = inputs["pixel_values"]
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        proprios = inputs["proprios"]

        causal_mask, vlm_pos_ids, proprio_pos_ids, action_pos_ids = (
            model.build_causal_mask_and_position_ids(attention_mask, dtype=dtype)
        )
        image_text_proprio_mask, action_mask = (
            model.split_full_mask_into_submasks(causal_mask)
        )
        image_text_proprio_mask = image_text_proprio_mask.to(device)
        action_mask = action_mask.to(device)
        vlm_pos_ids = vlm_pos_ids.to(device)
        proprio_pos_ids = proprio_pos_ids.to(device)
        action_pos_ids = action_pos_ids.to(device)

        bsz = pixel_values.size(0)

        # --- Phase E: SigLIP + text embedding + proprio encoder ---
        self.timer.mark_start("encode")
        torch.cuda.synchronize()

        inputs_embeds = model._forward_siglip_and_text_embedding(
            input_ids, pixel_values
        )
        proprio_embeds = model.proprio_encoder(proprios)

        torch.cuda.synchronize()
        self.timer.mark_end("encode")

        # --- Phase C: VLM + proprio joint prefill → KV cache ---
        self.timer.mark_start("context")
        torch.cuda.synchronize()

        kv_caches = model.joint_model.build_mixture_caches()
        _, kv_caches = model.joint_model(
            attention_mask=image_text_proprio_mask,
            position_ids_all={
                "vlm": vlm_pos_ids,
                "proprio": proprio_pos_ids,
            },
            embeds_all={
                "vlm": inputs_embeds,
                "proprio": proprio_embeds,
            },
            kv_caches=kv_caches,
            return_caches=True,
        )

        torch.cuda.synchronize()
        self.timer.mark_end("context")

        # --- Phase A: Action Expert flow denoising × N steps ---
        self.timer.mark_start("action")
        torch.cuda.synchronize()

        action = torch.randn(
            (bsz, model.horizon_steps, model.action_dim),
            device=device, dtype=dtype,
        )
        delta_t = 1.0 / model.num_inference_steps
        t = torch.zeros(bsz, device=device, dtype=dtype)

        for _ in range(model.num_inference_steps):
            time_cond = model.time_embedding(t)
            if model.action_expert_adaptive_mode:
                action_embeds = model.action_encoder(action)
            else:
                action_embeds = model.action_encoder(action, time_cond)

            action_embeds = model.joint_model(
                attention_mask=action_mask,
                position_ids_all={"action": action_pos_ids},
                embeds_all={"action": action_embeds},
                time_cond=time_cond,
                kv_caches=kv_caches,
                cache_mode="append_non_active",
            )["action"]

            action_vel = model.action_decoder(action_embeds)
            action = action + delta_t * action_vel
            t = t + delta_t

        if model.final_action_clip_value is not None:
            action = torch.clamp(
                action,
                -model.final_action_clip_value,
                model.final_action_clip_value,
            )

        torch.cuda.synchronize()
        self.timer.mark_end("action")

        return {
            "actions": action,
            "action_shape": list(action.shape),
            "denoise_steps": model.num_inference_steps,
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


def _default_pizero_config(pretrained_model_path: str = "") -> Any:
    """Default OmegaConf config matching open-pi-zero bridge.yaml defaults."""
    mixture = {
        "vlm": {
            "hidden_size": 2048,
            "intermediate_size": 16384,
            "use_final_norm": False,
            "cache": True,
            "use_quantize": False,
            "use_lora": False,
            "adaptive_mode": None,
            "rope_theta": 10000.0,
        },
        "proprio": {
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "use_final_norm": True,
            "cache": True,
            "use_quantize": False,
            "use_lora": False,
            "adaptive_mode": None,
            "rope_theta": 10000.0,
        },
        "action": {
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "use_final_norm": True,
            "cache": False,
            "use_quantize": False,
            "use_lora": False,
            "adaptive_mode": None,
            "rope_theta": 10000.0,
        },
    }

    cfg = OmegaConf.create({
        "pretrained_model_path": pretrained_model_path,
        "vocab_size": 257216,
        "pad_token_id": 0,
        "image_token_index": 257152,
        "max_image_text_tokens": 276,
        "max_seq_len": 276,
        "cond_steps": 1,
        "horizon_steps": 4,
        "action_dim": 7,
        "proprio_dim": 7,
        "num_inference_steps": 10,
        "final_action_clip_value": 1.0,
        "flow_sig_min": 0.001,
        "use_lm_head": False,
        "action_expert_adaptive_mode": None,
        "time_hidden_size": 256,
        "time_max_period": 10000.0,
        "mixture": mixture,
        "vision": {
            "_target_": "pizero_src.model.paligemma.siglip.SiglipVisionModel",
            "config": {
                "hidden_size": 1152,
                "intermediate_size": 4304,
                "num_hidden_layers": 27,
                "num_attention_heads": 16,
                "num_channels": 3,
                "image_size": 224,
                "patch_size": 14,
                "layer_norm_eps": 1e-6,
                "attention_dropout": 0.0,
                "num_image_tokens": 256,
                "lora": {"r": 32, "dropout": 0.0},
            },
            "use_quantize": False,
            "use_lora": False,
        },
        "quantize": False,
        "lora": False,
        "vision_projector": {
            "_target_": "pizero_src.model.paligemma.siglip.PaliGemmaMultiModalProjector",
            "config": {
                "vision_config": {
                    "hidden_size": 1152,
                    "projection_dim": 2048,
                },
                "lora": {"r": 32, "dropout": 0.0},
            },
            "use_quantize": False,
            "use_lora": False,
        },
        "joint": {
            "_target_": "pizero_src.model.vla.joint_model.JointModel",
            "config": {
                "action_expert_adaptive_mode": None,
                "time_hidden_size": 256,
                "mixture": mixture,
                "lora": {"r": 32, "dropout": 0.0},
                "num_hidden_layers": 18,
                "num_attention_heads": 8,
                "num_key_value_heads": 1,
                "head_dim": 256,
                "rms_norm_eps": 1e-6,
                "attention_bias": False,
                "attention_dropout": 0.0,
                "pad_token_id": 0,
            },
        },
    })
    return cfg


CONTROLLER_REGISTRY.register("pizero", PiZeroController)

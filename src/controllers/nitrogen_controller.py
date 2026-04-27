"""
NitroGen controller for profiling and analysis.

NitroGen (arXiv:2601.02427) is a 500M VA gaming foundation model from NVIDIA/MineDojo.
Architecture: SigLIP ViT (encode) → VL Self-Attention (context) → DiT (flow matching action denoise)

Phases: E/C/A (Encode → VL-Context → Action)
- E: SigLIP 2 ViT — 256x256 single frame → 256 image tokens
- C: VL Self-Attention Transformer — vision token mixing
- A: DiT action head × k denoising steps (flow matching, Euler integration)

Output: 16-step action chunk, 20-dim per step (16 binary buttons + 4 continuous joystick)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from src.controllers import CONTROLLER_REGISTRY
from src.controllers.base_vla_controller import BaseVLAController


logger = logging.getLogger(__name__)


class NitroGenController(BaseVLAController):
    """
    Hook controller for NitroGen (500M VA gaming model).

    Phases: E (SigLIP encode) / C (VL self-attention) / A (DiT flow denoise)
    Uses manual timer marks because the DiT denoise loop needs per-step granularity.
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

    def get_vision_encoder(self) -> nn.Module:
        return self.pipeline.model.vision_encoder

    def get_action_head(self) -> nn.Module:
        return self.pipeline.model.model

    def get_denoise_steps(self) -> int:
        return self.pipeline.model.num_inference_timesteps

    def get_language_model(self) -> Optional[nn.Module]:
        return None

    def get_layer_blocks(self) -> List[nn.Module]:
        try:
            return list(self.pipeline.model.model.transformer_blocks)
        except AttributeError:
            return []

    def has_context_phase(self) -> bool:
        return True

    @staticmethod
    def init_pipeline(cfg: Any) -> Any:
        """
        Load NitroGen model for profiling.

        In random mode: builds model with random weights (valid for timing).
        In full mode: loads pretrained checkpoint.
        """
        import sys
        from pathlib import Path
        from easydict import EasyDict as edict

        device = getattr(cfg, "device", "cuda:0")
        dtype_str = getattr(cfg, "precision", "bfloat16")
        dtype = getattr(torch, dtype_str, torch.bfloat16)

        cc = getattr(cfg, "controller_config", cfg)
        mode = getattr(cc, "weight_mode", getattr(cc, "mode", "random"))

        repo_path = getattr(cc, "repo_path", "/data1/ybyang/NitroGen")
        sys.path.insert(0, repo_path)

        from nitrogen.flow_matching_transformer.nitrogen import NitroGen, NitroGen_Config
        from nitrogen.flow_matching_transformer.modules import DiT, DiTConfig, SelfAttentionTransformer, SelfAttentionTransformerConfig

        num_inference_timesteps = getattr(cc, "num_inference_steps", 16)
        action_horizon = getattr(cc, "action_horizon", 16)
        action_dim = getattr(cc, "action_dim", 20)

        dit_num_layers = getattr(cc, "dit_num_layers", 12)
        dit_num_heads = getattr(cc, "dit_num_heads", 16)
        dit_head_dim = getattr(cc, "dit_head_dim", 64)
        vl_num_layers = getattr(cc, "vl_num_layers", 4)
        vl_num_heads = getattr(cc, "vl_num_heads", 12)
        vl_head_dim = getattr(cc, "vl_head_dim", 64)

        hidden_size = dit_num_heads * dit_head_dim

        dit_config = DiTConfig(
            num_attention_heads=dit_num_heads,
            attention_head_dim=dit_head_dim,
            output_dim=hidden_size,
            num_layers=dit_num_layers,
            dropout=0.0,
            attention_bias=True,
            activation_fn="gelu-approximate",
            norm_type="ada_norm",
            norm_elementwise_affine=False,
            positional_embeddings="sinusoidal",
            final_dropout=False,
            cross_attention_dim=getattr(cc, "vision_hidden_size", 768),
        )

        vl_config = SelfAttentionTransformerConfig(
            num_attention_heads=vl_num_heads,
            attention_head_dim=vl_head_dim,
            num_layers=vl_num_layers,
            dropout=0.0,
            attention_bias=True,
            activation_fn="gelu-approximate",
            positional_embeddings="sinusoidal",
            final_dropout=False,
        )

        vision_encoder_name = getattr(
            cc, "vision_encoder_name", "google/siglip-large-patch16-256"
        )

        model_config = NitroGen_Config(
            diffusion_model_cfg=dit_config,
            vl_self_attention_cfg=vl_config,
            hidden_size=hidden_size,
            max_seq_len=getattr(cc, "max_seq_len", 1024),
            action_dim=action_dim,
            action_horizon=action_horizon,
            num_inference_timesteps=num_inference_timesteps,
            vision_encoder_name=vision_encoder_name,
            vision_hidden_size=getattr(cc, "vision_hidden_size", 768),
            add_pos_embed=True,
            tune_vision_tower=False,
            tune_mm_projector=False,
            tune_diffusion_model=False,
            tune_multi_projector=False,
            tune_vl_mixing=False,
        )

        if mode == "random":
            logger.info("Building NitroGen with random weights (timing-only)...")
            from transformers import SiglipVisionConfig
            vision_cfg = SiglipVisionConfig(
                hidden_size=model_config.vision_hidden_size,
                intermediate_size=model_config.vision_hidden_size * 4,
                num_hidden_layers=27,
                num_attention_heads=16,
                image_size=256,
                patch_size=16,
            )
            from transformers import SiglipVisionModel
            random_vision = SiglipVisionModel(vision_cfg)
            model = NitroGen.__new__(NitroGen)
            nn.Module.__init__(model)
            model.config = model_config
            model.hidden_size = model_config.hidden_size
            model.vision_hidden_size = model_config.vision_hidden_size
            model.vision_encoder = random_vision.vision_model
            model.vision_encoder_type = "siglip"
            model.beta_dist = torch.distributions.Beta(
                model_config.noise_beta_alpha, model_config.noise_beta_beta
            )
            model.num_timestep_buckets = model_config.num_timestep_buckets
            model.model = DiT(config=model_config.diffusion_model_cfg)
            model.action_dim = model_config.action_dim
            model.action_horizon = model_config.action_horizon
            model.num_inference_timesteps = model_config.num_inference_timesteps
            model.vl_self_attention_model = SelfAttentionTransformer(
                config=model_config.vl_self_attention_cfg
            )
            model.mm_projector = None
            model.game_mapping = None

            from nitrogen.flow_matching_transformer.nitrogen import MultiEmbodimentActionEncoder, CategorySpecificMLP
            model.action_encoder = MultiEmbodimentActionEncoder(
                action_dim=model_config.action_dim,
                hidden_size=model_config.hidden_size,
                num_embodiments=model_config.max_num_embodiments,
            )
            model.action_decoder = CategorySpecificMLP(
                num_categories=model_config.max_num_embodiments,
                input_dim=model_config.hidden_size,
                hidden_dim=model_config.hidden_size,
                output_dim=model_config.action_dim,
            )
            if model_config.add_pos_embed:
                model.position_embedding = nn.Embedding(
                    model_config.max_seq_len, model_config.hidden_size
                )
                nn.init.normal_(model.position_embedding.weight, mean=0.0, std=0.02)
        else:
            ckpt_path = getattr(cc, "checkpoint_path", None)
            if ckpt_path is None:
                raise ValueError("checkpoint_path required for full mode")
            logger.info("Loading NitroGen from checkpoint: %s", ckpt_path)
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model = NitroGen(config=model_config, game_mapping=None)
            model.load_state_dict(checkpoint["model"], strict=False)

        model = model.to(device=device, dtype=dtype)
        model.eval()

        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        vision_params = sum(p.numel() for p in model.vision_encoder.parameters()) / 1e6
        dit_params = sum(p.numel() for p in model.model.parameters()) / 1e6
        vl_params = sum(p.numel() for p in model.vl_self_attention_model.parameters()) / 1e6
        logger.info(
            "NitroGen: %.0fM total (Vision=%.0fM, VL-SA=%.0fM, DiT=%.0fM)",
            total_params, vision_params, vl_params, dit_params,
        )

        return edict(
            model=model,
            device=device,
            dtype=dtype,
            config=model_config,
        )

    def prepare_inputs(self, cfg: Any) -> List[Dict[str, Any]]:
        from omegaconf import OmegaConf

        raw_inputs = getattr(cfg, "inputs", [])
        inputs = []

        for entry in raw_inputs:
            if hasattr(entry, "_iter_ex"):
                entry = OmegaConf.to_container(entry, resolve=True)
            name = entry.get("name", "unnamed")
            inputs = [*inputs, {"name": name, **entry}]

        if not inputs:
            inputs = [{"name": "single_frame"}]

        return inputs

    @torch.no_grad()
    def model_inference(
        self, pipeline: Any, cfg: Any, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run NitroGen inference with E/C/A phase timing.

        E: SigLIP ViT encode (256x256 → 256 tokens)
        C: VL self-attention transformer (vision token mixing)
        A: DiT flow matching denoise × k steps (Euler integration)
        """
        model = pipeline.model
        device = pipeline.device
        dtype = pipeline.dtype
        config = pipeline.config

        num_steps = inputs.get("num_inference_steps", model.num_inference_timesteps)
        action_horizon = config.action_horizon
        action_dim = config.action_dim
        num_visual_tokens = 256

        dummy_image = torch.randn(1, 1, 3, 256, 256, device=device, dtype=dtype)

        vl_token_ids = torch.full((1, num_visual_tokens), _IMG_TOKEN, device=device, dtype=torch.long)
        sa_token_ids = torch.full((1, action_horizon), _ACT_TOKEN, device=device, dtype=torch.long)
        vl_attn_mask = torch.ones(1, num_visual_tokens, device=device, dtype=dtype)
        dropped_images = torch.zeros(1, 1, device=device, dtype=torch.long)
        embodiment_id = torch.zeros(1, device=device, dtype=torch.long)

        # --- Phase E: Vision Encode ---
        self.timer.mark_start("encode")
        torch.cuda.synchronize()

        visual_features = model.encode_images(dummy_image)

        torch.cuda.synchronize()
        self.timer.mark_end("encode")

        # --- Phase C: VL Self-Attention ---
        self.timer.mark_start("context")
        torch.cuda.synchronize()

        vl_embs = torch.full(
            (1, num_visual_tokens, config.vision_hidden_size),
            0.0, dtype=dtype, device=device,
        )
        vision_mask = (vl_token_ids == _IMG_TOKEN)
        batch_idx, token_idx = vision_mask.nonzero(as_tuple=True)
        vision_flat = visual_features.reshape(1, -1, config.vision_hidden_size)
        vl_embs[batch_idx, token_idx] = vision_flat[0, :num_visual_tokens]

        vl_embs = model.vl_self_attention_model(vl_embs)

        torch.cuda.synchronize()
        self.timer.mark_end("context")

        # --- Phase A: DiT Flow Matching Denoise ---
        self.timer.mark_start("action")
        torch.cuda.synchronize()

        actions = torch.randn(1, action_horizon, action_dim, device=device, dtype=dtype)
        dt = 1.0 / num_steps

        per_step_times = []

        for i in range(num_steps):
            step_start = torch.cuda.Event(enable_timing=True)
            step_end = torch.cuda.Event(enable_timing=True)
            step_start.record()

            t_cont = i / float(num_steps)
            t_discretized = int(t_cont * model.num_timestep_buckets)

            action_features = model.action_encoder(
                actions,
                (torch.ones(1, device=device) * t_discretized),
                embodiment_id,
            )

            sa_embs = torch.full(
                (1, action_horizon, config.hidden_size),
                0.0, dtype=dtype, device=device,
            )
            action_mask = (sa_token_ids == _ACT_TOKEN).unsqueeze(-1).expand_as(sa_embs)
            sa_embs = sa_embs.masked_scatter(action_mask, action_features)

            if config.add_pos_embed:
                pos_ids = torch.arange(action_horizon, device=device, dtype=torch.long)
                sa_embs = sa_embs + model.position_embedding(pos_ids).unsqueeze(0)

            timesteps = torch.tensor([t_discretized], device=device, dtype=torch.long)
            model_output = model.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                encoder_attention_mask=vl_attn_mask,
                timestep=timesteps,
            )

            pred = model.action_decoder(model_output, embodiment_id)
            pred_velocity = pred[:, -action_horizon:]
            actions = actions + dt * pred_velocity

            step_end.record()
            torch.cuda.synchronize()
            per_step_times = [*per_step_times, step_start.elapsed_time(step_end)]

        torch.cuda.synchronize()
        self.timer.mark_end("action")

        return {
            "actions": actions,
            "actions_shape": list(actions.shape),
            "num_denoise_steps": num_steps,
            "per_step_ms": per_step_times,
            "mean_step_ms": sum(per_step_times) / len(per_step_times) if per_step_times else 0,
        }

    def register_profiling_hooks(self) -> None:
        self.logger.info(
            "NitroGen: profiling via manual timer marks in model_inference "
            "(E=SigLIP encode, C=VL self-attention, A=DiT flow denoise)"
        )

    def _register_layer_analysis_hooks(
        self, block: nn.Module, layer_idx: int
    ) -> None:
        if hasattr(block, "attn1"):
            self_attn = block.attn1
            if "self_q" in self.store_type:
                self._register_capture_hook(self_attn.to_q, layer_idx, "q")
            if "self_k" in self.store_type:
                self._register_capture_hook(self_attn.to_k, layer_idx, "k")
            if "self_v" in self.store_type:
                self._register_capture_hook(self_attn.to_v, layer_idx, "v")

    def save_results(
        self,
        inputs: Dict[str, Any],
        results: Dict[str, Any],
        cfg: Any,
    ) -> str:
        import json
        import os

        name = inputs.get("name", "unnamed")
        base_output = getattr(
            cfg, "output_path", getattr(cfg, "base_output_path", "./output")
        )
        save_dir = os.path.join(base_output, name)
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"{name}_result.json")
        serializable = {k: v for k, v in results.items() if not isinstance(v, torch.Tensor)}
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)

        self.logger.info("Results saved to %s", save_path)
        return save_dir


_IMG_TOKEN = 1
_ACT_TOKEN = 4

CONTROLLER_REGISTRY.register("nitrogen", NitroGenController)

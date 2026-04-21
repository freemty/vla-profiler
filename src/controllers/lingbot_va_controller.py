"""
LingBot-VA controller for profiling and analysis.

LingBot-VA (arXiv:2601.21998) is a full WAM (World Action Model) that generates
video (imagination) before predicting actions. Uses a shared 5B WanTransformer3DModel
(from Wan2.2-TI2V-5B) for both video and action denoising.

Architecture: VAE (encode) → DiT×20 (video denoise) → DiT×50 (action denoise)
Phases: E/V/A (inherits BaseVLAController but overrides PHASES)

Model hierarchy:
  VA_Server
    .vae                 (AutoencoderKLWan)
    .streaming_vae       (WanVAEStreamingWrapper wrapping vae)
    .text_encoder        (UMT5EncoderModel ~4.7B)
    .tokenizer           (T5TokenizerFast)
    .transformer         (WanTransformer3DModel ~5B, shared video+action)
      .patch_embedding_mlp   (video patch projector)
      .action_embedder       (action token projector)
      .condition_embedder    (video timestep embedder)
      .condition_embedder_action  (action timestep embedder)
      .blocks[0..29]         (WanTransformerBlock — shared)
      .proj_out              (video output head)
      .action_proj_out       (action output head)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.controllers import CONTROLLER_REGISTRY
from src.controllers.base_vla_controller import BaseVLAController


logger = logging.getLogger(__name__)


class LingBotVAController(BaseVLAController):
    """
    Hook controller for LingBot-VA (full WAM).

    Phases: E (VAE encode) / V (Video denoise) / A (Action denoise)
    Unlike standard VLAs, this model has NO language model in the loop —
    text embeddings are pre-computed and injected via cross-attention.
    """

    PHASES = ("encode", "video_denoise", "action_denoise")

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
        return self.pipeline.streaming_vae.encoder

    def get_action_head(self) -> nn.Module:
        return self.pipeline.transformer.action_proj_out

    def get_denoise_steps(self) -> int:
        return self.pipeline.config.action_num_inference_steps

    def get_video_denoise_steps(self) -> int:
        return self.pipeline.config.num_inference_steps

    def get_language_model(self) -> Optional[nn.Module]:
        return self.pipeline.text_encoder

    def get_layer_blocks(self) -> List[nn.Module]:
        try:
            return list(self.pipeline.transformer.blocks)
        except AttributeError:
            return []

    def has_context_phase(self) -> bool:
        return False

    @staticmethod
    def init_pipeline(cfg: Any) -> Any:
        """
        Load LingBot-VA components for profiling.

        In random mode: builds transformer + VAE with random weights (valid for timing).
        In full mode: loads pretrained Wan2.2-TI2V-5B weights.
        """
        import sys
        from pathlib import Path
        from easydict import EasyDict as edict

        device = getattr(cfg, "device", "cuda:0")
        dtype = torch.bfloat16
        mode = getattr(cfg, "mode", "random")

        repo_path = getattr(cfg, "repo_path", "/data1/ybyang/lingbot-va")
        sys.path.insert(0, repo_path)

        from wan_va.modules.model import WanTransformer3DModel
        from wan_va.modules.utils import WanVAEStreamingWrapper
        from wan_va.utils import FlowMatchScheduler, get_mesh_id

        config = edict(
            height=getattr(cfg, "height", 128),
            width=getattr(cfg, "width", 128),
            num_cameras=getattr(cfg, "num_cameras", 2),
            frame_chunk_size=getattr(cfg, "frame_chunk_size", 4),
            action_dim=getattr(cfg, "action_dim", 30),
            action_per_frame=getattr(cfg, "action_per_frame", 4),
            num_inference_steps=getattr(cfg, "video_steps", 20),
            action_num_inference_steps=getattr(cfg, "action_steps", 50),
            guidance_scale=getattr(cfg, "guidance_scale", 5),
            action_guidance_scale=getattr(cfg, "action_guidance_scale", 1),
            snr_shift=getattr(cfg, "snr_shift", 5.0),
            action_snr_shift=getattr(cfg, "action_snr_shift", 1.0),
            patch_size=(1, 2, 2),
            attn_window=getattr(cfg, "attn_window", 30),
        )

        if mode == "random":
            logger.info("Building LingBot-VA with random weights (timing-only)...")

            transformer = WanTransformer3DModel(
                patch_size=[1, 2, 2],
                num_attention_heads=24,
                attention_head_dim=128,
                in_channels=48,
                out_channels=48,
                action_dim=config.action_dim,
                text_dim=4096,
                freq_dim=256,
                ffn_dim=14336,
                num_layers=30,
                cross_attn_norm=True,
                eps=1e-6,
                rope_max_seq_len=1024,
                attn_mode="torch",
            ).to(device=device, dtype=dtype)
            transformer.eval()

            from diffusers import AutoencoderKLWan
            vae = AutoencoderKLWan(
                z_dim=48,
                in_channels=3,
                out_channels=3,
            ).to(device=device, dtype=dtype)
            vae.eval()

            text_encoder = None

        else:
            wan_path = getattr(
                cfg, "wan_path",
                "/data1/ybyang/huggingface/Wan-AI/Wan2.2-TI2V-5B-Diffusers"
            )
            logger.info("Loading LingBot-VA from pretrained: %s", wan_path)

            from wan_va.modules.utils import load_transformer, load_vae
            transformer = load_transformer(
                f"{wan_path}/transformer",
                torch_dtype=dtype,
                torch_device=device,
                attn_mode="torch",
            )
            transformer.eval()

            vae = load_vae(f"{wan_path}/vae", torch_dtype=dtype, torch_device=device)
            vae.eval()

            text_encoder = None

        streaming_vae = WanVAEStreamingWrapper(vae)

        video_scheduler = FlowMatchScheduler(
            shift=config.snr_shift, sigma_min=0.0, extra_one_step=True,
        )
        action_scheduler = FlowMatchScheduler(
            shift=config.action_snr_shift, sigma_min=0.0, extra_one_step=True,
        )

        param_count = sum(p.numel() for p in transformer.parameters()) / 1e6
        vae_params = sum(p.numel() for p in vae.parameters()) / 1e6
        logger.info("Transformer: %.0fM params, VAE: %.0fM params", param_count, vae_params)

        return edict(
            transformer=transformer,
            vae=vae,
            streaming_vae=streaming_vae,
            text_encoder=text_encoder,
            video_scheduler=video_scheduler,
            action_scheduler=action_scheduler,
            config=config,
            device=device,
            dtype=dtype,
            get_mesh_id=get_mesh_id,
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
            inputs = [{"name": "default_2cam"}]

        return inputs

    @torch.no_grad()
    def model_inference(
        self, pipeline: Any, cfg: Any, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run LingBot-VA i2va-style inference (single chunk).

        E: VAE encode (observation images → latents)
        V: Video denoise (DiT forward × N_video steps)
        A: Action denoise (DiT forward × N_action steps)
        """
        config = pipeline.config
        device = pipeline.device
        dtype = pipeline.dtype
        transformer = pipeline.transformer
        streaming_vae = pipeline.streaming_vae
        get_mesh_id = pipeline.get_mesh_id

        frame_chunk_size = config.frame_chunk_size
        patch_size = config.patch_size
        latent_height = config.height // 16
        latent_width = config.width // 16 * config.num_cameras

        prompt_embeds = torch.randn(1, 512, 4096, device=device, dtype=dtype)
        use_cfg = config.guidance_scale > 1

        # --- Phase E: VAE Encode ---
        self.timer.mark_start("encode")
        torch.cuda.synchronize()

        dummy_video = torch.randn(
            config.num_cameras, 3, frame_chunk_size,
            config.height, config.width,
            device=device, dtype=dtype,
        )
        video_input = dummy_video / 255.0 * 2.0 - 1.0
        streaming_vae.clear_cache()
        enc_out = streaming_vae.encode_chunk(video_input)
        mu, _ = torch.chunk(enc_out, 2, dim=1)

        init_latent = torch.randn(
            1, 48, frame_chunk_size, latent_height, latent_width,
            device=device, dtype=dtype,
        )

        torch.cuda.synchronize()
        self.timer.mark_end("encode")

        # --- Phase V: Video Denoise ---
        self.timer.mark_start("video_denoise")
        torch.cuda.synchronize()

        latents = torch.randn(
            1, 48, frame_chunk_size, latent_height, latent_width,
            device=device, dtype=dtype,
        )
        latents[:, :, 0:1] = init_latent[:, :, 0:1]

        pipeline.video_scheduler.set_timesteps(config.num_inference_steps)
        timesteps = pipeline.video_scheduler.timesteps
        timesteps = F.pad(timesteps, (0, 1), mode="constant", value=0)

        for step_i, t in enumerate(timesteps[:-1]):
            grid_id = get_mesh_id(
                latents.shape[-3] // patch_size[0],
                latents.shape[-2] // patch_size[1],
                latents.shape[-1] // patch_size[2],
                0, 1, 0,
            ).to(device)

            timestep_vec = torch.ones(
                [1, latents.shape[2]], dtype=torch.float32, device=device,
            ) * t

            input_dict = {
                "noisy_latents": latents,
                "timesteps": timestep_vec,
                "grid_id": grid_id[None],
                "text_emb": prompt_embeds.clone(),
            }

            noise_pred = transformer(
                input_dict, update_cache=0, cache_name="pos", action_mode=False,
            )

            noise_pred_spatial = noise_pred.view(
                1, frame_chunk_size,
                latent_height // patch_size[1],
                latent_width // patch_size[2],
                48 * patch_size[0] * patch_size[1] * patch_size[2],
            )
            noise_pred_spatial = rearrange(
                noise_pred_spatial,
                "b f h w (c p1 p2) -> b c (f) (h p1) (w p2)",
                p1=patch_size[1], p2=patch_size[2], c=48,
            )

            latents = pipeline.video_scheduler.step(
                noise_pred_spatial, t, latents, return_dict=False,
            )
            latents[:, :, 0:1] = init_latent[:, :, 0:1]

        torch.cuda.synchronize()
        self.timer.mark_end("video_denoise")

        # --- Phase A: Action Denoise ---
        self.timer.mark_start("action_denoise")
        torch.cuda.synchronize()

        actions = torch.randn(
            1, config.action_dim, frame_chunk_size, config.action_per_frame, 1,
            device=device, dtype=dtype,
        )

        pipeline.action_scheduler.set_timesteps(config.action_num_inference_steps)
        action_timesteps = pipeline.action_scheduler.timesteps
        action_timesteps = F.pad(action_timesteps, (0, 1), mode="constant", value=0)

        for step_i, t in enumerate(action_timesteps[:-1]):
            grid_id = get_mesh_id(
                actions.shape[-3],
                actions.shape[-2],
                actions.shape[-1],
                1, 1, 0, action=True,
            ).to(device)

            action_timestep_vec = torch.ones(
                [1, actions.shape[2]], dtype=torch.float32, device=device,
            ) * t

            input_dict = {
                "noisy_latents": actions,
                "timesteps": action_timestep_vec,
                "grid_id": grid_id[None],
                "text_emb": prompt_embeds.clone(),
            }

            action_pred = transformer(
                input_dict, update_cache=0, cache_name="pos", action_mode=True,
            )

            action_pred = rearrange(
                action_pred, "b (f n) c -> b c f n 1", f=frame_chunk_size,
            )
            actions = pipeline.action_scheduler.step(
                action_pred, t, actions, return_dict=False,
            )

        torch.cuda.synchronize()
        self.timer.mark_end("action_denoise")

        return {
            "latents_shape": list(latents.shape),
            "actions_shape": list(actions.shape),
            "video_steps": config.num_inference_steps,
            "action_steps": config.action_num_inference_steps,
        }

    def register_profiling_hooks(self) -> None:
        """
        LingBot-VA uses manual timer marks in model_inference instead of
        module hooks, because the same transformer is called for both
        video and action phases — module hooks can't distinguish them.
        """
        self.logger.info(
            "LingBot-VA: profiling via manual timer marks in model_inference "
            "(E=VAE encode, V=video denoise, A=action denoise)"
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


CONTROLLER_REGISTRY.register("lingbot_va", LingBotVAController)

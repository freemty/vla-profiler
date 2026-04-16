"""
Pi-Zero (π₀) controller via OpenPI PyTorch backend.

Architecture: PaliGemma (SigLIP + Gemma 2B) + Gemma 300M Action Expert
Phase model: E/C/A
  - E: SigLIP vision encoder
  - C: PaliGemma prefill (Gemma 2B LM, one-time, caches KV)
  - A: Flow matching denoise (Gemma 300M Expert, 10 Euler steps)

Requires: separate conda env (openpi) due to torch 2.7 + transformers 4.53 pinning.
Setup: see docs/knowhow/runbooks/setup-openpi-env.md
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from src.controllers import CONTROLLER_REGISTRY
from src.controllers.base_vla_controller import BaseVLAController


logger = logging.getLogger(__name__)


class PiZeroController(BaseVLAController):
    """
    Hook controller for Pi-Zero via OpenPI.

    Dual-stream Transformer: PaliGemma processes image+text,
    Action Expert processes noisy actions. Expert attends to PaliGemma KV
    but not vice versa — enabling KV cache reuse across denoise steps.
    """

    DEFAULT_DENOISE_STEPS = 10

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

    # ---- Abstract implementations ----

    def get_vision_encoder(self) -> nn.Module:
        """Return SigLIP ViT-So400m/14 vision tower."""
        return self.pipeline.model.paligemma_with_expert.paligemma.model.vision_tower

    def get_action_head(self) -> nn.Module:
        """Return Gemma 300M Action Expert."""
        return self.pipeline.model.paligemma_with_expert.gemma_expert

    def get_denoise_steps(self) -> int:
        return self._denoise_steps

    def get_language_model(self) -> Optional[nn.Module]:
        """Return PaliGemma Gemma 2B language model."""
        return self.pipeline.model.paligemma_with_expert.paligemma.language_model

    def get_layer_blocks(self) -> List[nn.Module]:
        """Return PaliGemma LM layers for analysis hooks."""
        lm = self.get_language_model()
        if lm is None:
            return []
        # Gemma model.layers
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return list(lm.model.layers)
        if hasattr(lm, "layers"):
            return list(lm.layers)
        return []

    def has_context_phase(self) -> bool:
        """Pi-Zero has PaliGemma context encoding."""
        return True

    @staticmethod
    def init_pipeline(cfg: Any) -> Any:
        """
        Initialize Pi-Zero via OpenPI.

        Requires openpi package in the active conda env.
        Model weights loaded from config path or downloaded via gcloud.
        """
        from easydict import EasyDict as edict

        try:
            from openpi.models import model as openpi_model
            from openpi.policies.pi0 import Pi0Policy
            from openpi.training.config import Pi0Config
        except ImportError as e:
            raise ImportError(
                "OpenPI not installed. Activate the openpi conda env first. "
                "See docs/knowhow/runbooks/setup-openpi-env.md"
            ) from e

        model_path = getattr(cfg, "model_name", "")
        denoise_steps = getattr(cfg, "denoise_steps", 10)

        # Load config and policy
        config = Pi0Config()
        if model_path:
            config.checkpoint_path = model_path

        policy = Pi0Policy(config)
        policy.model.eval()

        device = getattr(cfg, "device", "cuda:0")
        policy.model = policy.model.to(device)

        return edict(
            model=policy.model,
            policy=policy,
            config=config,
        )

    def prepare_inputs(self, cfg: Any) -> List[Dict[str, Any]]:
        """
        Prepare inputs for Pi-Zero inference.

        Pi-Zero expects: image(s), language instruction, robot state.
        For profiling, we can use synthetic data if no real data provided.
        """
        from omegaconf import OmegaConf

        raw_inputs = getattr(cfg, "inputs", [])
        inputs = []

        for entry in raw_inputs:
            if hasattr(entry, "_iter_ex"):
                entry = OmegaConf.to_container(entry, resolve=True)

            name = entry.get("name", "unnamed")
            image_shape = entry.get("image_shape", [3, 224, 224])
            state_dim = entry.get("state_dim", 32)
            device = getattr(cfg, "device", "cuda:0")

            observation = {
                "image": torch.randn(1, *image_shape, device=device, dtype=torch.float32),
                "state": torch.randn(1, state_dim, device=device, dtype=torch.float32),
                "instruction": entry.get("instruction", "pick up the object"),
            }

            inputs = [*inputs, {"name": name, "observation": observation}]

        return inputs

    @torch.no_grad()
    def model_inference(
        self, pipeline: Any, cfg: Any, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run Pi-Zero inference: prefix forward + N denoise steps.

        Uses policy.infer() or policy.sample_actions() depending on API.
        """
        policy = pipeline.policy
        observation = inputs.get("observation", {})

        # Call the policy inference
        try:
            # OpenPI API: policy.infer(observation)
            result = policy.infer(observation)
            actions = result.get("actions", result)
        except AttributeError:
            # Fallback: direct model call
            actions = pipeline.model.sample_actions(observation)

        return {
            "actions": actions.cpu() if torch.is_tensor(actions) else actions,
            "action_shape": list(actions.shape) if torch.is_tensor(actions) else [],
            "denoise_steps": self._denoise_steps,
        }

    def save_results(
        self,
        inputs: Dict[str, Any],
        results: Dict[str, Any],
        cfg: Any,
    ) -> str:
        """Save inference results."""
        name = inputs.get("name", "unnamed")
        base_output = getattr(cfg, "output_path", getattr(cfg, "base_output_path", "./output"))
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


CONTROLLER_REGISTRY.register("pizero", PiZeroController)

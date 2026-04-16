"""
ACT (Action Chunking Transformer) controller via LeRobot.

ACT is a single-forward VLA: ResNet backbone → CVAE/Transformer → action chunk.
No iterative decode or denoise — one forward pass produces the full action sequence.
Phase model: E (vision encode) → A (single forward action generation).

Requires: pip install lerobot
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


class ACTController(BaseVLAController):
    """
    Hook controller for ACT policy via LeRobot.

    Architecture: ResNet18 backbone → ACTEncoder → ACTDecoder → Linear action_head
    Single forward pass: no autoregressive decode, no denoise loop.
    Output: action chunk (chunk_size x action_dim), e.g., 100 steps x 14-DoF
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

    # ---- Abstract implementations ----

    def get_vision_encoder(self) -> nn.Module:
        """Return ResNet backbone."""
        return self.pipeline.model.model.backbone

    def get_action_head(self) -> nn.Module:
        """Return the decoder + action_head as the action generation module.
        We hook the decoder since it's the main compute for action generation."""
        return self.pipeline.model.model.decoder

    def get_denoise_steps(self) -> int:
        """ACT is single-forward — 1 step."""
        return 1

    def has_context_phase(self) -> bool:
        """ACT has no VLM backbone, no text input, no context phase."""
        return False

    def get_language_model(self) -> Optional[nn.Module]:
        """ACT has no language model."""
        return None

    def get_layer_blocks(self) -> List[nn.Module]:
        """Return encoder + decoder layers for analysis hooks."""
        model = self.pipeline.model.model
        encoder_layers = list(model.encoder.layers)
        decoder_layers = list(model.decoder.layers)
        return [*encoder_layers, *decoder_layers]

    @staticmethod
    def init_pipeline(cfg: Any) -> Any:
        """
        Initialize ACT policy from config.

        Supports two modes:
        - pretrained_path: load from a LeRobot checkpoint
        - default: create with config parameters
        """
        from easydict import EasyDict as edict
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.configs.types import FeatureType, PolicyFeature

        pretrained_path = getattr(cfg, "pretrained_path", None)

        if pretrained_path:
            policy = ACTPolicy.from_pretrained(pretrained_path)
        else:
            # Create with default config + overrides from cfg
            act_cfg = ACTConfig()

            # Apply overrides from cfg if present
            for key in ["dim_model", "n_heads", "n_encoder_layers", "n_decoder_layers", "chunk_size"]:
                if hasattr(cfg, key):
                    setattr(act_cfg, key, getattr(cfg, key))

            # Input/output features
            image_shape = getattr(cfg, "image_shape", [3, 480, 640])
            state_dim = getattr(cfg, "state_dim", 14)
            action_dim = getattr(cfg, "action_dim", 14)

            act_cfg.input_features = {
                "observation.images.top": PolicyFeature(
                    type=FeatureType.VISUAL, shape=image_shape
                ),
                "observation.state": PolicyFeature(
                    type=FeatureType.STATE, shape=[state_dim]
                ),
            }
            act_cfg.output_features = {
                "action": PolicyFeature(
                    type=FeatureType.ACTION, shape=[action_dim]
                ),
            }

            policy = ACTPolicy(act_cfg)

        device = getattr(cfg, "device", "cuda:0")
        policy = policy.to(device)
        policy.eval()

        return edict(model=policy, config=getattr(policy, "config", act_cfg))

    def prepare_inputs(self, cfg: Any) -> List[Dict[str, Any]]:
        """
        Prepare synthetic inputs for profiling.

        ACT takes observation dict {images, state}, not text prompts.
        For profiling, we generate random tensors matching expected shapes.
        """
        from src.utils import to_plain

        raw_inputs = getattr(cfg, "inputs", [])
        inputs = []

        for entry in raw_inputs:
            entry = to_plain(entry)

            name = entry.get("name", "unnamed")
            image_shape = entry.get("image_shape", [3, 480, 640])
            state_dim = entry.get("state_dim", 14)
            device = getattr(cfg, "device", "cuda:0")

            # Create synthetic observation matching ACT's expected format
            # observation.images is a list of camera tensors, one per camera
            observation = {
                "observation.images": [
                    torch.randn(1, *image_shape, device=device, dtype=torch.float32),
                ],
                "observation.state": torch.randn(
                    1, state_dim, device=device, dtype=torch.float32
                ),
            }

            inputs = [*inputs, {"name": name, "observation": observation}]

        return inputs

    @torch.no_grad()
    def model_inference(
        self, pipeline: Any, cfg: Any, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run ACT inference — single forward pass through the model.

        Calls model.forward() directly instead of select_action() to avoid
        the action queue caching (which skips forward on subsequent calls).
        """
        policy = pipeline.model
        observation = inputs.get("observation", {})

        # Call model.forward() directly to ensure hooks fire every time.
        # select_action() has an action queue that caches chunk_size steps,
        # meaning only 1 in N calls actually runs the model.
        actions, _ = policy.model(observation)

        return {
            "action": actions.cpu(),
            "action_shape": list(actions.shape),
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
            }, f, ensure_ascii=False, indent=2)

        self.logger.info("Results saved to %s", save_path)
        return save_dir

    # ---- Analysis hooks ----

    def _register_layer_analysis_hooks(
        self, block: nn.Module, layer_idx: int
    ) -> None:
        """Register hooks on encoder/decoder attention layers."""
        # ACTEncoder/Decoder layers have self_attn
        if hasattr(block, "self_attn"):
            self_attn = block.self_attn

            if "self_q" in self.store_type and hasattr(self_attn, "in_proj_weight"):
                # Multi-head attention uses combined in_proj
                store_key = f"{layer_idx}_attn_output"
                controller = self

                def _capture_attn(module, inputs, output):
                    if controller.should_store(store_key):
                        # output is (attn_output, attn_weights)
                        if isinstance(output, tuple) and len(output) >= 2:
                            attn_weights = output[1]
                            if attn_weights is not None:
                                detached = attn_weights.detach().to("cpu", non_blocking=True)
                                if store_key not in controller.step_store:
                                    controller.step_store[store_key] = [detached]
                                else:
                                    controller.step_store[store_key] = [
                                        *controller.step_store[store_key],
                                        detached,
                                    ]
                    return output

                hook = self_attn.register_forward_hook(_capture_attn)
                self.analysis_hooks[f"analysis:layer{layer_idx}_attn"] = hook


CONTROLLER_REGISTRY.register("act", ACTController)

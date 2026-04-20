"""
LingBot-VLA controller for profiling and analysis.

LingBot-VLA is a 4B VLA based on Qwen2.5-VL-3B-Instruct with Pi0-style
flow action head. Loaded via lingbotvla package (not AutoModelForCausalLM).

Architecture: Qwen2.5-VL (vision + LLM) -> Flow action head (10 denoise steps)
Phases: E/C/A (inherits BaseVLAController)

Model hierarchy:
  LingbotVlaPolicy
    .language_tokenizer  (AutoTokenizer)
    .model               (FlowMatching)
      .qwenvl_with_expert  (QwenvlWithExpertModel)
        .qwenvl              (Qwen2_5_VLForConditionalGeneration)
          .visual              <- vision encoder
          .model               <- language model (Qwen2.5-VL LLM)
            .layers              <- transformer blocks
        .qwen_expert         (Qwen2ForCausalLM — action expert)
      .state_proj          (Linear)
      .action_in_proj      (Linear)
      .action_out_proj     (Linear) <- action head output
      .action_time_mlp_in  (Linear)
      .action_time_mlp_out (Linear)
"""

from __future__ import annotations

import json
import logging
import os
from glob import glob
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from src.controllers import CONTROLLER_REGISTRY
from src.controllers.base_vla_controller import BaseVLAController


logger = logging.getLogger(__name__)


class LingBotVLAController(BaseVLAController):
    """
    Hook controller for LingBot-VLA-4B.

    Architecture: Qwen2.5-VL-3B (visual + LLM) -> Flow action head
    Action: 50-step chunk, 75-dim, 10 denoise steps
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
        return self.pipeline.policy.model.qwenvl_with_expert.qwenvl.visual

    def get_action_head(self) -> nn.Module:
        return self.pipeline.policy.model.action_out_proj

    def get_denoise_steps(self) -> int:
        return self.pipeline.config.get("num_steps", 10)

    def get_language_model(self) -> Optional[nn.Module]:
        return self.pipeline.policy.model.qwenvl_with_expert.qwenvl.model

    def get_layer_blocks(self) -> List[nn.Module]:
        try:
            return list(
                self.pipeline.policy.model.qwenvl_with_expert.qwenvl.model.layers
            )
        except AttributeError:
            return []

    @staticmethod
    def init_pipeline(cfg: Any) -> Any:
        """
        Load LingBot-VLA via lingbotvla package.

        LingbotVlaPolicy needs a PI0Config object (not raw dict) and a
        tokenizer_path pointing to the Qwen2.5-VL-3B-Instruct directory
        (used for both tokenizer and VLM architecture config).
        """
        from easydict import EasyDict as edict
        from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
        from safetensors import safe_open

        from lingbotvla.models.vla.pi0.modeling_lingbot_vla import LingbotVlaPolicy

        model_path = cfg.model_name
        qwen_path = getattr(cfg, "qwen_path", None) or os.environ.get(
            "QWEN25_PATH", "/data1/ybyang/huggingface/Qwen/Qwen2.5-VL-3B-Instruct"
        )

        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        # Build PI0Config with valid fields, then attach extra lingbotvla attrs
        import dataclasses
        valid_fields = {f.name for f in dataclasses.fields(PI0Config)}
        pi0_kwargs = {k: v for k, v in config_dict.items() if k in valid_fields}
        pi0_config = PI0Config(**pi0_kwargs)
        # Attach all fields from config.json that FlowMatching/QwenvlWithExpert need
        for k, v in config_dict.items():
            if not hasattr(pi0_config, k):
                setattr(pi0_config, k, v)
        pi0_config.tokenizer_path = qwen_path
        # Override flex attention (compat issues with PyTorch 2.8 block mask)
        pi0_config.attention_implementation = "eager"
        # Defaults for lingbotvla attrs not in config.json
        _lingbot_defaults = {
            "enable_expert_vision": False,
            "expert_vision_type": None,
            "train_expert_only": False,
            "loss_type": "l2",
            "align_params": {},
            "adanorm_time": False,
            "split_gate_liner": False,
            "nosplit_gate_liner": False,
            "separate_time_proj": False,
            "old_adanorm": False,
            "final_norm_adanorm": False,
            "norm_qkv": False,
            "action_dim": config_dict.get("max_action_dim", 75),
            "vlm_repo_id": None,
            "expert_vision_path": None,
            "incremental_training": False,
            "depth_incremental_training": False,
            "post_training": False,
        }
        for k, v in _lingbot_defaults.items():
            if not hasattr(pi0_config, k):
                setattr(pi0_config, k, v)

        policy = LingbotVlaPolicy(
            config=pi0_config,
            tokenizer_path=qwen_path,
            eval=True,
        )

        # Load trained weights from safetensors
        all_safetensors = sorted(glob(os.path.join(model_path, "*.safetensors")))
        if not all_safetensors:
            raise FileNotFoundError(
                f"No .safetensors files found in {model_path}. "
                "Check model_name path and ensure weights are downloaded."
            )
        merged_weights = {}
        for fpath in all_safetensors:
            with safe_open(fpath, framework="pt", device="cpu") as f:
                for key in f.keys():
                    merged_weights[key] = f.get_tensor(key)

        missing, unexpected = policy.load_state_dict(merged_weights, strict=False)
        logger.info(
            "Loaded %d weight tensors (%d missing, %d unexpected)",
            len(merged_weights),
            len(missing),
            len(unexpected),
        )

        device = getattr(cfg, "device", "cuda:0")
        policy = policy.to(device=device, dtype=torch.bfloat16)
        policy.eval()

        return edict(
            policy=policy,
            config=config_dict,
            device=device,
        )

    def prepare_inputs(self, cfg: Any) -> List[Dict[str, Any]]:
        from omegaconf import OmegaConf

        raw_inputs = getattr(cfg, "inputs", [])
        inputs = []

        for entry in raw_inputs:
            if hasattr(entry, "_iter_ex"):
                entry = OmegaConf.to_container(entry, resolve=True)
            name = entry.get("name", "unnamed")
            messages = entry.get("messages", [])
            inputs = [*inputs, {"name": name, "messages": messages}]

        return inputs

    @torch.no_grad()
    def model_inference(
        self, pipeline: Any, cfg: Any, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run LingBot-VLA inference via FlowMatching.sample_actions().

        Synthesizes dummy observation tensors (images + state + language)
        and runs the full flow-matching denoise loop. Profiling hooks on
        vision encoder / LLM layers / action_out_proj capture E/C/A timing.
        """
        policy = pipeline.policy
        device = pipeline.device
        config = pipeline.config

        num_cameras = 1
        messages = inputs.get("messages", [])
        for msg in messages:
            if msg.get("role") == "user":
                img_count = sum(
                    1 for c in msg.get("content", []) if c.get("type") == "image"
                )
                if img_count > 0:
                    num_cameras = img_count

        # Qwen2.5-VL ViT expects pre-patchified image tensors, not raw pixels.
        # For 224x224 input with patch_size=14: grid = 16x16 = 256 patches,
        # after spatial_merge_size=2: 256/4 = 64 merged patches per image.
        # embed_image receives (num_cam, num_patches, hidden_dim) where
        # num_patches = (H/patch_size)^2 and hidden_dim = C * temporal * patch^2 = 1176.
        img_size = config.get("resize_imgs_with_padding", [224, 224])
        patch_size = 14
        grid_h = img_size[0] // patch_size  # 16
        grid_w = img_size[1] // patch_size  # 16
        num_patches = grid_h * grid_w  # 256
        patch_dim = 3 * 2 * patch_size * patch_size  # 1176 (C * temporal * P^2)

        # Shape: (batch, num_cam, num_patches, patch_dim) — ndim==4 path in embed_prefix
        images = torch.randn(
            1, num_cameras, num_patches, patch_dim,
            dtype=torch.bfloat16, device=device,
        )
        img_masks = torch.ones(1, num_cameras, dtype=torch.bool, device=device)

        max_state_dim = config.get("max_state_dim", 75)
        state = torch.zeros(1, max_state_dim, dtype=torch.bfloat16, device=device)

        instruction = "pick up the red cup"
        tokenizer = policy.language_tokenizer
        max_len = config.get("tokenizer_max_length", 72)
        lang_encoded = tokenizer(
            instruction,
            return_tensors="pt",
            padding="max_length",
            max_length=max_len,
            truncation=True,
        )
        lang_tokens = lang_encoded["input_ids"].to(device)
        lang_masks = lang_encoded["attention_mask"].to(device)

        self._context_started = False
        actions = policy.model.sample_actions(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
        )

        return {
            "action_shape": list(actions.shape) if torch.is_tensor(actions) else "unknown",
            "num_cameras": num_cameras,
            "num_denoise_steps": config.get("num_steps", 10),
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
            json.dump(results, f, ensure_ascii=False, indent=2)

        self.logger.info("Results saved to %s", save_path)
        return save_dir

    # ---- Profiling hook override ----

    def register_profiling_hooks(self) -> None:
        """
        Override: lingbotvla calls each layer twice (compute_kqv + output_atten),
        so per-layer first/last hooks double-fire. Instead, register context
        timing on qwenvl_with_expert.forward as a whole.
        """
        timer = self.timer
        vision_encoder = self.get_vision_encoder()
        action_head = self.get_action_head()

        # E: vision encoder
        def _encode_pre(module, inputs):
            timer.mark_start("encode")

        def _encode_post(module, inputs, output):
            timer.mark_end("encode")

        self.analysis_hooks["profiling:encode_pre"] = (
            vision_encoder.register_forward_pre_hook(_encode_pre)
        )
        self.analysis_hooks["profiling:encode_post"] = (
            vision_encoder.register_forward_hook(_encode_post)
        )

        # C: context — hook on qwenvl_with_expert, only first call per inference
        # (first call = prefix KV cache fill; subsequent calls = denoise steps)
        qwenvl_with_expert = self.pipeline.policy.model.qwenvl_with_expert
        self._context_started = False

        def _context_pre(module, inputs):
            if not self._context_started:
                timer.mark_start("context")
                self._context_started = True

        def _context_post(module, inputs, output):
            if "context" in timer._active:
                timer.mark_end("context")

        self.analysis_hooks["profiling:context_pre"] = (
            qwenvl_with_expert.register_forward_pre_hook(_context_pre)
        )
        self.analysis_hooks["profiling:context_post"] = (
            qwenvl_with_expert.register_forward_hook(_context_post)
        )

        # A: action head (accumulates across denoise steps)
        def _action_pre(module, inputs):
            timer.mark_start("action")

        def _action_post(module, inputs, output):
            timer.mark_end("action")

        self.analysis_hooks["profiling:action_pre"] = (
            action_head.register_forward_pre_hook(_action_pre)
        )
        self.analysis_hooks["profiling:action_post"] = (
            action_head.register_forward_hook(_action_post)
        )

        self.logger.info(
            "Registered LingBot-VLA profiling hooks (E=visual, C=qwenvl_with_expert, A=action_out_proj)"
        )

    # ---- Analysis hooks ----

    def _register_layer_analysis_hooks(
        self, block: nn.Module, layer_idx: int
    ) -> None:
        self_attn = block.self_attn

        if "self_q" in self.store_type:
            self._register_capture_hook(self_attn.q_proj, layer_idx, "q")

        if "self_k" in self.store_type:
            self._register_capture_hook(self_attn.k_proj, layer_idx, "k")

        if "self_v" in self.store_type:
            self._register_capture_hook(self_attn.v_proj, layer_idx, "v")


CONTROLLER_REGISTRY.register("lingbot_vla", LingBotVLAController)

"""Minimal PI0Config stub for lingbotvla eval.

Only provides the dataclass shell. All real config values come from
config.json via setattr in the loading code.
"""
from dataclasses import dataclass, field
from lerobot.configs.policies import PreTrainedConfig


DEFAULT_IMAGE_SIZE = 224


@dataclass
class PI0Config(PreTrainedConfig):
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "float32"

    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    max_state_dim: int = 32
    max_action_dim: int = 32

    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    use_relative_actions: bool = False
    relative_exclude_joints: list = field(default_factory=lambda: ["gripper"])
    action_feature_names: list | None = None
    rtc_config: object | None = None

    image_resolution: tuple = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
    empty_cameras: int = 0

    normalization_mapping: dict = field(default_factory=dict)

    gradient_checkpointing: bool = False
    compile_model: bool = False
    compile_mode: str = "max-autotune"

    freeze_vision_encoder: bool = False
    train_expert_only: bool = False

    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

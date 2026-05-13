"""Minimal PreTrainedConfig stub for lingbotvla eval."""
from dataclasses import dataclass, field


@dataclass
class PreTrainedConfig:
    n_obs_steps: int = 1
    input_features: dict | None = field(default_factory=dict)
    output_features: dict | None = field(default_factory=dict)
    device: str | None = None
    use_amp: bool = False
    use_peft: bool = False
    push_to_hub: bool = True
    repo_id: str | None = None
    private: bool | None = None
    tags: list | None = None

    @classmethod
    def register_subclass(cls, name):
        def decorator(subclass):
            return subclass
        return decorator

"""Controller registry for VLM/VLA model controllers."""

from src.core.probe_core import Registry

CONTROLLER_REGISTRY = Registry("controller")

# Import triggers registration — add new controllers here
from src.controllers.qwen_vl_controller import QwenVLController  # noqa: F401
from src.controllers.openvla_controller import OpenVLAController  # noqa: F401
from src.controllers.act_controller import ACTController  # noqa: F401
from src.controllers.lingbot_vla_controller import LingBotVLAController  # noqa: F401
from src.controllers.lingbot_va_controller import LingBotVAController  # noqa: F401

# Pi-Zero requires separate conda env (openpi) — lazy import to avoid dependency crash
try:
    from src.controllers.pizero_controller import PiZeroController  # noqa: F401
except ImportError:
    pass

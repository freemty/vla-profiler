"""Controller registry for VLM/VLA model controllers."""

from src.core.probe_core import Registry

CONTROLLER_REGISTRY = Registry("controller")

# Import triggers registration — add new controllers here
from src.controllers.qwen_vl_controller import QwenVLController  # noqa: F401
from src.controllers.openvla_controller import OpenVLAController  # noqa: F401

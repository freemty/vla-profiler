"""
State management for model probing.

Provides:
- ``intervene_internal`` -- generic activation intervention (log / replace / apply)
- ``StoreMixin``         -- step_store / global_store lifecycle management
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from src.core.probe_core.controller import BaseController


# ------------------------------------------------------------------
# Generic intervention helper
# ------------------------------------------------------------------

def intervene_internal(
    module: nn.Module,
    input: Any,
    output: torch.Tensor,
    *,
    action: str,
    controller: Optional[BaseController] = None,
    layer_idx: Optional[int] = None,
    func: Optional[Callable] = None,
    new_value: Optional[torch.Tensor] = None,
    key: Optional[str] = None,
    scale: Any = 1.0,
) -> torch.Tensor:
    """
    Generic intervention on internal activations (forward-hook compatible).

    Supported actions
    -----------------
    - ``"log"``     : record *output* into ``controller.step_store``
    - ``"replace"`` : replace *output* with *new_value*
    - ``"apply"``   : apply *func* to *output*

    Returns the (possibly modified) output tensor.
    """
    if not torch.is_tensor(output):
        return output

    if action == "log":
        if controller is not None and layer_idx is not None and key is not None:
            store_key = f"{layer_idx}_{key}"
            log_output = output.detach().to("cpu", non_blocking=True)
            if store_key not in controller.step_store:
                controller.step_store[store_key] = [log_output]
            else:
                controller.step_store[store_key].append(log_output)
        return output

    if action == "replace":
        if new_value is None:
            raise ValueError("new_value is required for action 'replace'")
        replaced = output.clone()
        replaced[0] = new_value.to(device=output.device, dtype=output.dtype)
        return replaced

    if action == "apply":
        if func is None or not callable(func):
            raise ValueError("func callable is required for action 'apply'")
        applied = output.clone()
        applied[0] = func(output[0])
        return applied

    raise ValueError(f"Unknown action: {action}")


# ------------------------------------------------------------------
# Store lifecycle mixin
# ------------------------------------------------------------------

class StoreMixin:
    """
    Mixin that provides step-level and global activation storage.

    Attributes created by ``reset_state()``:
        step_store   -- dict accumulating tensors for the current step
        global_store -- dict accumulating tensors across steps
    """

    step_store: Dict[str, List[torch.Tensor]]
    global_store: Dict[str, List[torch.Tensor]]

    def reset_state(self) -> None:
        """Reset all stored state to empty."""
        self.step_store = {}
        self.global_store = {}

    def flush_step_store(self) -> None:
        """
        Merge current ``step_store`` into ``global_store``, then clear it.

        Each key's list is extended (not replaced) so that global_store
        accumulates across multiple steps.
        """
        for key in self.step_store:
            if key not in self.global_store:
                self.global_store[key] = self.step_store[key]
            else:
                self.global_store[key] = [
                    *self.global_store[key],
                    *self.step_store[key],
                ]
        self.step_store = {}

    def process_log_dict(
        self,
        log_dict: Dict[str, torch.Tensor],
        layer_idx: Optional[int] = None,
    ) -> None:
        """
        Insert entries from *log_dict* into ``step_store``.

        If *layer_idx* is provided the store key becomes ``"{layer_idx}_{key}"``.
        """
        for key, value in log_dict.items():
            store_key = f"{layer_idx}_{key}" if layer_idx is not None else key
            detached = value.detach().to("cpu", non_blocking=True)
            if store_key not in self.step_store:
                self.step_store[store_key] = [detached]
            else:
                self.step_store[store_key] = [*self.step_store[store_key], detached]

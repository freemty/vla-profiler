"""
Hook management utilities for model probing.

Provides path resolution, hook registration, and method patching for
attaching analysis/intervention hooks to PyTorch modules.

Extracted and generalized from rope2sink BaseController.HookManager.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from src.core.probe_core.controller import BaseController


class HookManager:
    """
    Stateless utility class for hook registration on PyTorch modules.

    All methods are classmethods/staticmethods -- no instance state.
    Subclass controllers embed this as a class attribute for convenient access.
    """

    # ------------------------------------------------------------------
    # Pre-built hook factories
    # ------------------------------------------------------------------

    class AnalysisHooks:
        """Factory for common analysis (logging) hooks."""

        @staticmethod
        def general(key: str, controller: BaseController, layer_idx: int):
            """Create a forward hook that logs output to controller.step_store."""
            from src.core.probe_core.state import intervene_internal

            def hook(module: nn.Module, input: Any, output: torch.Tensor):
                return intervene_internal(
                    module,
                    input,
                    output,
                    action="log",
                    controller=controller,
                    layer_idx=layer_idx,
                    key=key,
                )

            return hook

    class InterventionHooks:
        """Factory for common intervention hooks."""

        @staticmethod
        def general(action: str, **kwargs: Any):
            """Create a forward hook that applies an intervention action."""
            from src.core.probe_core.state import intervene_internal

            def hook(module: nn.Module, input: Any, output: torch.Tensor):
                return intervene_internal(
                    module, input, output, action=action, **kwargs
                )

            return hook

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    @classmethod
    def resolve_path(cls, block: nn.Module, path: str) -> nn.Module:
        """
        Resolve a nested attribute/index path relative to *block*.

        Example: ``resolve_path(block, 'ffn.net[0].proj')``
        """
        if path == "":
            return block
        current: Any = block
        for segment in path.split("."):
            if not segment:
                continue
            if "[" in segment and segment.endswith("]"):
                attr = segment[: segment.index("[")]
                idx_str = segment[segment.index("[") + 1 : -1]
                current = getattr(current, attr)
                current = current[int(idx_str)]
            else:
                current = getattr(current, segment)
        return current

    @classmethod
    def resolve_absolute(cls, model: nn.Module, abs_path: str) -> nn.Module:
        """
        Resolve an absolute dotted path starting from *model*.

        Example: ``resolve_absolute(model, 'transformer_blocks[3].ffn.net[-1]')``
        """
        current: Any = model
        for segment in abs_path.split("."):
            if not segment:
                continue
            if "[" in segment and segment.endswith("]"):
                attr = segment[: segment.index("[")]
                idx_str = segment[segment.index("[") + 1 : -1]
                current = getattr(current, attr)
                current = current[int(idx_str)]
            else:
                current = getattr(current, segment)
        return current

    # ------------------------------------------------------------------
    # Hook creation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def make_pre_apply_hook(
        apply_fn: Callable[[Any], Any],
    ) -> Callable:
        """Wrap *apply_fn* as a ``register_forward_pre_hook`` callback."""

        def pre_hook(module: nn.Module, inputs: Any) -> Any:
            if not inputs:
                return inputs
            return apply_fn(inputs)

        return pre_hook

    @staticmethod
    def make_post_apply_hook(
        apply_fn: Callable[[Any], Any],
    ) -> Callable:
        """Wrap *apply_fn* as a ``register_forward_hook`` callback."""

        def post_hook(module: nn.Module, inputs: Any, output: Any) -> Any:
            return apply_fn(output)

        return post_hook

    # ------------------------------------------------------------------
    # High-level registration helpers (absolute-path based)
    # ------------------------------------------------------------------

    @classmethod
    def register_variable_action_abs(
        cls,
        controller: BaseController,
        model: nn.Module,
        *,
        abs_path: str,
        point: str,
        apply_stage: str,
        func: Callable[[torch.Tensor], torch.Tensor],
        collection: str = "intervention_hooks",
        name_suffix: str = "",
    ) -> torch.utils.hooks.RemovableHook:
        """
        Register a pre- or post-forward hook on the module at *abs_path*.

        The hook handle is stored in ``getattr(controller, collection)``.
        """
        module = cls.resolve_absolute(model, abs_path)
        key = f"{abs_path}:{point}{(':' + name_suffix) if name_suffix else ''}"
        if apply_stage == "pre" or point == "input":
            hook = module.register_forward_pre_hook(cls.make_pre_apply_hook(func))
        else:
            hook = module.register_forward_hook(cls.make_post_apply_hook(func))
        getattr(controller, collection)[key] = hook
        return hook

    @classmethod
    def register_log_action_abs(
        cls,
        controller: BaseController,
        model: nn.Module,
        *,
        abs_path: str,
        point: str,
        key: str,
        index: Optional[int] = None,
        log_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        collection: str = "analysis_hooks",
        name_suffix: str = "",
    ) -> torch.utils.hooks.RemovableHook:
        """
        Register a logging hook that captures activations into ``controller.step_store``.

        If *index* is given, the hook extracts ``output[index]`` from tuple outputs.
        """
        store_key = key

        def log_identity(t: Any) -> Any:
            if controller.should_store(store_key):
                if isinstance(t, tuple):
                    t_ = t[index]
                else:
                    t_ = t
                if t_ is None:
                    return t
                out = t_.detach().to("cpu", non_blocking=True)
                if store_key not in controller.step_store:
                    controller.step_store[store_key] = [out]
                else:
                    controller.step_store[store_key].append(out)
            return t

        apply_stage = "pre" if point == "input" else "post"
        return cls.register_variable_action_abs(
            controller=controller,
            model=model,
            abs_path=abs_path,
            point=point,
            apply_stage=apply_stage,
            func=log_identity,
            collection=collection,
            name_suffix=name_suffix or f"log_{key}",
        )

    @classmethod
    def register_extractor_log_action_abs(
        cls,
        controller: BaseController,
        model: nn.Module,
        *,
        abs_path: str,
        key: str,
        extractor: Callable[[nn.Module, Tuple[Any, ...], Any], torch.Tensor],
        collection: str = "analysis_hooks",
        name_suffix: str = "",
    ) -> torch.utils.hooks.RemovableHook:
        """
        Register a hook that uses a custom *extractor* callable to pull data
        from a module's forward pass, then logs it into ``controller.step_store``.
        """
        module = cls.resolve_absolute(model, abs_path)
        store_key = key

        def hook_fn(
            mod: nn.Module, inputs: Tuple[Any, ...], output: Any
        ) -> Any:
            try:
                if controller.should_store(store_key):
                    data = extractor(mod, inputs, output)
                    out = data.detach().to("cpu", non_blocking=True)
                    if store_key not in controller.step_store:
                        controller.step_store[store_key] = [out]
                    else:
                        controller.step_store[store_key].append(out)
            except Exception:
                pass
            return output

        hook = module.register_forward_hook(hook_fn)
        key_name = name_suffix or f"log_extractor_{key}"
        getattr(controller, collection)[f"{abs_path}:extractor:{key_name}"] = hook
        return hook

    @classmethod
    def patch_method(
        cls,
        controller: BaseController,
        model: nn.Module,
        *,
        abs_path: str,
        method_name: str,
        wrapper_builder: Callable[[Callable[..., Any]], Callable[..., Any]],
        collection_originals: str = "intervention_original_processors",
    ) -> Callable:
        """
        Monkey-patch *method_name* on the module at *abs_path*.

        *wrapper_builder* receives the original method and returns the replacement.
        The original is stored in ``getattr(controller, collection_originals)``
        so it can be restored later.
        """
        module = cls.resolve_absolute(model, abs_path)
        original = getattr(module, method_name)
        wrapped = wrapper_builder(original)
        setattr(module, method_name, wrapped)
        getattr(controller, collection_originals)[module] = original
        return original

"""Attention overlay visualization task.

Computes attention from Q/K in global_store, maps to image space via
the controller's interpretability mixin, and renders heatmap overlays.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

import cv2
import numpy as np

from src.tasks import TASK_REGISTRY
from src.tasks.attention_task import _compute_attention_scores, _find_qk_keys
from src.viz.overlay_renderer import OverlayRenderer

logger = logging.getLogger(__name__)


def task_attention_overlay(
    controller: Any,
    save_dir: str,
    task_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Render attention overlays on original images."""
    output_dir = os.path.join(save_dir, "attention_overlay")
    os.makedirs(output_dir, exist_ok=True)

    cmap = task_config.get("cmap", "jet")
    overlay_alpha = task_config.get("overlay_alpha", 0.5)
    log_scale = task_config.get("log_scale", True)
    output_format = task_config.get("output_format", "both")
    multi_layer_strip = task_config.get("multi_layer_strip", True)
    gif_fps = task_config.get("gif_fps", 4)

    renderer = OverlayRenderer(
        cmap=cmap, overlay_alpha=overlay_alpha, log_scale=log_scale
    )

    qk_pairs = _find_qk_keys(controller.global_store)
    if not qk_pairs:
        logger.warning("No QK pairs found in global_store for overlay")
        return {"layers": [], "output_dir": output_dir}

    mappings = controller.get_token_spatial_mappings({})
    if not mappings:
        logger.warning("No token spatial mappings available")
        return {"layers": [], "output_dir": output_dir}

    original_images = getattr(controller, "_original_images", [])
    if not original_images:
        logger.warning("No original images stored on controller")
        return {"layers": [], "output_dir": output_dir}

    layer_indices = []
    all_layer_attn_maps: Dict[str, np.ndarray] = {}
    attention_data: Dict[str, Any] = {}

    for layer_idx, q_key, k_key in qk_pairs:
        q_tensors = controller.global_store[q_key]
        k_tensors = controller.global_store[k_key]

        q = q_tensors[0] if isinstance(q_tensors, list) else q_tensors
        k = k_tensors[0] if isinstance(k_tensors, list) else k_tensors

        attn_probs = _compute_attention_scores(q, k)
        attn_mean = attn_probs.mean(dim=(0, 1)).numpy()

        layer_indices = [*layer_indices, layer_idx]

        for img_idx, mapping in enumerate(mappings):
            if img_idx >= len(original_images):
                continue

            image = original_images[img_idx]
            heatmap = controller.map_attention_to_image(attn_mean, mapping)

            layer_name = f"layer_{layer_idx}"
            all_layer_attn_maps[layer_name] = heatmap

            overlay = renderer.render_overlay(image, heatmap)

            if output_format in ("frames", "both"):
                frame_name = f"{layer_name}_overlay"
                if len(mappings) > 1:
                    frame_name = f"{layer_name}_{mapping.image_key}_overlay"
                renderer.save_frames(
                    [(frame_name, overlay)], output_dir
                )

            attention_data[layer_name] = {
                "mean": float(heatmap.mean()),
                "max": float(heatmap.max()),
                "min": float(heatmap.min()),
                "std": float(heatmap.std()),
                "patch_shape": [mapping.patch_height, mapping.patch_width],
            }

    if multi_layer_strip and all_layer_attn_maps and original_images:
        strip = renderer.render_multi_layer_strip(
            original_images[0], all_layer_attn_maps
        )
        strip_path = os.path.join(output_dir, "multi_layer_strip.png")
        cv2.imwrite(strip_path, cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
        logger.info("Saved multi-layer strip to %s", strip_path)

    if output_format in ("gif", "both") and all_layer_attn_maps and original_images:
        gif_frames = []
        def _layer_sort_key(name: str) -> int:
            parts = name.split("_")
            return int(parts[-1]) if parts[-1].isdigit() else 0

        for layer_name in sorted(all_layer_attn_maps.keys(), key=_layer_sort_key):
            heatmap = all_layer_attn_maps[layer_name]
            overlay = renderer.render_overlay(original_images[0], heatmap)
            gif_frames = [*gif_frames, overlay]

        if gif_frames:
            gif_path = os.path.join(output_dir, "layers_sweep.gif")
            renderer.save_gif(gif_frames, gif_path, fps=gif_fps)

    data_path = os.path.join(output_dir, "attention_data.json")
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(attention_data, f, indent=2, ensure_ascii=False)
    logger.info("Saved attention data to %s", data_path)

    return {"layers": layer_indices, "output_dir": output_dir}


TASK_REGISTRY.register("attention_overlay", task_attention_overlay)

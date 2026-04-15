"""Attention heatmap overlay renderer.

Renders attention maps as colormap heatmaps overlaid on original images.
Supports multi-layer strip views and GIF animation output.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class OverlayRenderer:
    """Render attention heatmaps overlaid on original images.

    Args:
        cmap: Matplotlib colormap name (e.g. "jet", "viridis", "plasma").
        overlay_alpha: Blending alpha for the heatmap overlay (0=image, 1=heatmap).
        log_scale: Whether to apply log normalization to attention values.
    """

    def __init__(
        self,
        cmap: str = "jet",
        overlay_alpha: float = 0.5,
        log_scale: bool = True,
    ) -> None:
        self.cmap = cmap
        self.overlay_alpha = overlay_alpha
        self.log_scale = log_scale

    def render_overlay(
        self,
        image: np.ndarray,
        attn_map: np.ndarray,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Overlay attention heatmap on an image.

        Args:
            image: Original image (H, W, 3) uint8.
            attn_map: Attention map (H_patch, W_patch) float32.
            output_size: Optional (H, W) to resize the final output.

        Returns:
            Overlay image (H, W, 3) uint8.
        """
        h, w = image.shape[:2]

        attn = attn_map.astype(np.float32)
        if self.log_scale:
            attn = np.log(attn + 1e-9)

        vmin, vmax = attn.min(), attn.max()
        if vmax - vmin > 1e-8:
            attn = (attn - vmin) / (vmax - vmin)
        else:
            attn = np.zeros_like(attn)
        attn = np.clip(attn, 0.0, 1.0)

        attn_resized = cv2.resize(attn, (w, h), interpolation=cv2.INTER_LINEAR)

        colormap = matplotlib.colormaps[self.cmap]
        heatmap_rgba = colormap(attn_resized)[:, :, :3]
        heatmap = (heatmap_rgba * 255).astype(np.uint8)

        overlay = cv2.addWeighted(
            image, 1.0 - self.overlay_alpha,
            heatmap, self.overlay_alpha,
            0,
        )

        if output_size is not None:
            oh, ow = output_size
            overlay = cv2.resize(overlay, (ow, oh), interpolation=cv2.INTER_LINEAR)

        return overlay

    def render_token_type_border(
        self,
        vis: np.ndarray,
        token_type_attn: float,
        border_width: int = 15,
    ) -> np.ndarray:
        """Draw a magenta border whose intensity reflects token type attention.

        Args:
            vis: Image to draw on (H, W, 3) uint8. Will be copied.
            token_type_attn: Normalized attention value in [0, 1].
            border_width: Border thickness in pixels.

        Returns:
            Image with border drawn (H, W, 3) uint8.
        """
        result = vis.copy()
        intensity = int(255 * np.clip(token_type_attn, 0.0, 1.0))
        border_color = (intensity, 0, intensity)
        h, w = result.shape[:2]
        cv2.rectangle(result, (0, 0), (w - 1, h - 1), border_color, border_width)
        return result

    def render_multi_layer_strip(
        self,
        image: np.ndarray,
        attn_maps_by_layer: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Render side-by-side overlays for multiple layers.

        Args:
            image: Original image (H, W, 3) uint8.
            attn_maps_by_layer: Dict of layer_name -> attention map (H_patch, W_patch).

        Returns:
            Horizontal strip (H, W * num_layers, 3) uint8.
        """
        panels = []
        for layer_name in sorted(attn_maps_by_layer.keys()):
            attn_map = attn_maps_by_layer[layer_name]
            overlay = self.render_overlay(image, attn_map)

            labeled = overlay.copy()
            cv2.putText(
                labeled,
                layer_name,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            panels = [*panels, labeled]

        return np.concatenate(panels, axis=1)

    def save_frames(
        self,
        frames: List[Tuple[str, np.ndarray]],
        output_dir: str,
    ) -> List[str]:
        """Save named frames as individual PNG files.

        Args:
            frames: List of (name, image_array) tuples.
            output_dir: Directory to write files into.

        Returns:
            List of saved file paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        paths = []
        for name, frame in frames:
            path = os.path.join(output_dir, f"{name}.png")
            cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            paths = [*paths, path]
        logger.info("Saved %d frames to %s", len(paths), output_dir)
        return paths

    def save_gif(
        self,
        frames: List[np.ndarray],
        output_path: str,
        fps: int = 4,
    ) -> str:
        """Save frame sequence as an animated GIF.

        Args:
            frames: List of RGB image arrays (H, W, 3) uint8.
            output_path: Path for the output GIF file.
            fps: Frames per second.

        Returns:
            Path to the saved GIF.
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        pil_frames = [Image.fromarray(f) for f in frames]
        duration = int(1000 / fps)
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration,
            loop=0,
        )
        logger.info("Saved GIF to %s (%d frames, %d fps)", output_path, len(frames), fps)
        return output_path

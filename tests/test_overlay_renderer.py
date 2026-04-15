# tests/test_overlay_renderer.py
"""Unit tests for OverlayRenderer — no GPU required."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "core"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import tempfile


def test_render_overlay_shape():
    from src.viz.overlay_renderer import OverlayRenderer

    renderer = OverlayRenderer(cmap="jet", overlay_alpha=0.5, log_scale=False)
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    attn_map = np.random.rand(15, 20).astype(np.float32)

    result = renderer.render_overlay(image, attn_map)
    assert result.shape == (480, 640, 3)
    assert result.dtype == np.uint8


def test_render_overlay_log_scale():
    from src.viz.overlay_renderer import OverlayRenderer

    renderer = OverlayRenderer(cmap="viridis", overlay_alpha=0.4, log_scale=True)
    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    attn_map = np.random.rand(8, 8).astype(np.float32)

    result = renderer.render_overlay(image, attn_map)
    assert result.shape == (256, 256, 3)


def test_render_multi_layer_strip():
    from src.viz.overlay_renderer import OverlayRenderer

    renderer = OverlayRenderer()
    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    attn_maps = {
        "layer_0": np.random.rand(8, 8).astype(np.float32),
        "layer_7": np.random.rand(8, 8).astype(np.float32),
        "layer_14": np.random.rand(8, 8).astype(np.float32),
    }

    result = renderer.render_multi_layer_strip(image, attn_maps)
    assert result.shape[0] == 256
    assert result.shape[1] == 256 * 3
    assert result.shape[2] == 3


def test_render_token_type_border():
    from src.viz.overlay_renderer import OverlayRenderer

    renderer = OverlayRenderer()
    vis = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    result = renderer.render_token_type_border(vis, 0.8, border_width=10)
    assert result.shape == vis.shape


def test_save_frames(tmp_path):
    from src.viz.overlay_renderer import OverlayRenderer

    renderer = OverlayRenderer()
    frames = [
        ("layer_0", np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)),
        ("layer_7", np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)),
    ]
    output_dir = str(tmp_path / "frames")
    paths = renderer.save_frames(frames, output_dir)
    assert len(paths) == 2
    assert all(os.path.exists(p) for p in paths)


def test_save_gif(tmp_path):
    from src.viz.overlay_renderer import OverlayRenderer

    renderer = OverlayRenderer()
    frames = [
        np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
        np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
        np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
    ]
    gif_path = str(tmp_path / "test.gif")
    result_path = renderer.save_gif(frames, gif_path, fps=2)
    assert os.path.exists(result_path)

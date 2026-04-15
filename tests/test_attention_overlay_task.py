"""Unit tests for attention_overlay task — no GPU required."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "core"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


def test_task_registered():
    from src.tasks import TASK_REGISTRY
    import src.tasks.attention_overlay_task  # noqa: F401

    assert "attention_overlay" in TASK_REGISTRY


def test_task_produces_output(tmp_path):
    from src.tasks.attention_overlay_task import task_attention_overlay
    from src.interpretability.base_mixin import TokenSpatialMap
    import torch

    class MockController:
        def __init__(self):
            self.global_store = {}
            self._model_inputs_meta = {
                "image_grid_thw": [[1, 4, 5]],
                "vision_token_ranges": [(2, 22)],
                "image_sizes": [(128, 160)],
            }
            self._original_images = [
                np.random.randint(0, 255, (128, 160, 3), dtype=np.uint8)
            ]

            q = np.random.randn(1, 30, 128).astype(np.float32)
            k = np.random.randn(1, 30, 128).astype(np.float32)
            self.global_store["0_q_states"] = [torch.tensor(q)]
            self.global_store["0_k_states"] = [torch.tensor(k)]

        def get_token_spatial_mappings(self, inputs):
            return [
                TokenSpatialMap(
                    image_height=128,
                    image_width=160,
                    patch_height=4,
                    patch_width=5,
                    token_start_idx=2,
                    token_end_idx=22,
                    image_key="image_0",
                )
            ]

        def map_attention_to_image(self, attn_mean, mapping):
            from src.interpretability.base_mixin import BaseInterpretabilityMixin
            return BaseInterpretabilityMixin.map_attention_to_image(
                self, attn_mean, mapping
            )

    controller = MockController()
    save_dir = str(tmp_path)
    task_config = {
        "cmap": "jet",
        "overlay_alpha": 0.5,
        "log_scale": False,
        "output_format": "frames",
        "multi_layer_strip": False,
    }

    result = task_attention_overlay(controller, save_dir, task_config)

    overlay_dir = os.path.join(save_dir, "attention_overlay")
    assert os.path.isdir(overlay_dir)
    assert any(f.endswith(".png") for f in os.listdir(overlay_dir))
    assert "layers" in result

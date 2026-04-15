"""Unit tests for interpretability data structures and base mixin."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "core"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


def test_token_spatial_map_is_frozen():
    from src.interpretability.base_mixin import TokenSpatialMap

    m = TokenSpatialMap(
        image_height=480,
        image_width=640,
        patch_height=15,
        patch_width=20,
        token_start_idx=5,
        token_end_idx=305,
        image_key="image_0",
    )
    assert m.patch_height == 15
    assert m.patch_width == 20
    assert m.token_end_idx - m.token_start_idx == 300

    # Frozen — cannot mutate
    try:
        m.patch_height = 10
        assert False, "Should have raised"
    except AttributeError:
        pass


def test_token_type_enum():
    from src.interpretability.base_mixin import TokenType

    assert TokenType.VISUAL.value == "visual"
    assert TokenType.TEXT.value == "text"
    assert TokenType.ACTION.value == "action"
    assert TokenType.SPECIAL.value == "special"


def test_map_attention_to_image():
    from src.interpretability.base_mixin import (
        BaseInterpretabilityMixin,
        TokenSpatialMap,
    )

    mixin = BaseInterpretabilityMixin()
    mapping = TokenSpatialMap(
        image_height=480,
        image_width=640,
        patch_height=4,
        patch_width=5,
        token_start_idx=2,
        token_end_idx=22,
        image_key="image_0",
    )
    # Fake attention: (seq_q=30, seq_k=30), average already done over heads
    attn = np.random.rand(30, 30).astype(np.float32)
    result = mixin.map_attention_to_image(attn, mapping)

    assert result.shape == (4, 5)
    assert result.dtype == np.float32


def test_vlm_mixin_get_token_spatial_mappings():
    from src.interpretability.vlm_mixin import VLMInterpretabilityMixin

    mixin = VLMInterpretabilityMixin()
    mixin._model_inputs_meta = {
        "image_grid_thw": [[1, 15, 20]],
        "vision_token_ranges": [(2, 302)],
        "image_sizes": [(480, 640)],
    }

    mappings = mixin.get_token_spatial_mappings({})
    assert len(mappings) == 1
    m = mappings[0]
    assert m.patch_height == 15
    assert m.patch_width == 20
    assert m.token_start_idx == 2
    assert m.token_end_idx == 302
    assert m.image_height == 480
    assert m.image_width == 640
    assert m.image_key == "image_0"


def test_vlm_mixin_classify_token_types():
    from src.interpretability.vlm_mixin import VLMInterpretabilityMixin
    from src.interpretability.base_mixin import TokenType

    mixin = VLMInterpretabilityMixin()
    mixin._model_inputs_meta = {
        "vision_token_ranges": [(2, 12)],
    }

    types = mixin.classify_token_types(seq_len=20, inputs={})
    assert len(types) == 20
    assert types[0] == TokenType.SPECIAL
    assert types[1] == TokenType.SPECIAL
    assert types[2] == TokenType.VISUAL
    assert types[11] == TokenType.VISUAL
    assert types[12] == TokenType.TEXT
    assert types[19] == TokenType.TEXT


def test_vlm_mixin_multi_image():
    from src.interpretability.vlm_mixin import VLMInterpretabilityMixin

    mixin = VLMInterpretabilityMixin()
    mixin._model_inputs_meta = {
        "image_grid_thw": [[1, 10, 10], [1, 8, 12]],
        "vision_token_ranges": [(2, 102), (105, 201)],
        "image_sizes": [(320, 320), (256, 384)],
    }

    mappings = mixin.get_token_spatial_mappings({})
    assert len(mappings) == 2
    assert mappings[0].image_key == "image_0"
    assert mappings[1].image_key == "image_1"
    assert mappings[1].patch_height == 8
    assert mappings[1].patch_width == 12

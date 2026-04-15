"""VLM-specific interpretability mixin for Qwen2.5-VL."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from src.interpretability.base_mixin import (
    BaseInterpretabilityMixin,
    TokenSpatialMap,
    TokenType,
)

logger = logging.getLogger(__name__)


class VLMInterpretabilityMixin(BaseInterpretabilityMixin):
    """Interpretability mixin for VLM models (Qwen2.5-VL).

    Expects _model_inputs_meta to be populated during model_inference()
    with keys:
        - image_grid_thw: List of [t, h_patches, w_patches] per image
        - vision_token_ranges: List of (start_idx, end_idx) per image
        - image_sizes: List of (height, width) per image
    """

    _model_inputs_meta: Dict[str, Any]

    def get_token_spatial_mappings(
        self, inputs: Dict[str, Any]
    ) -> List[TokenSpatialMap]:
        """Build spatial mappings from stored processor metadata."""
        meta = self._model_inputs_meta
        grid_thw_list = meta["image_grid_thw"]
        ranges = meta["vision_token_ranges"]
        sizes = meta["image_sizes"]

        mappings = []
        for i, (grid_thw, (start, end), (img_h, img_w)) in enumerate(
            zip(grid_thw_list, ranges, sizes)
        ):
            _, h_patches, w_patches = grid_thw
            mappings = [
                *mappings,
                TokenSpatialMap(
                    image_height=img_h,
                    image_width=img_w,
                    patch_height=h_patches,
                    patch_width=w_patches,
                    token_start_idx=start,
                    token_end_idx=end,
                    image_key=f"image_{i}",
                ),
            ]
        return mappings

    def classify_token_types(
        self, seq_len: int, inputs: Dict[str, Any]
    ) -> List[TokenType]:
        """Classify tokens as VISUAL, TEXT, or SPECIAL.

        Tokens before the first vision range start are SPECIAL.
        Tokens within any vision range are VISUAL.
        All remaining tokens are TEXT.
        """
        meta = self._model_inputs_meta
        ranges = meta["vision_token_ranges"]

        types = [TokenType.TEXT] * seq_len

        # Mark special tokens before first vision range
        if ranges:
            first_start = ranges[0][0]
            for i in range(min(first_start, seq_len)):
                types[i] = TokenType.SPECIAL

        # Mark visual tokens
        for start, end in ranges:
            for i in range(start, min(end, seq_len)):
                types[i] = TokenType.VISUAL

        return types

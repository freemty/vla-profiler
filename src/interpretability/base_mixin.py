"""Base interpretability mixin and data structures."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List

import numpy as np


@dataclass(frozen=True)
class TokenSpatialMap:
    """Mapping from visual tokens in the sequence to image patch grid.

    Attributes:
        image_height: Original image height in pixels.
        image_width: Original image width in pixels.
        patch_height: Number of patch rows in the token grid.
        patch_width: Number of patch columns in the token grid.
        token_start_idx: Index of the first visual token in the full sequence.
        token_end_idx: Index one past the last visual token (exclusive).
        image_key: Identifier for the source image (e.g. "image_0").
    """

    image_height: int
    image_width: int
    patch_height: int
    patch_width: int
    token_start_idx: int
    token_end_idx: int
    image_key: str


class TokenType(Enum):
    """Classification of tokens in the input sequence."""

    VISUAL = "visual"
    TEXT = "text"
    ACTION = "action"
    SPECIAL = "special"


class BaseInterpretabilityMixin:
    """Mixin providing token-to-spatial-mapping for attention overlay.

    Subclasses implement get_token_spatial_mappings() and
    classify_token_types() for their specific model architecture.
    map_attention_to_image() is generic and usually not overridden.
    """

    @abstractmethod
    def get_token_spatial_mappings(
        self, inputs: Dict[str, Any]
    ) -> List[TokenSpatialMap]:
        """Return spatial mappings for each input image."""
        ...

    @abstractmethod
    def classify_token_types(
        self, seq_len: int, inputs: Dict[str, Any]
    ) -> List[TokenType]:
        """Classify each token position in the sequence."""
        ...

    def map_attention_to_image(
        self,
        attn_mean: np.ndarray,
        mapping: TokenSpatialMap,
    ) -> np.ndarray:
        """Map head-averaged attention to a 2D image-patch grid.

        Args:
            attn_mean: Attention matrix (seq_q, seq_k) averaged over heads.
            mapping: Token spatial mapping for one image.

        Returns:
            Attention heatmap shaped (patch_height, patch_width) as float32.
        """
        start = mapping.token_start_idx
        end = mapping.token_end_idx
        n_tokens = end - start

        # Column-mean over all query positions for visual key tokens
        # = how much each visual token is attended to on average
        visual_attn = attn_mean[:, start:end].mean(axis=0)

        expected = mapping.patch_height * mapping.patch_width
        if n_tokens != expected:
            # Truncate or pad to fit grid
            if n_tokens > expected:
                visual_attn = visual_attn[:expected]
            else:
                padded = np.zeros(expected, dtype=np.float32)
                padded[:n_tokens] = visual_attn
                visual_attn = padded

        heatmap = visual_attn.reshape(
            mapping.patch_height, mapping.patch_width
        ).astype(np.float32)
        return heatmap

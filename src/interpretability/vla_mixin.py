"""VLA-specific interpretability mixin — placeholder for Pi-Zero."""

from __future__ import annotations

from typing import Any, Dict, List

from src.interpretability.base_mixin import (
    BaseInterpretabilityMixin,
    TokenSpatialMap,
    TokenType,
)


class VLAInterpretabilityMixin(BaseInterpretabilityMixin):
    """Placeholder for VLA models (Pi-Zero, Octo, etc.).

    Will be implemented once model weights and architecture
    details are available. The interface is identical to
    VLMInterpretabilityMixin — only the internal mapping logic
    differs (vision + action token separation).
    """

    def get_token_spatial_mappings(
        self, inputs: Dict[str, Any]
    ) -> List[TokenSpatialMap]:
        raise NotImplementedError(
            "Pi-Zero token spatial mapping not yet implemented. "
            "Requires model access to determine vision token layout."
        )

    def classify_token_types(
        self, seq_len: int, inputs: Dict[str, Any]
    ) -> List[TokenType]:
        raise NotImplementedError(
            "Pi-Zero token classification not yet implemented. "
            "Requires model access to determine vision/action/text boundaries."
        )

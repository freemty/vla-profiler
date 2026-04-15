# Attention Overlay Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Map VLM attention weights back to input image space and render heatmap overlays on original images.

**Architecture:** Interpretability Mixin (token→spatial mapping) bolted onto controllers via multiple inheritance. A new `attention_overlay` task reads QKV from global_store, calls the mixin for spatial mapping, then feeds the result to an OverlayRenderer that produces overlay frames and GIFs.

**Tech Stack:** PyTorch (existing), OpenCV (cv2), matplotlib (colormaps), Pillow (GIF), Hydra (existing)

**Spec:** `docs/superpowers/specs/2026-04-15-attention-overlay-visualization-design.md`

---

## File Map

| File | Responsibility |
|------|---------------|
| `src/interpretability/__init__.py` | Package exports |
| `src/interpretability/base_mixin.py` | `TokenSpatialMap`, `TokenType`, `BaseInterpretabilityMixin` |
| `src/interpretability/vlm_mixin.py` | `VLMInterpretabilityMixin` — Qwen2.5-VL token-to-image mapping |
| `src/interpretability/vla_mixin.py` | `VLAInterpretabilityMixin` — Pi-Zero placeholder |
| `src/viz/__init__.py` | Package exports |
| `src/viz/overlay_renderer.py` | `OverlayRenderer` — heatmap overlay, strips, GIF |
| `src/tasks/attention_overlay_task.py` | `task_attention_overlay` — orchestration function |
| `configs/qwen_vl_7b/attention_overlay.yaml` | Hydra config for the new task |
| `src/controllers/qwen_vl_controller.py` | **Modify**: add mixin inheritance + store `_model_inputs` metadata |
| `src/run_tasks.py` | **Modify**: import `attention_overlay_task` for registry |
| `tests/test_interpretability.py` | Unit tests for mixin + data structures |
| `tests/test_overlay_renderer.py` | Unit tests for renderer |
| `tests/test_attention_overlay_task.py` | Unit tests for task orchestration |
| `tests/test_registry.py` | **Modify**: add `attention_overlay` to expected tasks |

---

### Task 1: Data Structures — TokenSpatialMap & TokenType

**Files:**
- Create: `src/interpretability/__init__.py`
- Create: `src/interpretability/base_mixin.py`
- Create: `tests/test_interpretability.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_interpretability.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/sum_young/code/projects/vlla && python -m pytest tests/test_interpretability.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.interpretability'`

- [ ] **Step 3: Implement data structures**

```python
# src/interpretability/__init__.py
"""Interpretability tools for VLM/VLA attention analysis."""

from src.interpretability.base_mixin import (
    BaseInterpretabilityMixin,
    TokenSpatialMap,
    TokenType,
)

__all__ = [
    "BaseInterpretabilityMixin",
    "TokenSpatialMap",
    "TokenType",
]
```

```python
# src/interpretability/base_mixin.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/sum_young/code/projects/vlla && python -m pytest tests/test_interpretability.py -v`
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/interpretability/__init__.py src/interpretability/base_mixin.py tests/test_interpretability.py
git commit -m "feat: add TokenSpatialMap, TokenType, BaseInterpretabilityMixin"
```

---

### Task 2: OverlayRenderer

**Files:**
- Create: `src/viz/__init__.py`
- Create: `src/viz/overlay_renderer.py`
- Create: `tests/test_overlay_renderer.py`

- [ ] **Step 1: Write the failing test**

```python
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
    # 3 layers side by side, each 256 wide
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/sum_young/code/projects/vlla && python -m pytest tests/test_overlay_renderer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.viz'`

- [ ] **Step 3: Implement OverlayRenderer**

```python
# src/viz/__init__.py
"""Visualization tools for attention analysis."""

from src.viz.overlay_renderer import OverlayRenderer

__all__ = ["OverlayRenderer"]
```

```python
# src/viz/overlay_renderer.py
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

        # Normalize attention values
        attn = attn_map.astype(np.float32)
        if self.log_scale:
            attn = np.log(attn + 1e-9)

        vmin, vmax = attn.min(), attn.max()
        if vmax - vmin > 1e-8:
            attn = (attn - vmin) / (vmax - vmin)
        else:
            attn = np.zeros_like(attn)
        attn = np.clip(attn, 0.0, 1.0)

        # Resize attention to image size
        attn_resized = cv2.resize(attn, (w, h), interpolation=cv2.INTER_LINEAR)

        # Apply colormap
        colormap = matplotlib.colormaps[self.cmap]
        heatmap_rgba = colormap(attn_resized)[:, :, :3]
        heatmap = (heatmap_rgba * 255).astype(np.uint8)

        # Blend
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

        Used for VLA models to show action/proprioception attention strength.

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
            attn_maps_by_layer: Dict of layer_name → attention map (H_patch, W_patch).

        Returns:
            Horizontal strip (H, W * num_layers, 3) uint8.
        """
        panels = []
        for layer_name in sorted(attn_maps_by_layer.keys()):
            attn_map = attn_maps_by_layer[layer_name]
            overlay = self.render_overlay(image, attn_map)

            # Add layer label
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
            # cv2 expects BGR, matplotlib produces RGB — convert
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/sum_young/code/projects/vlla && python -m pytest tests/test_overlay_renderer.py -v`
Expected: 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/viz/__init__.py src/viz/overlay_renderer.py tests/test_overlay_renderer.py
git commit -m "feat: add OverlayRenderer with heatmap overlay, strip, and GIF output"
```

---

### Task 3: VLMInterpretabilityMixin (Qwen2.5-VL)

**Files:**
- Create: `src/interpretability/vlm_mixin.py`
- Modify: `src/controllers/qwen_vl_controller.py`
- Modify: `tests/test_interpretability.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_interpretability.py`:

```python
def test_vlm_mixin_get_token_spatial_mappings():
    from src.interpretability.vlm_mixin import VLMInterpretabilityMixin

    mixin = VLMInterpretabilityMixin()

    # Simulate Qwen2.5-VL model_inputs metadata
    # image_grid_thw: (temporal=1, height_patches=15, width_patches=20)
    # vision tokens start after special tokens [BOS, vision_start] = idx 2
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
    assert types[0] == TokenType.SPECIAL   # BOS
    assert types[1] == TokenType.SPECIAL   # vision_start
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/sum_young/code/projects/vlla && python -m pytest tests/test_interpretability.py::test_vlm_mixin_get_token_spatial_mappings -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.interpretability.vlm_mixin'`

- [ ] **Step 3: Implement VLMInterpretabilityMixin**

```python
# src/interpretability/vlm_mixin.py
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
```

- [ ] **Step 4: Update `src/interpretability/__init__.py`**

```python
# src/interpretability/__init__.py
"""Interpretability tools for VLM/VLA attention analysis."""

from src.interpretability.base_mixin import (
    BaseInterpretabilityMixin,
    TokenSpatialMap,
    TokenType,
)
from src.interpretability.vlm_mixin import VLMInterpretabilityMixin

__all__ = [
    "BaseInterpretabilityMixin",
    "TokenSpatialMap",
    "TokenType",
    "VLMInterpretabilityMixin",
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/sum_young/code/projects/vlla && python -m pytest tests/test_interpretability.py -v`
Expected: 6 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/interpretability/vlm_mixin.py src/interpretability/__init__.py tests/test_interpretability.py
git commit -m "feat: add VLMInterpretabilityMixin for Qwen2.5-VL token mapping"
```

---

### Task 4: VLA Placeholder Mixin

**Files:**
- Create: `src/interpretability/vla_mixin.py`
- Modify: `src/interpretability/__init__.py`

- [ ] **Step 1: Create VLA placeholder**

```python
# src/interpretability/vla_mixin.py
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
```

- [ ] **Step 2: Update `src/interpretability/__init__.py`**

Add to imports:

```python
from src.interpretability.vla_mixin import VLAInterpretabilityMixin
```

Add `"VLAInterpretabilityMixin"` to `__all__`.

- [ ] **Step 3: Commit**

```bash
git add src/interpretability/vla_mixin.py src/interpretability/__init__.py
git commit -m "feat: add VLAInterpretabilityMixin placeholder for Pi-Zero"
```

---

### Task 5: Wire Mixin into QwenVLController

**Files:**
- Modify: `src/controllers/qwen_vl_controller.py`
- Modify: `tests/test_registry.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_registry.py`:

```python
def test_qwen_controller_has_interpretability_mixin():
    from src.controllers.qwen_vl_controller import QwenVLController
    from src.interpretability.base_mixin import BaseInterpretabilityMixin

    assert issubclass(QwenVLController, BaseInterpretabilityMixin)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/sum_young/code/projects/vlla && python -m pytest tests/test_registry.py::test_qwen_controller_has_interpretability_mixin -v`
Expected: FAIL — `QwenVLController` does not inherit from `BaseInterpretabilityMixin`

- [ ] **Step 3: Modify QwenVLController**

In `src/controllers/qwen_vl_controller.py`, make these changes:

**Add import** (after existing imports):
```python
from src.interpretability.vlm_mixin import VLMInterpretabilityMixin
```

**Change class declaration** from:
```python
class QwenVLController(BaseVLMController):
```
to:
```python
class QwenVLController(BaseVLMController, VLMInterpretabilityMixin):
```

**In `__init__`**, add after `super().__init__(...)`:
```python
        self._model_inputs_meta: Dict[str, Any] = {}
```

**In `model_inference()`**, add metadata capture after `model_inputs = model_inputs.to(model.device)` and before `generated_ids = model.generate(...)`:
```python
        # Store metadata for interpretability mixin
        self._model_inputs_meta = self._extract_model_inputs_meta(
            model_inputs, processor, messages, image_inputs
        )
```

**Add new method** to QwenVLController:
```python
    def _extract_model_inputs_meta(
        self,
        model_inputs: Any,
        processor: Any,
        messages: Any,
        image_inputs: Any,
    ) -> Dict[str, Any]:
        """Extract token mapping metadata from processor output.

        Reads image_grid_thw to determine patch grid dimensions and
        scans input_ids for vision token boundaries.
        """
        meta: Dict[str, Any] = {
            "image_grid_thw": [],
            "vision_token_ranges": [],
            "image_sizes": [],
        }

        if image_inputs is None or len(image_inputs) == 0:
            return meta

        # image_grid_thw from processor: (num_images, 3) — [t, h, w]
        if hasattr(model_inputs, "image_grid_thw") and model_inputs.image_grid_thw is not None:
            grid_thw = model_inputs.image_grid_thw.tolist()
            meta["image_grid_thw"] = grid_thw
        else:
            return meta

        # Image sizes from PIL images
        for img in image_inputs:
            if hasattr(img, "size"):
                w, h = img.size
                meta["image_sizes"] = [*meta["image_sizes"], (h, w)]
            else:
                meta["image_sizes"] = [*meta["image_sizes"], (0, 0)]

        # Find vision token ranges by scanning input_ids for vision placeholder tokens
        input_ids = model_inputs.input_ids[0].tolist()
        # Qwen2.5-VL uses token ID 151655 for <|image_pad|>
        # Scan for contiguous blocks of vision tokens
        vision_token_id = 151655
        ranges = []
        in_vision = False
        start = 0
        for idx, tid in enumerate(input_ids):
            if tid == vision_token_id and not in_vision:
                start = idx
                in_vision = True
            elif tid != vision_token_id and in_vision:
                ranges = [*ranges, (start, idx)]
                in_vision = False
        if in_vision:
            ranges = [*ranges, (start, len(input_ids))]

        meta["vision_token_ranges"] = ranges
        return meta
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/sum_young/code/projects/vlla && python -m pytest tests/test_registry.py -v`
Expected: All tests PASS (including new one)

- [ ] **Step 5: Commit**

```bash
git add src/controllers/qwen_vl_controller.py tests/test_registry.py
git commit -m "feat: wire VLMInterpretabilityMixin into QwenVLController"
```

---

### Task 6: attention_overlay_task

**Files:**
- Create: `src/tasks/attention_overlay_task.py`
- Create: `tests/test_attention_overlay_task.py`
- Modify: `src/run_tasks.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_attention_overlay_task.py
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

    # Build a mock controller
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

            # Fake Q/K tensors: (batch=1, seq=30, hidden=128)
            q = np.random.randn(1, 30, 128).astype(np.float32)
            k = np.random.randn(1, 30, 128).astype(np.float32)
            self.global_store["0_q_states"] = [__import__("torch").tensor(q)]
            self.global_store["0_k_states"] = [__import__("torch").tensor(k)]

        def get_token_spatial_mappings(self, inputs):
            meta = self._model_inputs_meta
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

    # Should have produced overlay files
    overlay_dir = os.path.join(save_dir, "attention_overlay")
    assert os.path.isdir(overlay_dir)
    assert any(f.endswith(".png") for f in os.listdir(overlay_dir))
    assert "layers" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/sum_young/code/projects/vlla && python -m pytest tests/test_attention_overlay_task.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement task**

```python
# src/tasks/attention_overlay_task.py
"""Attention overlay visualization task.

Computes attention from Q/K in global_store, maps to image space via
the controller's interpretability mixin, and renders heatmap overlays.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

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
    """Render attention overlays on original images.

    Reads Q/K from controller.global_store, computes attention scores,
    maps them to image space via the controller's interpretability mixin,
    and produces overlay frames and optionally a GIF.

    Args:
        controller: Controller with global_store and interpretability mixin.
        save_dir: Base output directory.
        task_config: Config with keys: cmap, overlay_alpha, log_scale,
            output_format ("frames"|"gif"|"both"), multi_layer_strip, gif_fps.

    Returns:
        Dict with keys: layers (list of layer indices), output_dir.
    """
    output_dir = os.path.join(save_dir, "attention_overlay")
    os.makedirs(output_dir, exist_ok=True)

    # Config
    cmap = task_config.get("cmap", "jet")
    overlay_alpha = task_config.get("overlay_alpha", 0.5)
    log_scale = task_config.get("log_scale", True)
    output_format = task_config.get("output_format", "both")
    multi_layer_strip = task_config.get("multi_layer_strip", True)
    gif_fps = task_config.get("gif_fps", 4)

    renderer = OverlayRenderer(
        cmap=cmap, overlay_alpha=overlay_alpha, log_scale=log_scale
    )

    # Get QK pairs from global_store
    qk_pairs = _find_qk_keys(controller.global_store)
    if not qk_pairs:
        logger.warning("No QK pairs found in global_store for overlay")
        return {"layers": [], "output_dir": output_dir}

    # Get token spatial mappings from the interpretability mixin
    mappings = controller.get_token_spatial_mappings({})
    if not mappings:
        logger.warning("No token spatial mappings available")
        return {"layers": [], "output_dir": output_dir}

    # Get original images
    original_images = getattr(controller, "_original_images", [])
    if not original_images:
        logger.warning("No original images stored on controller")
        return {"layers": [], "output_dir": output_dir}

    # Process each image
    layer_indices = []
    all_layer_attn_maps: Dict[str, np.ndarray] = {}
    attention_data: Dict[str, Any] = {}

    for layer_idx, q_key, k_key in qk_pairs:
        q_tensors = controller.global_store[q_key]
        k_tensors = controller.global_store[k_key]

        q = q_tensors[0] if isinstance(q_tensors, list) else q_tensors
        k = k_tensors[0] if isinstance(k_tensors, list) else k_tensors

        attn_probs = _compute_attention_scores(q, k)
        # Average over batch and heads → (seq_q, seq_k)
        attn_mean = attn_probs.mean(dim=(0, 1)).numpy()

        layer_indices = [*layer_indices, layer_idx]

        # For each image, map attention to spatial grid and render overlay
        for img_idx, mapping in enumerate(mappings):
            if img_idx >= len(original_images):
                continue

            image = original_images[img_idx]
            heatmap = controller.map_attention_to_image(attn_mean, mapping)

            layer_name = f"layer_{layer_idx}"
            all_layer_attn_maps[layer_name] = heatmap

            # Render overlay
            overlay = renderer.render_overlay(image, heatmap)

            # Save as frame
            if output_format in ("frames", "both"):
                frame_name = f"{layer_name}_overlay"
                if len(mappings) > 1:
                    frame_name = f"{layer_name}_{mapping.image_key}_overlay"
                renderer.save_frames(
                    [(frame_name, overlay)], output_dir
                )

            # Collect attention stats for JSON
            attention_data[layer_name] = {
                "mean": float(heatmap.mean()),
                "max": float(heatmap.max()),
                "min": float(heatmap.min()),
                "std": float(heatmap.std()),
                "patch_shape": [mapping.patch_height, mapping.patch_width],
            }

    # Multi-layer strip
    if multi_layer_strip and all_layer_attn_maps and original_images:
        strip = renderer.render_multi_layer_strip(
            original_images[0], all_layer_attn_maps
        )
        strip_path = os.path.join(output_dir, "multi_layer_strip.png")
        import cv2
        cv2.imwrite(strip_path, cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
        logger.info("Saved multi-layer strip to %s", strip_path)

    # GIF
    if output_format in ("gif", "both") and all_layer_attn_maps and original_images:
        gif_frames = []
        for layer_name in sorted(all_layer_attn_maps.keys()):
            heatmap = all_layer_attn_maps[layer_name]
            overlay = renderer.render_overlay(original_images[0], heatmap)
            gif_frames = [*gif_frames, overlay]

        if gif_frames:
            gif_path = os.path.join(output_dir, "layers_sweep.gif")
            renderer.save_gif(gif_frames, gif_path, fps=gif_fps)

    # Save attention data JSON
    data_path = os.path.join(output_dir, "attention_data.json")
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(attention_data, f, indent=2, ensure_ascii=False)
    logger.info("Saved attention data to %s", data_path)

    return {"layers": layer_indices, "output_dir": output_dir}


TASK_REGISTRY.register("attention_overlay", task_attention_overlay)
```

- [ ] **Step 4: Register in run_tasks.py**

In `src/run_tasks.py`, add after the existing task imports:

```python
import src.tasks.attention_overlay_task  # noqa: F401 — register task
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/sum_young/code/projects/vlla && python -m pytest tests/test_attention_overlay_task.py -v`
Expected: 2 tests PASS

- [ ] **Step 6: Update existing registry test**

In `tests/test_registry.py`, update `test_task_registry_has_all_tasks`:

```python
def test_task_registry_has_all_tasks():
    from src.tasks import TASK_REGISTRY
    expected = [
        "epd_profiling", "visual_text_attention", "sink_detection",
        "per_layer_stats", "attention_overlay",
    ]
    for task_name in expected:
        assert task_name in TASK_REGISTRY, f"Missing task: {task_name}"
```

- [ ] **Step 7: Run full test suite**

Run: `cd /Users/sum_young/code/projects/vlla && python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 8: Commit**

```bash
git add src/tasks/attention_overlay_task.py src/run_tasks.py tests/test_attention_overlay_task.py tests/test_registry.py
git commit -m "feat: add attention_overlay task with overlay rendering pipeline"
```

---

### Task 7: Hydra Config + Store Original Images

**Files:**
- Create: `configs/qwen_vl_7b/attention_overlay.yaml`
- Modify: `src/controllers/qwen_vl_controller.py` (store original images)

- [ ] **Step 1: Create Hydra config**

```yaml
# configs/qwen_vl_7b/attention_overlay.yaml
defaults:
  - /base
  - _self_

model_name: "${oc.env:HF_HOME,/data1/ybyang/huggingface}/Qwen/Qwen2.5-VL-7B-Instruct"
controller_name: "qwen_vl"

controller_config:
  mode: "analysis"
  store_layers:
    - 0
    - 7
    - 14
    - 21
    - 27
  store_type:
    - "self_q"
    - "self_k"

tasks:
  - "attention_overlay"

task_configs:
  attention_overlay:
    cmap: "jet"
    overlay_alpha: 0.5
    log_scale: true
    output_format: "both"
    multi_layer_strip: true
    gif_fps: 4

max_new_tokens: 128

inputs:
  - name: "single_image_overlay"
    messages:
      - role: "user"
        content:
          - type: "image"
            image: "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
          - type: "text"
            text: "Describe this image in detail."
```

- [ ] **Step 2: Store original images in QwenVLController**

In `src/controllers/qwen_vl_controller.py`, modify `model_inference()` to store original images for the overlay renderer. Add after `image_inputs, video_inputs = process_vision_info(messages)`:

```python
        # Store original images for overlay rendering
        self._original_images = []
        if image_inputs:
            for img in image_inputs:
                if hasattr(img, "size"):
                    img_array = np.array(img)
                    if img_array.ndim == 2:
                        img_array = np.stack([img_array] * 3, axis=-1)
                    self._original_images = [*self._original_images, img_array]
```

Add `import numpy as np` to the imports at the top of the file.

- [ ] **Step 3: Commit**

```bash
git add configs/qwen_vl_7b/attention_overlay.yaml src/controllers/qwen_vl_controller.py
git commit -m "feat: add attention_overlay Hydra config, store original images in controller"
```

---

### Task 8: End-to-End Smoke Test

This task validates the full pipeline on the server with real model weights.

**Files:** None new — verification only.

- [ ] **Step 1: Sync code to xdlab23**

```bash
bash scripts/sync_to_remote.sh
```

- [ ] **Step 2: Run on server**

```bash
ssh xdlab23_yang
cd /data1/ybyang/vlla
conda activate vit-probe
pip install opencv-python Pillow  # if not already installed
CUDA_VISIBLE_DEVICES=0 python -m src.run_tasks \
    --config-path ../configs \
    --config-name qwen_vl_7b/attention_overlay
```

- [ ] **Step 3: Download and inspect results**

```bash
bash scripts/download-results.sh
# Check output/Qwen2.5-VL-7B-Instruct/single_image_overlay/attention_overlay/
# Verify: layer_*_overlay.png files exist, multi_layer_strip.png, layers_sweep.gif
```

- [ ] **Step 4: Visual check**

Open `multi_layer_strip.png` — should show 5 panels (layers 0/7/14/21/27) with colored heatmap overlaid on the input image. Earlier layers should show more diffuse attention, later layers more focused.

Open `layers_sweep.gif` — should animate across layers.

- [ ] **Step 5: Run full local test suite one final time**

```bash
cd /Users/sum_young/code/projects/vlla && python -m pytest tests/ -v
```
Expected: All tests PASS.

- [ ] **Step 6: Final commit if any adjustments needed**

```bash
git add -A
git commit -m "fix: adjustments from e2e smoke test"
```

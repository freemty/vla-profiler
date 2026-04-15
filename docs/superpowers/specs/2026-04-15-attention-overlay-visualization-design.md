# Attention Overlay Visualization Design

## Goal

Implement attention visualization with image overlay for VLM/VLA models.
Map transformer attention weights back to input image space and render
heatmap overlays — enabling intuitive visual debugging and interpretability.

## Scope

- **This iteration**: Attention overlay for Qwen2.5-VL (VLM), with Pi-Zero (VLA) placeholder
- **Output style**: Real-time image overlay (JET heatmap on original image), per-layer frames, GIF animation
- **Future**: Gradient saliency (Integrated Gradients), SAE feature extraction + OOD detection

## Prior Art

| Source | What we take |
|--------|-------------|
| physical-AI-interpretability (villekuosmanen) | Overlay rendering approach (cv2.addWeighted), token type border (magenta proprio), GIF output |
| rope2sink (our sibling project) | Log-scale normalization, colormap system, render_attn_map(), academic styling |
| VLLA existing code | QKV hook capture, _compute_attention_scores(), TASK_REGISTRY, Hydra config |

## Architecture

```
Controller (hook QKV capture)
    → Interpretability Mixin (token-to-spatial mapping)
        → Task (attention computation + orchestration)
            → OverlayRenderer (heatmap overlay rendering)
```

### Layer 1: Interpretability Mixin

Mixin classes injected into controllers via multiple inheritance.
Provides the bridge between raw token-level attention and image-space visualization.

#### Data structures

```python
@dataclass(frozen=True)
class TokenSpatialMap:
    image_height: int          # original image H
    image_width: int           # original image W
    patch_height: int          # patch grid rows
    patch_width: int           # patch grid cols
    token_start_idx: int       # visual tokens start in sequence
    token_end_idx: int         # visual tokens end (exclusive)
    image_key: str             # input image identifier ("image_0")

class TokenType(Enum):
    VISUAL = "visual"
    TEXT = "text"
    ACTION = "action"          # VLA only
    SPECIAL = "special"        # BOS, EOS, pad
```

#### Interface

```python
class BaseInterpretabilityMixin:
    @abstractmethod
    def get_token_spatial_mappings(self, inputs) -> List[TokenSpatialMap]: ...

    @abstractmethod
    def classify_token_types(self, seq_len, inputs) -> List[TokenType]: ...

    def map_attention_to_image(self, attn_scores, mapping) -> np.ndarray:
        """Generic: extract visual token attention, reshape to (H_patch, W_patch).
        Subclasses normally do not need to override."""
        ...
```

#### VLM implementation (Qwen2.5-VL)

```python
class VLMInterpretabilityMixin(BaseInterpretabilityMixin):
    def get_token_spatial_mappings(self, inputs):
        # Read image_grid_thw from processor output
        # Compute token position ranges per image
        # Return List[TokenSpatialMap]

    def classify_token_types(self, seq_len, inputs):
        # <|vision_start|>...<|vision_end|> = VISUAL
        # Remaining = TEXT / SPECIAL
```

#### VLA placeholder (Pi-Zero)

```python
class VLAInterpretabilityMixin(BaseInterpretabilityMixin):
    def get_token_spatial_mappings(self, inputs):
        raise NotImplementedError("Pi-Zero mapping pending model access")

    def classify_token_types(self, seq_len, inputs):
        raise NotImplementedError
```

#### Integration

```python
class QwenVLController(BaseVLMController, VLMInterpretabilityMixin):
    ...  # existing code unchanged
```

### Layer 2: OverlayRenderer

New module: `src/viz/overlay_renderer.py`

```python
class OverlayRenderer:
    def __init__(self, cmap="jet", overlay_alpha=0.5, log_scale=True): ...
    def render_overlay(self, image, attn_map, output_size=None) -> np.ndarray: ...
    def render_token_type_border(self, vis, token_type_attn, border_width=15) -> np.ndarray: ...
    def render_multi_layer_strip(self, image, attn_maps_by_layer) -> np.ndarray: ...
    def save_frames(self, frames, output_dir) -> List[str]: ...
    def save_gif(self, frames, output_path, fps=4) -> str: ...
```

Rendering pipeline per frame:
1. `attn_map` (H_patch, W_patch) → cv2.resize to image size
2. Optional log-scale normalization (from rope2sink)
3. Apply matplotlib colormap → heatmap RGB
4. cv2.addWeighted(original, 1-alpha, heatmap, alpha, 0)
5. Optional token type border (VLA: action/proprio strength → magenta border width)

### Layer 3: attention_overlay_task

New module: `src/tasks/attention_overlay_task.py`

```python
def task_attention_overlay(controller, save_dir, task_config) -> Dict:
    # 1. Read QK from controller.global_store
    # 2. _compute_attention_scores(Q, K) → attn_probs
    # 3. Average over heads → (seq_q, seq_k)
    # 4. controller.get_token_spatial_mappings() → List[TokenSpatialMap]
    # 5. controller.map_attention_to_image(attn, mapping) → (H_patch, W_patch) per image
    # 6. Load original image from inputs
    # 7. OverlayRenderer.render_overlay() → overlay frame
    # 8. Optionally render_multi_layer_strip()
    # 9. Save frames / GIF
    # 10. Save attention_data.json (per-layer token importance values)
```

Registered as: `TASK_REGISTRY.register("attention_overlay", task_attention_overlay)`

## New Files

| File | Purpose | ~Lines |
|------|---------|--------|
| `src/interpretability/__init__.py` | Package init, exports | ~10 |
| `src/interpretability/base_mixin.py` | BaseInterpretabilityMixin, TokenSpatialMap, TokenType | ~80 |
| `src/interpretability/vlm_mixin.py` | VLMInterpretabilityMixin (Qwen2.5-VL) | ~120 |
| `src/interpretability/vla_mixin.py` | VLAInterpretabilityMixin (Pi-Zero placeholder) | ~40 |
| `src/viz/__init__.py` | Package init | ~5 |
| `src/viz/overlay_renderer.py` | OverlayRenderer class | ~200 |
| `src/tasks/attention_overlay_task.py` | Task function | ~150 |
| `configs/qwen_vl_7b/attention_overlay.yaml` | Hydra config | ~35 |

## Modified Files

| File | Change |
|------|--------|
| `src/controllers/qwen_vl_controller.py` | Add VLMInterpretabilityMixin to inheritance, store model_inputs metadata for token mapping |
| `src/controllers/base_vlm_controller.py` | Store original image references during inference for overlay rendering |

## Config

```yaml
# configs/qwen_vl_7b/attention_overlay.yaml
defaults:
  - /base
  - _self_

model_name: "${oc.env:HF_HOME}/Qwen/Qwen2.5-VL-7B-Instruct"
controller_name: "qwen_vl"

controller_config:
  mode: "analysis"
  store_type: ["self_q", "self_k"]
  store_layers: [0, 7, 14, 21, 27]
  store_phases: ["-1"]

tasks:
  - "attention_overlay"

task_configs:
  attention_overlay:
    layers: [0, 7, 14, 21, 27]
    cmap: "jet"
    overlay_alpha: 0.5
    log_scale: true
    output_format: "both"         # "frames" | "gif" | "both"
    multi_layer_strip: true
    gif_fps: 4

inputs:
  - name: "single_image"
    messages:
      - role: "user"
        content:
          - type: "image"
            image: "https://example.com/photo.jpg"
          - type: "text"
            text: "Describe this image in detail."
```

## Output Structure

```
output/qwen2.5-vl-7b/single_image/attention_overlay/
  layer_0_overlay.png
  layer_7_overlay.png
  layer_14_overlay.png
  layer_21_overlay.png
  layer_27_overlay.png
  multi_layer_strip.png
  layers_sweep.gif
  attention_data.json
```

## Key Design Decisions

1. **Mixin over wrapper**: Unlike physical-AI-interpretability's `ACTPolicyWithAttention` wrapper,
   we use mixin to integrate with existing controller hierarchy. No duplicate inference path.

2. **Reuse existing QKV capture**: No new hooks needed — the existing `_register_qk_hook` and
   `_compute_attention_scores` from `attention_task.py` work as-is.

3. **Token mapping from processor metadata**: Qwen2.5-VL's processor returns `image_grid_thw`
   which directly tells us the patch grid dimensions. We store this during `model_inference()`
   and expose it via the mixin.

4. **VLA extensibility**: `VLAInterpretabilityMixin` defines the same interface. Pi-Zero's
   vision/action token separation maps naturally to `classify_token_types()`. The renderer's
   `render_token_type_border()` handles action token attention display.

5. **Renderer independent of model**: `OverlayRenderer` takes raw numpy arrays, not model-specific
   objects. Any future model (Pi-Zero, Octo, OpenVLA) can use it once the mixin provides the mapping.

## Dependencies

- opencv-python (cv2) — already used in physical-AI-interpretability, likely in rope2sink too
- matplotlib — for colormaps (already a dependency)
- Pillow — for GIF generation
- No new heavy dependencies (no LeRobot, no SAE)

## Testing Strategy

1. Unit test: `TokenSpatialMap` construction from mock processor output
2. Unit test: `map_attention_to_image()` with synthetic attention tensor
3. Unit test: `OverlayRenderer.render_overlay()` with synthetic image + heatmap
4. Integration test: Full pipeline on xdlab23 with Qwen2.5-VL-7B + real image input
5. Visual verification: Compare overlay output against rope2sink's attention vis for sanity

# VLM/VLA Profiling & Attention Analysis Framework

**Date:** 2026-04-14
**Status:** Approved
**Scope:** Phase 1 — Qwen2.5-VL 7B profiling + attention analysis

---

## 1. Goal

Build a reusable framework for VLM/VLA inference characterization with two independent modes:
- **Profiling mode**: E/P/D (Encode-Prefill-Decode) timing measurement
- **Analysis mode**: Attention pattern extraction and analysis (visual-text attention, sink detection)

Phase 1 targets Qwen2.5-VL 7B. Phase 2 extends to VLA models (ACT, OpenVLA) with attention overlay visualization, inspired by [physical-AI-interpretability](https://github.com/villekuosmanen/physical-AI-interpretability).

## 2. Architecture

### 2.1 Shared Core Package (`model-probe-core`)

Extracted from rope2sink's BaseController as a thin, model-agnostic library (~500 lines). Consumed via git submodule by both rope2sink and vlla.

```
model-probe-core/
  probe_core/
    __init__.py
    controller.py      # BaseController + HookMode enum
    hooks.py           # HookManager (path resolution, log/replace/apply helpers)
    state.py           # StepStore/GlobalStore lifecycle management
    registry.py        # Controller registry + task registry pattern
```

**What's included (from rope2sink):**
- `HookMode` enum: `ANALYSIS | INTERVENE | PROFILING | BOTH | NONE`
- `HookManager`: `resolve_path()`, `resolve_absolute()`, `register_log_action_abs()`, `register_variable_action_abs()`
- `BaseController`: `step_store` -> `global_store` aggregation (`_between_steps()`), `should_store()` filtering, `reset_state()`, `remove_hooks()`
- `intervene_internal()`: log/replace/apply actions
- Registry pattern: dict-based dispatch for controllers and tasks

**What's NOT included:**
- `_get_rope_band_config()` (RoPE-specific)
- `inference_step_callback()` (denoising loop specific)
- `replace_processor()` / `_create_custom_processor()` (diffusers-specific)
- `update_timestep()` denoising logic
- All visualization utilities

**Key changes from rope2sink:**
- `store_time_ids` replaced with `store_phases: List[str]` (e.g., `["prefill", "decode", "encode"]`)
- `HookMode` extended with `PROFILING` value
- `should_store()` phase filtering instead of timestep filtering

### 2.2 vlla Project Structure

```
vlla/
  src/
    core/                        -> git submodule (model-probe-core)
    controllers/
      __init__.py                # CONTROLLER_REGISTRY
      base_vlm_controller.py     # VLM base class (extends core.BaseController)
      qwen_vl_controller.py      # Qwen2.5-VL implementation
    tasks/
      __init__.py                # TASK_REGISTRY
      profiling_task.py          # E/P/D timing analysis
      attention_task.py          # Visual-text attention pattern analysis
    utils/
      timing.py                  # PhaseTimer (torch.cuda.Event wrapper)
    run_tasks.py                 # Main entry point (Hydra)
  configs/
    base.yaml
    qwen_vl_7b/
      profiling.yaml
      attention.yaml
```

### 2.3 Inheritance Chain

```
probe_core.BaseController          # Model-agnostic hook lifecycle
  -> BaseVLMController             # VLM-specific: autoregressive inference, E/P/D phases
    -> QwenVLController            # Qwen2.5-VL model-specific details
```

Future extensions:
- `BaseVLAController -> ACTController` (Phase 2, robot policies)
- `BaseVLMController -> InternVLController` (additional VLM models)

## 3. BaseVLMController Interface

```python
class BaseVLMController(BaseController):
    """Base class for autoregressive VLM models."""
    
    PHASES = ["encode", "prefill", "decode"]
    
    # Subclass must implement
    @abstractmethod
    def get_vision_encoder(self) -> nn.Module: ...
    
    @abstractmethod
    def get_language_model(self) -> nn.Module: ...
    
    @abstractmethod
    def get_layer_blocks(self) -> List[nn.Module]: ...
    
    @abstractmethod
    def init_pipeline(cfg) -> Any: ...
    
    @abstractmethod
    def prepare_inputs(self, cfg) -> List[dict]: ...
    
    @abstractmethod
    def model_inference(self, pipeline, cfg, inputs) -> Any: ...
    
    # VLM common methods (implemented here)
    def register_profiling_hooks(self): ...
    def register_analysis_hooks(self): ...
```

### QwenVLController Implementation

```python
class QwenVLController(BaseVLMController):
    
    def get_vision_encoder(self):
        return self.pipeline.model.visual
    
    def get_language_model(self):
        return self.pipeline.model.model
    
    def get_layer_blocks(self):
        return list(self.pipeline.model.model.layers)
```

Migrated from rope2sink's existing `QwenVLController`: `init_pipeline()`, `prepare_inputs()`, `model_inference()`.

## 4. Profiling Mode: E/P/D Timing

### 4.1 Phase Decomposition for Qwen2.5-VL

```
Input (image + text)
  -> E: Vision Encode    model.visual(pixel_values)
  -> P: Prefill           LLM forward with [visual_tokens + text_tokens], seq_len > 1
  -> D: Decode            Autoregressive generation, seq_len == 1 per step
```

### 4.2 Instrumentation

`PhaseTimer` wraps `torch.cuda.Event` for precise GPU timing:

```python
class PhaseTimer:
    def __init__(self): ...
    def mark_start(self, phase: str): ...
    def mark_end(self, phase: str): ...
    def elapsed_ms(self, phase: str) -> float: ...
```

Hook insertion points:

| Phase | Hook Target | Method |
|-------|------------|--------|
| E (Encode) | `model.visual` | forward pre/post hook |
| P (Prefill) | `model.model.layers[0]` to `layers[-1]` | pre/post hook, `seq_len > 1` check |
| D (Decode) | Same layers | pre/post hook, `seq_len == 1` check |

Prefill vs Decode distinguished by **input sequence length**: prefill has `seq_len > 1` (all tokens parallel), decode has `seq_len == 1` (single token).

### 4.3 Output Format

```json
{
  "model": "Qwen2.5-VL-7B",
  "input": "single_image",
  "num_visual_tokens": 1225,
  "num_text_tokens": 42,
  "num_decode_tokens": 128,
  "timing_ms": {
    "encode": 18.3,
    "prefill": 45.7,
    "decode_total": 312.5,
    "decode_per_token": 2.44
  },
  "num_runs": 10,
  "std_ms": { "encode": 0.8, "prefill": 1.2, "decode_total": 5.3, "decode_per_token": 0.04 }
}
```

Multiple runs with warmup discarded. Mean + std reported.

### 4.4 Why Profiling Must Be Separate

Tensor collection (`to("cpu", non_blocking=True)`) and CUDA synchronization from analysis hooks corrupt timing measurements. Profiling mode only inserts lightweight CUDA event markers.

## 5. Analysis Mode: Attention Tasks

### 5.1 Task Registry

```python
TASK_REGISTRY = {
    # Phase 1: VLM
    "epd_profiling":          task_epd_profiling,
    "visual_text_attention":  task_visual_text_attn,
    "sink_detection":         task_sink_detection,
    "per_layer_stats":        task_per_layer_stats,
    
    # Phase 2: VLA (future)
    # "attention_overlay":    ...,
    # "sae_feature_extraction": ...,
}
```

### 5.2 Phase 1 Tasks

| Task | Collects | Analyzes |
|------|----------|----------|
| `epd_profiling` | CUDA timing events | E/P/D latency breakdown per input type |
| `visual_text_attention` | QK tensors at specified layers | visual->text / text->visual attention sparsity, top-k concentration |
| `sink_detection` | Attention received distribution | Sink token identification (visual vs text vs boundary) |
| `per_layer_stats` | Per-layer attention statistics | Layer-wise attention pattern evolution |

All tasks follow the rope2sink signature: `task_X(controller, save_dir, task_config)`.

### 5.3 Phase 2 Tasks (Future, VLA)

Inspired by physical-AI-interpretability:

| Capability | Their Approach | Our Approach |
|-----------|---------------|-------------|
| Attention overlay | ACT-specific, hardcoded | Generic BaseVLAController, any VLA |
| Model support | ACT only | ACT -> OpenVLA -> Pi-Zero (incremental) |
| SAE features | Standalone training | Task plugin in framework |
| Profiling | None | Built-in, same framework |

Extending to VLA requires only: new `BaseVLAController` -> `ACTController`, new tasks, new configs. No framework changes.

## 6. Configuration System

Hydra-based, consistent with rope2sink.

### 6.1 Base Config

```yaml
# configs/base.yaml
debug: false
device: "cuda:0"
seed: 42
base_output_path: ./outputs
num_warmup_runs: 3
num_benchmark_runs: 10
```

### 6.2 Experiment Configs

```yaml
# configs/qwen_vl_7b/profiling.yaml
defaults:
  - ../base

model_name: "Qwen2.5-VL-7B-Instruct"
controller_name: "QwenVLController"
controller_config:
  mode: profiling
  store_phases: ["encode", "prefill", "decode"]

tasks: ["epd_profiling"]
task_config:
  epd_profiling:
    output_format: "json"
    include_per_token_decode: true

inputs:
  - name: "text_only"
    messages: [{ role: "user", content: [{ type: "text", text: "Hello" }] }]
  - name: "single_image"
    messages: [{ role: "user", content: [
      { type: "image", image: "assets/demo.jpg" },
      { type: "text", text: "Describe this image." }
    ]}]
  - name: "multi_image"
    messages: [{ role: "user", content: [
      { type: "image", image: "assets/img1.jpg" },
      { type: "image", image: "assets/img2.jpg" },
      { type: "text", text: "Compare these images." }
    ]}]
```

Adding new input variants requires only config changes, no code modifications.

## 7. Output Structure

```
outputs/
  qwen_vl_7b/
    profiling/
      text_only/
        epd_timing.json
      single_image/
        epd_timing.json
      multi_image/
        epd_timing.json
      summary.json                # Cross-input comparison
    attention/
      single_image/
        visual_text_attention/
          layer_7_attn_heatmap.png
          sparsity_stats.json
        sink_detection/
          sink_tokens.json
          sink_distribution.png
```

## 8. Implementation Phases

### Phase 1 (Current Scope)
1. Create `model-probe-core` repo, extract thin core from rope2sink
2. Add as git submodule to both rope2sink and vlla
3. Implement `BaseVLMController` + `QwenVLController` in vlla
4. Implement `PhaseTimer` and `epd_profiling` task
5. Implement `visual_text_attention`, `sink_detection`, `per_layer_stats` tasks
6. Run on 3 input variants, validate results

### Phase 2 (Future)
7. Add `BaseVLAController` + `ACTController`
8. Attention overlay visualization (camera frame + heatmap)
9. SAE feature extraction task
10. Additional VLM models (InternVL, LLaVA)
11. Additional input variants (resolution, video)

## 9. Dependencies

- `torch` (CUDA events for timing, hooks API)
- `transformers` (Qwen2.5-VL model loading)
- `hydra-core` + `omegaconf` (configuration)
- `einops` (tensor reshaping)
- `easydict` (config convenience)
- `matplotlib` (visualization, analysis mode only)

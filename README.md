# vlla — VLM/VLA Real-Time Inference Profiling & Analysis

Hook-based profiling and attention analysis framework for Vision-Language Models (VLM) and Vision-Language-Action (VLA) models. Measures phase-level latency breakdown and captures attention patterns for interpretability research.

PhD research direction: VLM/VLA real-time inference systems.
Advisor: Hao Zhang (UCSD) — vLLM / FastVideo / Chatbot Arena.

## Status Overview

### Done

| Component | Status | Details |
|-----------|--------|---------|
| **Framework core** | Done | `model-probe-core` submodule (shared with rope2sink), BaseController / StoreMixin / HookManager / Registry |
| **VLM controller** | Done | BaseVLMController (E/P/D phases) -> QwenVLController (Qwen2.5-VL-7B) |
| **VLA controller** | Done | BaseVLAController (E/C/A phases) -> ACTController, LingBotVLAController, PiZeroController |
| **Profiling task** | Done | `epd_profiling` — CUDA event timing, median/P10/P90/P99/CV stats |
| **Attention tasks** | Done | `visual_text_attention`, `sink_detection`, `per_layer_stats` |
| **Attention overlay** | Done | Interpretability Mixin + OverlayRenderer (heatmap/strip/GIF) |
| **Timing validation** | Done | `timing_validation` — PhaseTimer vs torch.profiler cross-check |
| **PhaseTimer** | Done | CUDA event wrapper with CPU fallback, cumulative decode support |
| **Server deployment** | Done | xdlab23 scripts (sync/launch/download/monitor) |
| **Hydra configs** | Done | base.yaml + qwen_vl_7b/{profiling,attention,attention_overlay} + act/profiling + lingbot_vla_4b/profiling + pizero/profiling |
| **Survey** | Done | 180+ papers across 4 documents (landscape, recent-papers, va-world-models, va-world-models-web) |
| **Profiling survey** | Done | 8-system ML profiling comparison (vLLM/SGLang/FastVideo/TensorRT-LLM/DeepSpeed/Triton/llama.cpp/MLC LLM) |
| **Tests** | Done | Unit tests for timing, registry, attention task, overlay renderer, interpretability mixin, attention overlay task |
| **Viewer** | Done | Flask server + research presentation viewer (hub, slides, experiments) |

### Experiment Results

| Exp | Model | Key Finding |
|-----|-------|-------------|
| **exp01a** | Qwen2.5-VL-7B | Encode 253ms (58% of total), scales linearly with images. Decode ~18-21ms/tok, stable across modalities. |
| **exp01b** | Qwen2.5-VL-7B | Pos 2 (first visual patch) is universal attention sink (12-28x). Text->Visual Gini >0.91 (extreme sparsity). Layer 21 entropy lowest. |
| **exp02a** | ACT (LeRobot) | Total ~3ms, 850x faster than VLM. Encode (ResNet18) 80%, Action 20%. VLA latency lower bound. |
| **exp03a** | LingBot-VLA-4B | E=35.7ms / C=38.3ms / A=0.48ms (total 74.5ms ≈ 13Hz). 3B backbone 7x faster than 7B. Context ≈ Encode. Flow action head negligible. |
| **pi-zero** | Pi-Zero (3.2B) | Total 211ms ≈ 4.7Hz (224x224, bf16, 10 denoise steps, random weights). open-pi-zero backend. |

Hardware: RTX 5880 Ada 48GB on xdlab23, 10 benchmark runs + 3 warmup.

### TODO

| Item | Priority | Status | Notes |
|------|----------|--------|-------|
| **Attention overlay on server** | High | Not started | Config ready (`qwen_vl_7b/attention_overlay`), needs to run on xdlab23 with more input variants |
| **OpenVLA profiling** | High | Config ready | Controller done (`openvla_controller.py`), config done (`openvla_7b/`). Blocked: HF model weights download on server |
| **LingBot-VLA attention analysis** | High | Not started | 3B backbone attention patterns vs 7B (exp01b). Need attention config YAML |
| **OpenVLA attention analysis** | Medium | Config ready | `openvla_7b/attention.yaml` exists, depends on profiling run first |
| **Pi-Zero E/C/A breakdown** | Medium | Baseline done | Total 211ms/4.7Hz (random weights, bf16). Need pretrained weights for real profiling |
| **Gradient saliency** | Low | Not started | Extend interpretability framework beyond attention |
| **More VLA models** | Low | Not started | InternVL, Llava, etc. — controller stubs needed |

## Architecture

```
probe_core.BaseController           # Model-agnostic: hook lifecycle, StoreMixin, HookMode
  |
  +-> BaseVLMController             # VLM: E/P/D phases (Encode/Prefill/Decode), PhaseTimer
  |     +-> QwenVLController        # Qwen2.5-VL: model loading, QKV hooks, VLMInterpretabilityMixin
  |     +-> OpenVLAController       # OpenVLA (AR VLA): DINOv2+SigLIP -> Llama-2 7B
  |
  +-> BaseVLAController             # VLA: E/C/A phases (Encode/Context/Action)
        +-> ACTController           # ACT (LeRobot): ResNet18 -> CVAE -> action chunk
        +-> LingBotVLAController    # LingBot-VLA-4B: Qwen2.5-VL-3B + 10-step flow action head
        +-> PiZeroController        # Pi-Zero: SigLIP -> Gemma 2B + Gemma 300M Expert
```

Interpretability layer (multi-inheritance mixin):

```
BaseInterpretabilityMixin           # ABC: token spatial mapping, attention-to-image projection
  +-> VLMInterpretabilityMixin      # Qwen2.5-VL: image_grid_thw -> patch grid, <|image_pad|> scan
  +-> VLAInterpretabilityMixin      # Pi-Zero placeholder
```

Two strict modes:
- **Profiling** — CUDA event timing at phase boundaries (no tensor copies, no overhead)
- **Analysis** — QKV tensor capture for attention study (sink detection, sparsity, entropy)

## Project Structure

```
src/
  core/                 -> model-probe-core submodule (shared with rope2sink)
  controllers/          BaseVLMController, BaseVLAController, Qwen/OpenVLA/ACT/PiZero
  tasks/                profiling_task, attention_task, attention_overlay_task, validation_task
  interpretability/     Mixin system for attention-to-image mapping
  viz/                  OverlayRenderer (heatmap, strip, GIF)
  utils/                PhaseTimer
  run_tasks.py          Hydra entry point
vendor/
  open_pi_zero/         Vendored open-pi-zero model code (allenzren/open-pi-zero)
configs/                Hydra experiment configs (base + per-model)
scripts/                Server deployment and experiment scripts
survey/                 Literature survey (180+ papers, 4 documents)
exp/                    Experiment log and results
docs/knowhow/           Infrastructure, toolchain, debug solutions, runbooks
viewer/                 Flask server + survey dashboard
tests/                  Unit tests
pyproject.toml          uv project definition (torch>=2.5, transformers>=4.47)
```

## Getting Started

### Prerequisites

```bash
# Option A: uv (recommended for Pi-Zero / LingBot-VLA)
uv sync                     # core deps (torch, transformers, hydra, etc.)
uv sync --extra qwen        # + qwen-vl-utils

# Option B: conda (legacy, for Qwen/OpenVLA/ACT)
conda activate vit-probe
```

### Clone

```bash
git clone --recursive <repo-url>
cd vlla
```

### Run Locally (if GPU available)

```bash
export HF_HOME=/path/to/huggingface/cache
CUDA_VISIBLE_DEVICES=0 python -m src.run_tasks \
    --config-path ../configs --config-name qwen_vl_7b/profiling
```

### Run on Server (xdlab23)

```bash
# 1. Sync code
bash scripts/sync_to_remote.sh

# 2. Launch experiment (from local — runs on server via SSH)
bash scripts/run_remote.sh 0 qwen_vl_7b/profiling

# 3. Monitor (optional)
bash scripts/monitor_exp.sh exp01a

# 4. Download results
bash scripts/download-results.sh
```

See `docs/knowhow/runbooks/deploy-to-xdlab23.md` for first-time setup.

## Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/sync_to_remote.sh` | Git bundle sync to xdlab23 (GitHub blocked by firewall) | `bash scripts/sync_to_remote.sh` |
| `scripts/launch_exp.sh` | Launch experiment on server (run after SSH) | `bash scripts/launch_exp.sh <GPU> <CONFIG>` |
| `scripts/run_remote.sh` | One-command remote launch via SSH | `bash scripts/run_remote.sh <GPU> <CONFIG>` |
| `scripts/download-results.sh` | Rsync results from server to local | `bash scripts/download-results.sh [MODEL_DIR]` |
| `scripts/monitor_exp.sh` | Check experiment status (for /loop) | `bash scripts/monitor_exp.sh <EXP_ID>` |
| `scripts/run_local.sh` | Run experiment locally (for testing) | `bash scripts/run_local.sh <GPU> <CONFIG>` |
| `scripts/run_viewer.sh` | Start Flask viewer server | `bash scripts/run_viewer.sh [PORT]` |
| `scripts/run_tests.sh` | Run test suite with coverage | `bash scripts/run_tests.sh` |

## Configuration

Hydra-based. Add new experiments by creating YAML under `configs/`:

```yaml
# configs/your_model/profiling.yaml
defaults:
  - /base
  - _self_

model_name: "${oc.env:HF_HOME,...}/your/model-path"
controller_name: "your_model"
controller_config:
  mode: "profiling"       # or "analysis"
tasks:
  - "epd_profiling"
inputs:
  - name: "single_image"
    messages:
      - role: "user"
        content:
          - type: "image"
            image: "path/or/url"
          - type: "text"
            text: "Describe this image."
```

Available configs:

| Config | Model | Tasks |
|--------|-------|-------|
| `qwen_vl_7b/profiling` | Qwen2.5-VL-7B | E/P/D timing |
| `qwen_vl_7b/attention` | Qwen2.5-VL-7B | Attention analysis (sink, sparsity, entropy) |
| `qwen_vl_7b/attention_overlay` | Qwen2.5-VL-7B | Attention heatmap overlay on input images |
| `act/profiling` | ACT (LeRobot) | E/A timing |
| `openvla_7b/profiling` | OpenVLA-7B | E/P/D timing |
| `openvla_7b/attention` | OpenVLA-7B | Attention analysis |
| `lingbot_vla_4b/profiling` | LingBot-VLA-4B | E/C/A timing (flow VLA, requires uv env) |
| `pizero/profiling` | Pi-Zero (3.2B) | E/C/A timing (open-pi-zero backend, vendored) |

## Adding New Models

1. Create `src/controllers/your_model_controller.py`
2. Extend `BaseVLMController` (for autoregressive VLM/VLA) or `BaseVLAController` (for non-AR VLA)
3. Implement: `get_vision_encoder()`, `get_language_model()`, `get_layer_blocks()`, `init_pipeline()`, `prepare_inputs()`, `model_inference()`
4. Register: `CONTROLLER_REGISTRY.register("your_model", YourModelController)`
5. Add Hydra config YAML under `configs/your_model/`

## Server (xdlab23)

| Item | Value |
|------|-------|
| SSH | `ssh xdlab23_yang` (port 66) |
| Path | `/data1/ybyang/vlla` |
| Conda | `vit-probe` (shared with rope2sink, Qwen/OpenVLA/ACT) |
| uv venv | `.venv/` (Pi-Zero/LingBot-VLA, torch 2.5+cu121, Python 3.12) |
| GPUs | 8x RTX 5880 Ada 48GB |
| HF cache | `/data1/ybyang/huggingface` |
| Code sync | Git bundle (GitHub blocked by firewall) |

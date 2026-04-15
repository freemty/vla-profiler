# vlla — VLM/VLA Inference Profiling & Analysis

A hook-based profiling and attention analysis framework for Vision-Language Models (VLM) and Vision-Language-Action (VLA) models. Measures E/P/D (Encode-Prefill-Decode) latency breakdown and captures attention patterns for interpretability research.

Built on a shared core (`model-probe-core`) extracted from [rope2sink](https://github.com/freemty/vit-probe).

## Key Results (exp01a)

Qwen2.5-VL-7B on RTX 5880 Ada (48GB), 10 benchmark runs:

| Input | Encode | Prefill | Decode/token | Total |
|-------|--------|---------|-------------|-------|
| text only | — | 20ms | 18ms | 146ms |
| single image | 253ms | 156ms | 18.6ms | 2.8s |
| 2 images | 541ms | 332ms | 21ms | 1.1s |

Vision encoding scales linearly with image count. Decode per-token is stable across modalities.

## Architecture

```
probe_core.BaseController       # Model-agnostic hook lifecycle (shared with rope2sink)
  -> BaseVLMController          # E/P/D phase timing, layer hook registration
    -> QwenVLController         # Qwen2.5-VL model loading, inference, QKV capture
```

Two independent modes:
- **Profiling** — CUDA event timing at E/P/D boundaries (no tensor copies, no overhead)
- **Analysis** — QKV tensor capture for attention pattern study (sink detection, sparsity, entropy)

## Getting Started

### Prerequisites

```bash
conda activate vit-probe   # or any env with:
# PyTorch 2.9+ (CUDA), transformers, hydra-core, omegaconf, einops, easydict, qwen-vl-utils
```

### Clone

```bash
git clone --recursive <repo-url>
cd vlla
```

### Run Profiling

```bash
export HF_HOME=/path/to/huggingface/cache
CUDA_VISIBLE_DEVICES=0 python -m src.run_tasks \
    --config-path ../configs --config-name qwen_vl_7b/profiling
```

Output: `output/Qwen2.5-VL-7B-Instruct/{text_only,single_image,multi_image}/epd_profiling/epd_timing.json`

### Run Attention Analysis

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.run_tasks \
    --config-path ../configs --config-name qwen_vl_7b/attention
```

Output: `output/Qwen2.5-VL-7B-Instruct/*/{visual_text_attention,sink_detection,per_layer_stats}/`

### Server Deployment (xdlab23)

```bash
bash scripts/sync_to_remote.sh                          # sync code
bash scripts/launch_exp.sh 0 qwen_vl_7b/profiling       # run on GPU 0
bash scripts/download-results.sh                         # download results
```

See `docs/knowhow/runbooks/deploy-to-xdlab23.md` for details.

## Configuration

Hydra-based. Add new experiments by creating YAML files under `configs/`:

```yaml
# configs/qwen_vl_7b/profiling.yaml
defaults:
  - /base
  - _self_

model_name: "${oc.env:HF_HOME,...}/Qwen/Qwen2.5-VL-7B-Instruct"
controller_name: "qwen_vl"
controller_config:
  mode: "profiling"       # or "analysis"
tasks:
  - "epd_profiling"       # timing breakdown
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

## Adding New Models

1. Create `src/controllers/your_model_controller.py`
2. Extend `BaseVLMController` (VLM) or `BaseVLAController` (VLA, future)
3. Implement: `get_vision_encoder()`, `get_language_model()`, `get_layer_blocks()`, `init_pipeline()`, `prepare_inputs()`, `model_inference()`
4. Register in `CONTROLLER_REGISTRY`
5. Add config YAML

## Project Structure

```
src/
  core/              -> model-probe-core submodule
  controllers/       BaseVLMController, QwenVLController
  tasks/             profiling_task, attention_task
  utils/             PhaseTimer
  run_tasks.py       Hydra entry point
configs/             Experiment configs
scripts/             Server deployment (sync, launch, download)
survey/              Literature survey (180+ papers)
exp/                 Experiment log
docs/knowhow/        Infrastructure, toolchain, debug solutions, runbooks
```

## Research Context

PhD research direction exploration: VLM/VLA real-time inference systems.
Advisor: Hao Zhang (UCSD) — author of vLLM, FastVideo, Chatbot Arena.

The core question: how to make VLM/VLA run under real-time constraints for robotics (10ms control loops), autonomous driving, and interactive AI?

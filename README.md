# vla-profiler — VLM/VLA Real-Time Inference Profiling

Phase-level latency profiling for Vision-Language-Action models. Measures E/C/A breakdown (Encode / Context / Action) across 10 models spanning 5 paradigms.

## Key Findings

| Paradigm | Model | Total | Hz | Bottleneck |
|----------|-------|-------|-----|-----------|
| Single-forward | ACT | 3ms | 300 | Encode 80% |
| VLM + flow head | LingBot-VLA (3B) | 74ms | 13 | E ≈ C, A free |
| VLM + OFT MLP | StarVLA-OFT (3B) | 63ms | 16 | E+C 99.8% |
| VLM + OFT MLP | OpenVLA-OFT (7B) | 109ms | 9 | C (7B prefill) 84% |
| Dual-stream flow | Pi-Zero (2.7B) | 200ms | 5 | Action Expert 82% |
| Pure DiT (VA) | NitroGen (174M) | 7.2ms/step | 56 @k=1 | Linear in params |
| WAM skip-imagination | Fast-WAM (5B) | 407ms | 2.5 | Action DiT 89% |
| Monolithic DiT | Cosmos Policy (2B) | 659ms | 1.5 | DiT denoise 90%+ |
| Full WAM | LingBot-VA (5B) | 2518ms | 0.4 | Video+Action 93% |

Hardware: RTX 5880 Ada 48GB, bf16, CUDA event timing, warmup=15, iter=20, median.

**Two acceleration paths identified:**
- **Path A**: Compress Action DiT (FastVideo STA / step distillation / caching)
- **Path A'**: Kill action head entirely (OFT MLP) + compress backbone (flash-attn / quantization)

## Architecture

```
probe_core.BaseController
  ├── BaseVLMController (E/P/D)     → QwenVLController, OpenVLAController
  └── BaseVLAController (E/C/A)     → ACT, LingBot-VLA, LingBot-VA, NitroGen,
                                       PiZero, OpenVLA-OFT, StarVLA-OFT
```

Two modes:
- **Profiling** — CUDA event timing at phase boundaries, zero overhead
- **Analysis** — QKV tensor capture for attention study

## Quick Start

```bash
# Clone
git clone --recursive <repo-url> && cd vlla

# Install
uv sync  # or: conda activate vit-probe

# Run profiling
export HF_HOME=/path/to/huggingface
CUDA_VISIBLE_DEVICES=0 python -m src.run_tasks \
    --config-path ../configs --config-name pizero/profiling

# Run on xdlab23
bash scripts/sync_to_remote.sh
bash scripts/launch_exp.sh 0 pizero/profiling
bash scripts/download-results.sh
```

## Project Structure

```
src/controllers/    9 model controllers (VLM + VLA + OFT)
src/tasks/          profiling, attention, validation tasks
src/utils/          PhaseTimer (CUDA event + CPU fallback)
configs/            Hydra configs per model × task
scripts/            server deploy, experiment launch, LIBERO eval
survey/             180+ paper survey (landscape, deep-dives)
exp/                experiment specs + results
slides/             presentation deck (Swiss Knife design)
viewer/             Flask dashboard
```

## Adding Models

1. Create `src/controllers/your_controller.py` extending `BaseVLAController`
2. Implement: `get_vision_encoder()`, `get_action_head()`, `init_pipeline()`, `prepare_inputs()`, `model_inference()`
3. Register: `CONTROLLER_REGISTRY.register("name", YourController)`
4. Add Hydra config: `configs/your_model/profiling.yaml`

## Server

| | |
|---|---|
| SSH | `ssh xdlab23_yang` (port 66) |
| GPUs | 8× RTX 5880 Ada 48GB |
| Sync | Git bundle (GitHub blocked by firewall) |
| Envs | `vit-probe` (default), `fastwam` (WAM), `.venvs/pizero` (Pi-Zero) |

## Slides

Live: [freemty.github.io/slides/vla-design-space.html](https://freemty.github.io/slides/vla-design-space.html)

## Docs

See `CLAUDE.md` for full index. Key docs:
- `exp/summary.md` — experiment flight recorder
- `docs/TODO.md` — prioritized task backlog
- `docs/hao-meeting-prep-v2.md` — meeting outline
- `survey/papers/` — 25+ deep-dive survey notes

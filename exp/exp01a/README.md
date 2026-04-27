# exp01a — Qwen2.5-VL-7B E/P/D Profiling

## Motivation

Establish the pure-VLM latency baseline on RTX 5880 Ada. All downstream VLA
comparisons (exp03a, exp05a, exp07a) reference these numbers.

## Model

- `Qwen/Qwen2.5-VL-7B-Instruct`, bf16, `sdpa`
- Backbone: Qwen2.5 7B LLM + Qwen2-VL ViT

## Method

Hydra-driven, three input profiles:

| Profile | Config | What varies |
|---------|--------|-------------|
| text_only | `configs/qwen_vl_7b/profiling.yaml` override `prompt=text_only` | No image, P/D only |
| single_image | `prompt=single_image` | 1 image |
| multi_image | `prompt=multi_image` | 3 images |

PhaseTimer separates Encode (ViT forward) from Prefill (first LLM forward with
KV build) from Decode (per-token).

## Hardware

- Server: xdlab23, GPU 0 (RTX 5880 Ada 48GB)
- Env: `vit-probe` conda

## Canonical results (rerun, 2026-04-26, warmup=15, iter=20, median)

| Profile | Encode (E) | Prefill (P) | Decode (D, per-token) |
|---------|-----------|-------------|-----------------------|
| text_only | — | 20 ms | 18 ms |
| single_image | 253 ms | 156 ms | 18.6 ms |
| multi_image (3 img) | 541 ms | 332 ms | 21 ms |

**Key finding:** Encode scales ~linearly with image count, Decode per-token
stable. These are the anchor numbers for exp08 roofline analysis (D = HBM-BW
bound, P = tensor-core bound).

## Commands

```bash
# On xdlab23 (from /data1/ybyang/vlla)
bash scripts/launch_exp.sh 0 qwen_vl_7b/profiling

# Local: pull results
bash scripts/download-results.sh Qwen_Qwen2.5-VL-7B-Instruct
```

## Results

- `results_rerun3/` — canonical warmup=15 rerun (used by all downstream comparisons)
  - `text_only.json`, `single_image.json`, `multi_image.json`

Earlier runs (warmup=3) systematically underestimated by ~18% due to GPU
power bimodal — see Engineering Lessons #52 in project-skill.

## Status

- [x] text_only / single_image / multi_image profiling
- [x] warmup=15 rerun (canonical)
- [x] Results downloaded to local (`results_rerun3/`)

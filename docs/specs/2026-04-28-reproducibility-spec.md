# Reproducibility Spec v0.9.0

**Date:** 2026-04-28
**Author:** @freemty
**Status:** Active contract — all subsequent experiment runs MUST match these settings.

This document is the single source of truth for the **official** inference configuration of each profiled model. Every param table below specifies what the real model uses in its canonical evaluation. If a profiling run deviates from this spec, it must be flagged as non-official.

**Scope:** 7 models profiled in exp01–exp07. Latency realignment (exp09) uses this spec as its contract.

---

## ACT (Zhao 2023)

Lightweight CVAE policy. No iterative denoising.

| param | required value | source | verify command |
|-------|---------------|--------|----------------|
| ckpt | `tonyzhaozh/act` (ALOHA pretrained, ~40MB) | HuggingFace hub | `huggingface-cli download tonyzhaozh/act` |
| backbone | ResNet18 | paper S4 | — |
| transformer dim | 512 | paper S4 | `python -c "from policy import ACTPolicy; print(ACTPolicy.default_config())"` |
| encoder layers | 4 | paper S4 | config dump |
| decoder layers | 7 | paper S4 | config dump |
| action_chunk | 100 steps | paper S4 | config dump |
| denoise steps | N/A (CVAE one-shot) | — | — |
| benchmark | ALOHA sim (insertion / transfer_cube) | paper S6 | **deferred to exp10** |

---

## LingBot-VLA (Robbyant 2025)

Flow-matching VLA on frozen Qwen2.5-VL-3B backbone. **Already using real weights.**

| param | required value | source | verify command |
|-------|---------------|--------|----------------|
| ckpt | `Robbyant/lingbot-vla-4b` (ModelScope) | ModelScope model card | `ls /data1/ybyang/modelscope/Robbyant/lingbot-vla-4b/` |
| ckpt on disk | `/data1/ybyang/modelscope/Robbyant/lingbot-vla-4b` | exp03a verified | `md5sum` on safetensors |
| backbone | Qwen2.5-VL-3B frozen | model card | — |
| action head | 10-step flow matching MLP | model card | config dump |
| denoise steps | 10 | model card | `denoise_steps: 10` in config |
| benchmark | LIBERO-4 suite | author eval | `libero_eval.py` |

---

## NitroGen (NVIDIA 2025)

500M DiT-based VA model. SigLIP vision encoder + DiT action generator.

| param | required value | source | verify command |
|-------|---------------|--------|----------------|
| ckpt | `nvidia/NitroGen/ng.pt` (500M) | HuggingFace | `ls -lh /data1/ybyang/huggingface/nvidia/NitroGen/ng.pt` |
| ckpt on disk | `/data1/ybyang/huggingface/nvidia/NitroGen/ng.pt` | demo_reproduce verified | `md5sum ng.pt` |
| vision_hidden_size | **1024** | ng.pt weight shape | `python -c "import torch; s=torch.load('ng.pt',map_location='cpu'); print([k for k in s if 'vision' in k])"` |
| dit_num_layers | 12 | ng.pt | weight key count |
| dit_num_heads | 16 | ng.pt | — |
| dit_head_dim | 64 | ng.pt | — |
| vl_num_layers | 4 | ng.pt | — |
| vl_num_heads | 16 | ng.pt | — |
| vl_head_dim | 64 | ng.pt | — |
| action_dim | **25** | ng.pt output layer shape | `python -c "import torch; s=torch.load('ng.pt',map_location='cpu'); print(s['action_head.weight'].shape)"` |
| action_horizon | 16 | ng.pt | — |
| denoise steps | 16 (default); sweep k=1,2,4,8,16 | paper | — |
| benchmark | NVIDIA internal VA (no public harness) | paper | **latency only — no task success eval possible** |

---

## Pi-Zero (Physical Intelligence 2024)

Dual-stream flow VLA: PaliGemma backbone + Gemma 300M Action Expert.

| param | required value | source | verify command |
|-------|---------------|--------|----------------|
| ckpt | `pi0-base` from `gs://openpi-assets/checkpoints/pi0_libero/` or HF mirror | openpi repo | `ls /data1/ybyang/huggingface/physical-intelligence/pi0-base/` |
| ckpt on disk | `/data1/ybyang/huggingface/physical-intelligence/pi0-base` | **needs download** | `md5sum` on shards |
| backbone | PaliGemma (SigLIP ViT-So400m/14 + Gemma 2B) | model card | — |
| action expert | Gemma 300M | config | — |
| JointModel layers | 18 | `_default_pizero_config` | shape check |
| JointModel heads | 8 | config | — |
| GQA num_key_value_heads | 1 | config | — |
| horizon_steps | 4 | `_default_pizero_config` | — |
| action_dim | 7 | `_default_pizero_config` | — |
| denoise steps | 10 | paper | `denoise_steps: 10` in config |
| benchmark | LIBERO-4 suite | openpi examples | `openpi/examples/libero/main.py` |

---

## Fast-WAM (Yuan 2025)

Skip-imagination WAM: Wan2.2-TI2V-5B frozen video expert + 350M ActionDiT.

| param | required value | source | verify command |
|-------|---------------|--------|----------------|
| ckpt | `libero_uncond_2cam224.pt` | fastwam_release | `ls -lh /data1/ybyang/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224.pt` |
| ckpt on disk | `/data1/ybyang/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224.pt` | exp04a verified | `md5sum` |
| backbone | Wan2.2-TI2V-5B frozen + 350M ActionDiT | model card | — |
| action_horizon | 10 | dataset_stats.json | `cat dataset_stats.json \| jq .action_horizon` |
| action_dim | 14 (7-DoF x 2 arms) | dataset_stats.json | `cat dataset_stats.json \| jq .action_dim` |
| denoise steps | **5** (paper default) | paper Table 3 | `num_inference_steps: 5` |
| denoise steps (comparison) | 10 (for cross-model comparison with Pi-Zero) | our decision | — |
| benchmark | LIBERO-4 suite | paper S5 | `experiments/libero/eval_libero_single.py` |

---

## LingBot-VA (Robbyant 2025)

Full WAM: Wan2.2-TI2V-5B backbone + shared action head, video imagination + action generation.

| param | required value | source | verify command |
|-------|---------------|--------|----------------|
| ckpt | **TBD — location unknown as of 2026-04-28** | — | Task 8 will locate |
| backbone | Wan2.2-TI2V-5B + shared action head | lingbot-va/README | — |
| video denoise steps | 20 | lingbot-va config | config dump |
| action denoise steps | 50 | lingbot-va config | config dump |
| benchmark | LIBERO-4 suite | lingbot-va/evaluation/libero/ | — |

**Note:** exp04b used random weights with the correct architecture. Latency numbers are retained with a `random_weights: true` caveat. Task success eval (exp09d) is blocked until checkpoint is located.

---

## Qwen2.5-VL (Alibaba 2025)

Pure VLM baseline. No VLA action head.

| param | required value | source | verify command |
|-------|---------------|--------|----------------|
| ckpt (7B) | `Qwen/Qwen2.5-VL-7B-Instruct` | HuggingFace | `ls /data1/ybyang/huggingface/Qwen/Qwen2.5-VL-7B-Instruct/` |
| ckpt (3B) | `Qwen/Qwen2.5-VL-3B-Instruct` | HuggingFace | `ls /data1/ybyang/huggingface/Qwen/Qwen2.5-VL-3B-Instruct/` |
| benchmark | N/A — VLM only, profiling for E/P/D phase breakdown reference | — | — |

---

## Canonical Profiling Protocol

All latency measurements share these settings:

| param | value | rationale |
|-------|-------|-----------|
| warmup | 15 iterations | exp07a bimodal contamination: runs 1-12 slow 1.25x from GPU power ramp |
| measurement | 20 iterations | sufficient for stable median + IQR |
| timing | CUDA events (`torch.cuda.Event`) | wall-clock unreliable under multi-stream |
| statistic | median (report mean + std as secondary) | robust to outliers |
| GPU power mode | `nvidia-smi -pm 1` before session | prevents power ramp artifacts |
| dtype | bf16 + sdpa | matches all model defaults |
| hardware | RTX 5880 Ada 48GB (xdlab23) | single-GPU, no tensor parallel |

---

## Known Deviations in Prior Experiments

Three experiments used non-official configurations. This section documents each deviation and its impact on published numbers.

### 1. NitroGen (exp06a): Shrunk architecture

| param | exp06a value | official value | impact |
|-------|-------------|---------------|--------|
| vision_hidden_size | 768 | **1024** | Underestimates encode + context latency |
| action_dim | 20 | **25** | Underestimates action head latency |
| weight_mode | shrunk (174M variant) | full (500M ng.pt) | Per-step 7.2ms is for 174M, not 500M |

**Consequence:** exp06a per-step latency (7.2ms) is for a 174M model, not the real 500M. The 500M rerun (exp09) is expected to show ~20-25ms/step (2.8-3.5x increase from 2.8x more parameters in compute-bound regime).

### 2. Fast-WAM (exp04a): Double denoise steps

| param | exp04a value | official value | impact |
|-------|-------------|---------------|--------|
| num_inference_steps | 10 | **5** | Action phase doubled (362ms vs expected ~180ms) |

**Consequence:** exp04a total 407ms (2.5Hz) overstates latency by ~2x in action phase. Paper reports 190ms on A100; 5-step rerun on RTX 5880 expected ~200-230ms.

### 3. Pi-Zero (exp07a): Random weights

| param | exp07a value | official value | impact |
|-------|-------------|---------------|--------|
| model weights | random initialization | pi0-base pretrained | Timing impact expected <5% (same FLOPs) |

**Consequence:** Timing likely accurate (random vs real weights share identical compute graph). exp09 rerun with real weights will empirically validate this assumption. If delta >10%, kernel dispatch paths may differ.

### 4. ACT (exp02a): Random weights

| param | exp02a value | official value | impact |
|-------|-------------|---------------|--------|
| model weights | random initialization | tonyzhaozh/act ALOHA pretrained | Timing impact negligible for ~3ms total |

**Consequence:** At 3ms total latency, weight values have no measurable timing impact. ALOHA benchmark eval requires real weights (deferred to exp10).

### 5. LingBot-VA (exp04b): Random weights

| param | exp04b value | official value | impact |
|-------|-------------|---------------|--------|
| model weights | random initialization | TBD (checkpoint not located) | Timing impact uncertain for 2.5s total |

**Consequence:** At 2518ms total, VAE encode shows ~20% CV even with warmup=15. Weight values unlikely to change timing significantly but this is unverified. Task success eval impossible without real checkpoint.

---

## Version History

| version | date | changes |
|---------|------|---------|
| v0.9.0 | 2026-04-28 | Initial contract. 7 models pinned. 5 known deviations documented. |

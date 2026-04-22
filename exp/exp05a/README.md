# exp05a — LingBot-VLA-4B Attention Analysis

## Motivation

exp01b 分析了 vanilla Qwen2.5-VL-7B 的 attention pattern，发现:
- Pos 2 (first visual patch) 是 universal attention sink (12K-18K received, 12-28x vs #2)
- Text→Visual Gini >0.91 (extreme sparsity → token pruning 可行)
- Layer 21 entropy 最低 (3.44 vs 4.0-4.2)

**核心问题: VLA fine-tuning 后这些 pattern 是否改变？**

LingBot-VLA-4B 在 Qwen2.5-VL-3B 上做了 flow VLA fine-tuning (Pi0 action head)。
如果 attention pattern 显著变化 → VLA fine-tuning 改变了模型处理视觉信息的方式。
如果 pattern 保持 → attention structure 是 architecture property, token pruning 方法可从 VLM 直接迁移到 VLA。

## Core Questions

1. VLA fine-tuning 是否改变 attention pattern (vs exp01b vanilla VLM)?
2. Universal attention sink (Pos 2) 是否在 VLA 中持续?
3. Text→Visual sparsity 是否保持 (Gini >0.91)?
4. 3B backbone 36 层的 layer-wise entropy profile vs 7B 28 层?
5. (Phase 2) Action-related tokens 如何 attend visual tokens?

## Method

复用 exp01b 的三个 analysis task:
- `visual_text_attention` — Gini sparsity + top-k concentration
- `sink_detection` — attention sink token detection
- `per_layer_stats` — per-layer entropy + mean attention

Config: `configs/lingbot_vla_4b/attention.yaml`
- 6 layers sampled: 0, 7, 14, 21, 28, 35 (36 total)
- Store Q/K projections
- `visual_token_count: 64` (256 patches / 4 spatial_merge)

## Baseline Comparison

| Metric | exp01b (Qwen2.5-VL-7B) | exp05a Prediction |
|--------|------------------------|-------------------|
| Sink position | Pos 2 (12K-18K received) | Pos 2, 可能幅度降低 |
| Text→Visual Gini | >0.91 | 0.85-0.92 (fine-tuning 松散化) |
| Lowest entropy layer | L21/28 = 75% depth | L21-22/36 ≈ 60% depth |
| Overall pattern | — | 定性相似，定量有差异 |

## Hardware

- GPU: RTX 5880 Ada 48GB (xdlab23)
- Model: LingBot-VLA-4B (Qwen2.5-VL-3B backbone + Pi0 flow head)

## Commands

```bash
bash scripts/sync_to_remote.sh
bash scripts/launch_exp.sh 0 lingbot_vla_4b/attention
bash scripts/download-results.sh
```

## Status

**done** — results analyzed 2026-04-21

## Results Summary

VLA fine-tuning 彻底重塑了 attention 结构 (所有预测均失败):

| Metric | exp01b (Vanilla VLM) | exp05a (VLA) | Change |
|--------|---------------------|-------------|--------|
| Sink position | Pos 2 (12-28x ratio) | Pos 64 boundary (3.7x L0, disappears L14+) | Migrated |
| Text->Visual Gini | >0.91 | 0.07-0.45 | 10x collapse |
| Entropy profile | V-shape (L21=3.44) | Flat (4.79-4.90) | Flattened |
| Max attention | High concentration | 0.012-0.019 (except L0) | Dispersed |

**Key conclusion:** Attention structure is training-objective property, not architecture property. VLM token pruning methods (FastV, FlashVLM) likely cannot transfer to VLA.

**Caveat:** Single-sample, confounded by model size (7B vs 3B). Need vanilla Qwen2.5-VL-3B baseline.

# exp05b — Qwen2.5-VL-3B-Instruct Attention Analysis (Vanilla VLM Baseline)

## Motivation

exp05a 发现 LingBot-VLA-4B 的 attention pattern 与 vanilla VLM (exp01b) 截然不同:
- Gini 从 >0.91 崩塌到 0.07-0.45
- Sink 从 Pos 2 迁移到 Pos 64 (boundary)
- Entropy 从 V-shape 变为 flat

**但 exp05a vs exp01b 混淆了两个变量:**
1. Model size: 3B vs 7B
2. Training: VLA fine-tuning vs vanilla VLM

exp05b 用 **vanilla Qwen2.5-VL-3B** (未经 VLA fine-tuning) 跑相同 attention analysis，隔离 model size 效应。

## Design

| 变量 | exp01b | exp05a | exp05b |
|------|--------|--------|--------|
| Backbone | Qwen2.5-VL-7B | Qwen2.5-VL-3B | Qwen2.5-VL-3B |
| Training | Vanilla VLM | VLA fine-tuned | Vanilla VLM |
| Layers | 28 (sampled 5) | 36 (sampled 6) | 36 (sampled 6) |

**消歧逻辑:**
- 如果 exp05b ≈ exp01b (高 Gini, Pos 2 sink) → Gini 崩塌归因于 VLA fine-tuning
- 如果 exp05b ≈ exp05a (低 Gini, no sink) → Gini 崩塌归因于 model size (3B)
- 如果 exp05b 介于两者之间 → 两个因素都有贡献

## Config

`configs/qwen_vl_3b/attention.yaml`
- Same 6 layers: 0, 7, 14, 21, 28, 35
- Same image + prompt as exp05a: "pick up the red cup"
- Uses QwenVLController (vanilla VLM, no VLA action head)

## Commands

```bash
# Uses vit-probe conda env (no lingbotvla dependency needed)
bash scripts/launch_exp.sh 0 qwen_vl_3b/attention
```

## Status

**done** — results analyzed 2026-04-22

## Results

| Metric | exp01b (7B VLM) | exp05b (3B VLM) | exp05a (3B VLA) |
|--------|----------------|----------------|----------------|
| Text→Visual Gini | >0.91 | **0.80-0.98** | 0.07-0.45 |
| Sink position | Pos 2 (12K-18K) | Pos 9 visual (2054) | Pos 64 boundary (283) |
| Entropy profile | V-shape (L21=3.44) | V-shape (L7=2.69) | Flat (4.79-4.90) |
| Max attention | High | 0.15-0.87 | 0.012-0.019 |
| seq_len | ~1200+ | 1273 (424 visual) | 136 (64 visual) |

**消歧结论:** exp05b (vanilla 3B) 和 exp01b (vanilla 7B) 定性一致:
- Gini 都极高 (>0.80)
- 都有 visual attention sink
- 都有 entropy V-shape profile

→ **Gini 崩塌归因于 VLA fine-tuning，不是 model size。**

注意: seq_len 差异显著 (1273 vs 136)。3B vanilla VLM 产生 424 visual tokens (无 spatial_merge)，而 VLA 的 LingBot 仅 64 (spatial_merge=4)。这说明 VLA fine-tuning 不仅改变了 attention pattern，也改变了 visual tokenization pipeline。

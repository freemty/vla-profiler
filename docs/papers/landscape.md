# Domain Literature Landscape

> Research domain: ML Systems / VLM-VLA Inference Efficiency

## Key Papers

(详见 survey/landscape.md 和 survey/papers/recent-papers.md — 已包含 80+ 篇论文)

## VLA Frameworks & Infrastructure

| Paper | Year | Type | Key Contribution | Deep Dive |
|-------|------|------|-----------------|-----------|
| [StarVLA](https://arxiv.org/abs/2604.05014) | 2026 | Framework | Modular platform unifying 4 VLA architectures (AR/MLP/Flow/Dual-system) + 5 VLM backbones; WM4A integration | [deep-dive](starvla-framework-deep-dive.md) |
| [LeRobot](https://github.com/huggingface/lerobot) | 2024 | Framework | HuggingFace's robot learning library; VA-focused (ACT, Diffusion Policy) | -- |

## Research Gaps

1. VLM serving 无 dominant framework (类似 vLLM 之于 LLM)
2. VLA inference 系统层面几乎空白
3. Algorithm-System co-design 未整合
4. StarVLA 覆盖了 VLA 训练/评估，但 inference profiling 和 serving 完全空白 -- 与我们的 vlla 互补

# exp03a: LingBot-VLA-4B Inference Profiling

## Motivation

LingBot-VLA 是蚂蚁灵波 (RobbyAnt) 基于 Qwen2.5-VL-3B 的 4B VLA foundation model。
测量其 E/P/D/Action 延迟 breakdown，量化 VLA fine-tuning 相对于 pure VLM 的 overhead。

**对比基线:**
- exp01a: Qwen2.5-VL-7B (pure VLM) — E=253ms/P=156ms/D=18.6ms/tok (single_img)
- exp02a: ACT (LeRobot, ResNet18) — Total ~3ms

**关键问题:**
1. VLA action head 引入多少额外延迟？
2. 3B backbone vs 7B backbone 的 E/P/D scaling？
3. Depth injection 的延迟成本？

## Model

| 变体 | HuggingFace | 描述 |
|------|-------------|------|
| lingbot-vla-4b | robbyant/lingbot-vla-4b | 标准版 (无 depth) |
| lingbot-vla-4b-depth | robbyant/lingbot-vla-4b-depth | Depth 蒸馏版 |

- Backbone: Qwen2.5-VL-3B-Instruct
- Action dim: 最大 75
- Framework: LeRobot 生态
- License: Apache-2.0

## Method

使用 PhaseTimer 测量四阶段延迟:
1. **Encode (E):** Vision encoder (ViT) 处理图像
2. **Prefill (P):** LLM prefill (visual + text tokens)
3. **Decode (D):** Autoregressive action token generation
4. **Action (A):** Action de-tokenization / post-processing

输入条件:
- 单图 (224x224, 标准 manipulation 视角)
- 多视角 (2-3 cameras)
- 带 depth (depth 蒸馏版)

## Hardware

- Server: xdlab23
- GPU: RTX 5880 Ada 48GB
- Conda: lingbot-vla (新建)

## Expected Results

基于 3B vs 7B 的粗略 scaling 预估:
- Encode: ~120-150ms (约 exp01a 的 50-60%)
- Prefill: ~80-100ms (约 exp01a 的 50-65%)
- Decode/Action: ~10-15ms/tok (与 exp01a 相近，memory-bandwidth bound)
- Total single-step: ~250-350ms

## Status

- [ ] 环境搭建 (conda + 模型下载)
- [ ] 验证 inference
- [ ] PhaseTimer 适配
- [ ] Profiling 运行
- [ ] 分析 + 对比报告

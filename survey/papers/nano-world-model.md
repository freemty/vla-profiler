# nano-world-model — 极简 DiT Video World Model 框架

**Repo**: https://github.com/simchowitzlabpublic/nano-world-model
**来源**: Max Simchowitz lab, CMU
**类型**: 开源框架（nanoGPT 风格）

## 概述

nano-world-model 是一个极简、可复现的 video world model 训练框架。核心设计：DiT backbone (NanoWM-B/2, L/2) + diffusion forcing（逐帧 autoregressive denoising，非 full-sequence EDM）。支持 RT-1 robot、CSGO、DINO-WM 三个环境，包含 MPC planning 评估。单 GPU 数小时可训练完成。

## 与 vlla 相关的关键 Ablation 发现

### 1. Action Injection 方式

| Method | Quality | 备注 |
|--------|---------|------|
| Additive (FiLM-style) | Best | 最简单却最有效 |
| Cross-attention | Mid | 额外计算开销 |
| Concatenation | Worst | 信息稀释 |

**对 candidate A 的启示**: Action Expert (exp07a 的 300M Gemma cross-attn) 可以考虑简化为 additive injection，减少 cross-attn 的 2.3x 开销。

### 2. Prediction Target

pred-v > pred-noise，尤其在 few-step 场景下优势明显。

**对 exp09a 的启示**: 如果做 step distillation (将 DiT action head 从 10-step 蒸馏到 1-2 step)，应优先使用 v-prediction 目标。

### 3. Diffusion Forcing vs Full-Sequence Denoising

Diffusion forcing 逐帧 denoise，支持可变长度生成和 streaming inference。对比 Cosmos Policy 的 full-sequence EDM denoise (一次性去噪整个视频序列)，diffusion forcing 更适合 real-time streaming 场景（每帧就绪即可输出，不需等待整个序列去噪完成）。

**系统级启示**: Streaming world model serving 可能需要支持两种模式 —— full-sequence (planning quality) vs frame-by-frame (低延迟 reactive)。

## 与 agentic-datapipe 的关系

nano-world-model 是 agentic-datapipe 项目的 fitness evaluator：进化数据管线产出的 action-conditioned video 数据，用 NanoWM 快速训练并获取 PSNR/FID/FVD 作为 fitness signal，远比训练完整 Cosmos-Predict2-2B 经济。

## 定位

| 维度 | nano-world-model | Cosmos Policy | DreamZero |
|------|-----------------|---------------|-----------|
| 参数量 | 34M-130M | 2B+ | 14B |
| 训练成本 | 单 GPU 数小时 | 多节点数天 | 多节点数天 |
| Denoising | Diffusion forcing (per-frame) | Full-sequence EDM | Full-sequence |
| 用途 | Research playground / fitness proxy | Production policy | Zero-shot policy |

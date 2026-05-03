---
title: "Dreamverse: Vibe Directing in FastVideo + Real-Time 1080p Inference Stack"
authors: Hao AI Lab (Will Lin, Matthew Noto, Junda Su, Yechen Xu, Peiyuan Zhang, Hao Zhang et al.)
venue: Blog post (haoailab.com)
urls:
  - https://haoailab.com/blogs/dreamverse/
  - https://haoailab.com/blogs/fastvideo_realtime_1080p/
date_published: 2026-03-11 / 2026-03-15
date_saved: 2026-05-03
tags: [FastVideo, DiT, video-generation, real-time, Blackwell, NVFP4, inference-system, Hao-Zhang]
---

# Dreamverse / FastVideo Real-Time 1080p Stack

## Why this matters for vlla

Hao Zhang (我们的导师) 的最新工作。从原版 FastVideo (STA/VSA 稀疏 attention + 蒸馏) 演进到 **全栈硬件-系统协同优化**，在单 B200 GPU 上实现 5s 1080p 视频的 4.55s 生成（real-time）。Dreamverse 是其上的应用层——"Vibe Directing" 交互式视频导演。

对 vlla "Fast VLA first" 直接相关：
1. **技术路线验证**：Hao 自己的最新工作就是把 video DiT 加速到 real-time 交互，和 candidate A (Action DiT 加速) 完全同线
2. **技术栈可迁移**：NVFP4 + Blackwell kernel + graph fusion 如果应用到 Cosmos Policy 2B DiT (exp09a 测得 76.8ms/step on RTX 5880)，压缩空间巨大
3. **Dreamverse → VLA 交互**：当 VLA 推理 real-time 化后，"vibe directing for robotics" 是自然延伸

## 架构关系

```
FastVideo 原版 (2025)        → Dreamverse stack (2026-03)
  算法优化 (STA/VSA + 蒸馏)     全栈系统优化 (NVFP4 + kernel + graph + 系统工程)
  通用 GPU (A100/H100)          Blackwell B200/B300 专用
  480p-720p                     1080p (1088×1920)
  多 GPU + SP                   单 GPU (无需 SP)
  无 serving                    Rust middleware + Dynamo 集成
  无应用层                      Dreamverse (vibe directing)
```

## 核心数字

| 指标 | 值 |
|------|-----|
| 端到端延迟 | **4.55s** (5s 1080p clip, single B200) |
| vs next best | **3.9x faster** |
| vs Veo-3 Fast | 55s → 4.55s (12x) |
| 分辨率 | 1088×1920 (1080p), 24 FPS |
| 硬件 | Single NVIDIA B200 GPU |
| Base model | LTX-2.3 (Lightricks, TI2AV) |
| Serving | Half GB200 NVL72 (36 GPU replicas) |

## 新增 Technical Components (vs 原版 FastVideo)

### 1. NVFP4 量化 (Blackwell-native)
原版全程 BF16。新版利用 B200/B300 Tensor Core 原生 NVFP4 (4-bit floating point) 对 DiT linear layers 做低精度推理。硬件级新数据类型，理论吞吐比 BF16 高数倍。原版论文完全没涉及量化。

### 2. SM100/SM103 专用 attention kernel
原版 STA/VSA 是算法层稀疏 attention pattern。新版写了 Blackwell 架构专用的 3D spatiotemporal attention kernel，针对 SM100/SM103 硬件特性。Blog 不再单独提 STA/VSA——可能已融入硬件 kernel 或在 Blackwell 上用不同实现。

### 3. 全栈 graph + kernel fusion
原版只优化 DiT attention 和 denoise loop。新版优化整条 pipeline：
- Prompt encoding (T5)
- Latent preparation (VAE encode + noise schedule)
- Denoising loop (DiT)
- Decoding (VAE decode + frame output)

每个阶段都做了 kernel fusion。从"优化一个模块"到"优化整个系统"。

### 4. 系统级优化 (模型之外)
原版不碰 model 之外。新版发现真实 e2e 有大量非 model 开销：
- IPC overhead 消除
- ffmpeg 针对目标 CPU 编译优化
- Frame/audio I/O pipeline 流式化
- 用 FastVideo 的 system profiling 工具定位瓶颈

### 5. 1080p 分辨率
480p → 1080p 不只是分辨率翻倍：spatial workload ~4x 增长，attention quadratic 增长。所有优化重新针对 1080p 调参。

### 6. 单 GPU 部署 (去掉 SP)
原版多 GPU + sequence parallelism。新版单 B200 不需要 SP，大幅简化部署。

### 7. Serving 层
- Rust-based middleware (比 Python 低延迟)
- NVIDIA Dynamo 官方集成 (FastVideo 作为 diffusion backend)

### 8. Vibe Directing 应用层 (Dreamverse)
- 链式生成：多个 5s clip 串成 30s 场景
- 聊天窗口实时导演："keep subject, change background"
- 多版本分支探索 ("multiverse")
- K2-V2 (IFM) 语言模型驱动自然语言指令

## 没有再提到的原版 component

| 原版 | 新版状态 |
|------|---------|
| STA 稀疏 attention | 可能融入 SM100 kernel，不再单独提名 |
| VSA | 同上 |
| 蒸馏 (step reduction) | 完全没提。LTX-2.3 可能步数已少，或 NVFP4 + fusion 够快 |
| Triton kernel | 可能被 SM100 native kernel 替代 |

## 一句话对比

原版 FastVideo = **算法优化 (稀疏 attention + 蒸馏)**，通用 GPU 上加速 DiT。
Dreamverse stack = **全栈硬件-系统协同优化 (NVFP4 + Blackwell kernel + graph fusion + 系统工程)**，Blackwell 专用硬件上把 1080p 推到 real-time。

## Bridge to vlla experiments

- **exp09a Cosmos Policy**: 76.8ms/step (2B DiT, RTX 5880 Ada BF16)。如果能移植 NVFP4 + graph fusion，理论压到 ~15-30ms/step (4-5x)，1-step distilled total ~280-295ms → **3.4-3.6 Hz** (接近 Pi-Zero 5Hz)
- **但前提是 Blackwell 硬件**：RTX 5880 Ada 不支持 NVFP4 (那是 SM89, 不是 SM100)。在现有硬件上，graph fusion + INT8/FP8 量化是更现实的路径
- **Hao 的 co-design 方法论**：Dreamverse 完美体现了 "profile → identify bottleneck → kernel+algorithm+system co-optimize" 的 5-step 流程。我们的 exp09a step sweep 数据 (fixed cost 265ms 占主导) 暗示**VAE encode 和 system overhead 才是下一个要攻的瓶颈**，和 Dreamverse blog 的发现一致

# DreamDojo + DreamZero -- Dual Paper Deep Dive

**Paper 1:** DreamDojo: A Generalist Robot World Model from Large-Scale Human Videos (arXiv:2602.06949)
**Paper 2:** DreamZero: World Action Models are Zero-shot Policies (arXiv:2602.15922)
**Authors:** NVIDIA GEAR Lab + Berkeley + UT Austin (重叠核心作者: Shenyuan Gao, Kaiyuan Zheng, Seonghyeon Ye, Sihyun Yu 等)
**Year:** 2026
**Read date:** 2026-04-21

---

## 1. Methodology Skeleton

### 1.1 DreamDojo -- 从人类视频学习 World Dynamics

**核心问题:** 如何从海量无标注人类视频中学习通用 robot world model，post-training 后精确模拟 contact-rich 机器人交互?

**架构与训练 Pipeline:**

```
Phase 1: LAM Training (Latent Action Model)
  输入: 无标注视频帧对 (o_t, o_{t+1})
  目标: z_t = LAM_encoder(o_t, o_{t+1}) -- 编码帧间变化为 latent action
  作用: 将 "无标注视频" 转化为 "带隐式 action 的训练数据"

Phase 2: Pretraining (Foundation World Model)
  Backbone: Cosmos Predict2 (NVIDIA 自研视频生成基座)
  数据: 44k 小时 egocentric human video (目前 WM pretraining 最大规模)
  条件: latent action z_t
  目标: 给定当前帧 + latent action, 预测未来视频

Phase 3: Post-training (Target Robot Adaptation)
  数据: 小规模目标 robot 数据 (带真实 action labels)
  方法: 将 latent action space 映射到 robot action space

Phase 4: Distillation (Real-time Acceleration)
  Teacher → Student: 多步 → 少步, 14B → 可能更小
  结果: 10.81 FPS + 上下文一致性改善
```

**模型规模:** 2B 和 14B 两个变体。

**LAM 核心创新:**
- 解决 human video 有丰富物理交互但无 action label 的矛盾
- 无监督编码帧间变化为 continuous latent action
- WM 通过 latent action 学会 "action -> effect" 因果关系
- Post-training 时只需对齐 z_t <-> a_t^{robot}

### 1.2 DreamZero -- WAM 即零样本策略

**核心问题:** 能否通过 joint video-action modeling 构建 WAM, 不依赖重复 demonstration 实现 zero-shot 跨任务、跨环境泛化?

**架构:**

```
Backbone: Wan2.1-I2V-14B-480P (阿里万象 I2V 14B)
Tokenizer: umt5-xxl
Joint output: video frames (latent) + action trajectory

[Current frame + language instruction]
    -> Text encoder (umt5-xxl) + Image encoder
    -> DiT backbone (Wan2.1, 14B params)
    -> Joint denoising (video latent + action tokens)
    -> Future video + Action trajectory
```

**训练 Pipeline:**

```
Step 1: Start from pretrained Wan2.1-I2V-14B (Internet-scale video prior)
Step 2: Joint video-action fine-tuning on DROID (76k demos)
  - DeepSpeed ZeRO Stage 2 + LoRA
  - L = L_video + lambda * L_action (joint denoising loss)
Step 3: DiT Caching + TensorRT for inference optimization
```

**零样本泛化来源:**
1. Wan2.1 的 Internet-scale video prior (物体运动模式)
2. Joint training 使 video 空间的物理 prior 迁移到 action 空间
3. 结果: zero-shot 比 SOTA VLA 提升 2x

**Cross-embodiment transfer:**
- Video-only demonstrations (无 action label): 42%+ improvement (10-20 min data)
- Few-shot embodiment adaptation: 30 min play data 适配新 robot

### 1.3 实时推理实现方式

**DreamDojo: 蒸馏 (model-level)**
- Full model: 14B, 多步 diffusion → ~1-3 FPS
- Distilled model: step reduction + model compression → 10.81 FPS (~92ms/frame)
- 蒸馏同时改善 context consistency

**DreamZero: DiT Caching + TensorRT (system-level)**
- DiT Caching 核心观察: 相邻 denoising steps 间, 深层 activation 变化小 → 可缓存
- 缓存策略: 偶数步全算+缓存, 奇数步浅层重算+深层用缓存 → ~50% 计算减少
- GB200: ~0.6s (~1.7Hz raw, 7Hz with TensorRT) 
- H100: ~3s (~0.3Hz) -- 硬件依赖极强

---

## 2. Assumptions & Limitations

### DreamDojo

| 假设 | 评估 |
|------|------|
| Egocentric video 包含可迁移物理 prior | 标准, Ego4D 等验证 |
| LAM 能学到 action-effect causality | 限制性 -- 可能学到 appearance change 而非 physical causality |
| 蒸馏保留 world model 质量 | 限制性 -- contact-rich 场景质量损失可能大 |

Failure modes: 透明/反光物体、极长 horizon planning、精确力控制

### DreamZero

| 假设 | 评估 |
|------|------|
| Video diffusion backbone 隐式编码物理规律 | 标准偏限制 -- Internet video 含非物理内容 |
| Joint video-action denoising 有效 | 限制性 -- video 和 action 可能 interfere |
| GB200 hardware 可获得 | 高度限制性 -- 绝大多数实验室无 GB200 |
| DROID 代表性充足 | 限制性 -- 仅 tabletop manipulation |

Failure modes: 硬件依赖 (GB200)、DROID 外 embodiment 需额外 adaptation、video hallucination → 非物理 action

---

## 3. Bridge Analysis (与本项目实验的桥接)

### 延迟对比

| Metric | exp04a (Fast-WAM) | exp04b (LingBot-VA) | DreamDojo (蒸馏) | DreamZero (GB200) | DreamZero (H100) |
|--------|-------------------|--------------------|--------------------|-------------------|-------------------|
| Total latency | 407ms | 2091ms | ~92ms | ~143ms | ~3000ms |
| Control freq | 2.5Hz | 0.5Hz | 10.81Hz | 7Hz | ~0.3Hz |
| Hardware | RTX 5880 Ada | RTX 5880 Ada | 未公开 | GB200 | H100 |

### 关键对比发现

1. **exp04b (0.5Hz) ≈ DreamZero on H100 (~0.3Hz)** → full WAM 在非旗舰硬件普遍无法 real-time
2. **DreamZero H100→GB200: 23x 提速** → WAM real-time 高度依赖硬件代际
3. **"step reduction is highest ROI" 被两篇论文印证** (DreamDojo 蒸馏 / DreamZero DiT caching)
4. **Action dominates 68-89% (exp04a/04b)** → DreamDojo 蒸馏直接压缩; DreamZero DiT caching 优化 per-step

### 架构对比的 Systems 含义

**DreamDojo (两阶段: WM → action decode):**
- WM 和 action decoder 可 disaggregate 到不同硬件
- WM output 可 cache/reuse (同场景不同 action queries)
- 对应 ERA disaggregation 概念

**DreamZero (一阶段: joint denoising):**
- 必须在同一 GPU cluster 跑完整 14B DiT
- 没有中间缓存点
- 优化手段: DiT caching + hardware acceleration

### DiT Caching 在 memory-bandwidth bound regime 的疑问

exp04a/04b 发现 WAM action phase 是 memory-bandwidth bound。DiT caching 减少 compute 但增加 memory (缓存中间 activation)。在 RTX 5880 Ada 上可能是净损失 -- 这是一个值得验证的关键问题。

---

## 4. Comparative Analysis (DreamDojo vs DreamZero)

### 论文关系
- 同一实验室 (NVIDIA GEAR Lab), 核心作者高度重叠
- DreamDojo: **foundation model 工程** (数据规模 + LAM + 蒸馏)
- DreamZero: **方法论创新** (joint video-action + zero-shot transfer)

### 技术选型

| 维度 | DreamDojo | DreamZero |
|------|-----------|-----------|
| Video backbone | Cosmos Predict2 (闭源) | Wan2.1-I2V-14B (开源) |
| Action 处理 | LAM latent proxy (两阶段) | Joint denoising (一阶段) |
| Pretraining 数据 | 44k hr human egocentric video | Internet-scale (Wan2.1) |
| Robot 数据 | Post-training 小规模 | DROID 76k demos |
| 实时化 | 蒸馏 (model-level) | DiT caching + TensorRT (system-level) |
| 速度 | 10.81 FPS | 7Hz (GB200 only) |

### 对本研究的价值

**DreamZero 更有价值:**
1. DiT caching 可被 profiling 框架精确测量
2. 与 exp04a/04b 的 DiT 架构更接近 (per-step ~28-32ms baseline 可直接对比)
3. Hardware-performance curve 是关键研究问题
4. 开源可复现 (Wan2.1 + 训练代码 + checkpoint)

**DreamDojo 的补充价值:**
- LAM 概念补充 landscape (从无标注视频学 action 的路线)
- 蒸馏 pipeline 验证 step distillation 在 WAM 上的可行性

---

## Open Questions

1. DreamZero 在 RTX 5880 Ada 上的实际延迟? (潜在 exp05)
2. DiT caching 在 memory-bandwidth bound regime 下的真实收益?
3. DreamDojo 蒸馏质量在 contact-rich 场景下如何?
4. Joint denoising vs separate denoising 的质量-效率 Pareto 前沿?
5. DreamZero zero-shot claim 的 robustness (评估场景代表性)?

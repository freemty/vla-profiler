# Meeting Cheat Sheet — 面谈前 15 分钟速览

> 目标：能用自己的话说清这些概念，不是背诵。

---

## FastVideo — 你必须能说的 3 件事

### 1. 它解决了什么？

Video DiT 生成 5 秒 720p 视频要 945 秒。Profiling 发现 **3D full attention 占 85% (800s)**。FastVideo 就是干掉这 85%。

### 2. 怎么干的？(STA + VSA + 蒸馏，三板斧)

**STA (Sliding Tile Attention)** — 最核心的
- 观察：预训练 DiT 的 attention score 天然有**3D 局部性**（时空邻域内 attention mass 集中）
- 做法：用 tile（一组 token 的矩形块）代替逐 token 的 sliding window，让 GPU block 边界和 tile 边界对齐
- 结果：attention 加速 2.8-17x，端到端 945s→268s (fine-tune 后质量损失 0.09%)
- **口头版**："STA 发现 video DiT 的 attention 天然是局部的，用 tile 粒度的 sliding window 替代 full attention，硬件效率和 FlashAttention 一样。"

**VSA (Video Sparse Attention)** — STA 的进化版
- 不用固定窗口，而是动态识别每个 query tile 的高权重 key tiles (coarse-to-fine)
- 端到端可训练 (可微分 Triton kernel)
- Wan2.1 端到端 31s→18s

**蒸馏** — 减 denoising steps
- DMD2: teacher 多步 → student 少步匹配分布
- Self-Forcing: autoregressive 蒸馏，每帧条件化于已生成帧
- 组合加速：STA/VSA + step 压缩 = **>50x** denoising 加速

### 3. 能迁移到 VLA 吗？(meeting 核心论点)

| VLA 场景 | STA 适用性 | 为什么 |
|----------|-----------|--------|
| NitroGen 174M DiT | 低 | 序列短，attention 不是主要 overhead |
| Pi-Zero 300M (cross-attn) | 中 | cross-attn to VLM KV 有 35% overhead，可能有局部性 |
| Fast-WAM 350M (30L MoT) | **高** | 30 层 per-layer cross-attn，若有局部性可省 2-3x |
| LingBot-VA 5B (full WAM) | **高** | 和 Wan2.1 结构最像，STA 直接适用 |

**关键前提**：需要先验证 action DiT 的 attention score 是否也有局部性 → 这就是第一个实验。

**蒸馏更直接**：NitroGen k=16→k=1 已从 126ms 降到 18ms (56Hz)。step 蒸馏对 VLA 是最低 hanging fruit。

---

## DistServe — 你必须能说的 2 件事

### 1. PD disaggregation 是什么？

Prefill (处理 prompt, compute-bound) 和 Decode (生成 token, memory-BW-bound) **结构性不同**，放同一 GPU 互相干扰。DistServe 把它们分到不同 GPU，各自用最优并行配置。

**口头版**："Prefill 是 compute-bound，Decode 是 memory-bandwidth-bound，两者资源需求结构性不同。放一起 GPU 两边都用不满，分开各自最优，goodput 提升 7.4x。"

### 2. 为什么 meeting 里说 "serving later"？

- DistServe 解决的是 **多用户并发** 下的资源争用 — 前提是单请求已经够快
- VLA 现在单请求还差 2-10x (Pi-Zero 5Hz，需要 10-50Hz)
- 类比：FastVideo 先把视频生成从 **分钟压到秒**，serving 需求才跟着来
- 而且 vLLM-Omni / SGLang Diffusion 已经在做 multimodal serving → 我们不该重复

**如果 Hao 问 "为什么不直接做 serving"**：
> "VLA 还没有一个统一的推理加速框架。5 个模型 5 套 env 5 个入口。先统一 profiling、把 Action DiT 加速做到位，serving 需求自然会浮现 — 就像 FastVideo 先把单次推理做快，serving 才有意义。"

---

## 你的数据速查 (口头引用)

```
ACT           3ms   300Hz   ← VLA 下界，窄任务
LingBot-VLA  74ms    13Hz   ← 轻 flow head，瓶颈在 backbone
NitroGen     18ms    56Hz   ← k=1 step 蒸馏后，已实时
Pi-Zero     200ms     5Hz   ← Action 82%，cross-attn +35%
Fast-WAM    407ms   2.5Hz   ← Action 89%，MoT cross-attn
LingBot-VA 2518ms   0.4Hz   ← Full WAM，video imagination 太贵
```

**DiT scaling**: 174M=7.2ms/step → 300M=16.5ms → 350M=32ms (2x params, 4.4x latency)

**Attention 发现**: VLM Gini >0.91 (pruning 可行) → VLA fine-tune 后 Gini 崩塌到 0.07 (不可迁移)

---

## exp08 口头版 (备用，他问才说)

"我还做了 EPDA 四阶段共存的干扰测量。发现 Decode 和 Prefill 极度脆弱 (2.4-2.9x 膨胀)，Encode 和 Action 鲁棒 (<1.3x)。部署建议是 {E,A} 同卡，{P,D} 必须分开 — 和 DistServe 的 PD disaggregation 一脉相承，只是扩展到四阶段。这个 park 住了，等 serving 成为真需求再展开。"

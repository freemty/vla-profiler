# UVA vs Cosmos Policy — 对比精读: 从 Decoupled 到 Monolithic 的研究轨迹演进

## Papers

**Paper 1: Unified Video Action Model (UVA)**
- Authors: Shuang Li, **Yihuai Gao**, Dorsa Sadigh, **Shuran Song** — Stanford
- Date: 2025-02 (arXiv:2503.00200)
- Tags: video-action joint model, decoupled diffusion, masked training, MAR backbone

**Paper 2: Cosmos Policy**
- Authors: Moo Jin Kim, **Yihuai Gao**, Tsung-Yi Lin, Yen-Chen Lin, Yunhao Ge, Grace Lam, Percy Liang, **Shuran Song**, Ming-Yu Liu, Chelsea Finn, Jinwei Gu — NVIDIA + Stanford
- Date: 2026-01 (arXiv:2601.16163, ICLR 2026)
- Tags: video model adaptation, latent frame injection, monolithic DiT, test-time planning

**Shared authors: Yihuai Gao (执行核心) + Shuran Song (senior PI)** — 同一组人的两代作品。

---

## 1. Author-Level Connection

UVA → Cosmos Policy 间隔 ~11 个月。Cosmos Policy 新增 NVIDIA 阵营 (Ming-Yu Liu, Jinwei Gu, Tsung-Yi Lin) + Stanford Chelsea Finn + Percy Liang。从 Stanford 内部项目升级为 Stanford-NVIDIA 联合旗舰。

Yihuai Gao 大概率在 UVA 期间就开始与 NVIDIA Cosmos 团队接触。核心 thesis ("video + action 联合建模") 被带到 Cosmos 框架，但 **实现路径发生根本性转变**。

---

## 2. Technical Lineage

### 被保留的核心 thesis
- Video 和 action 应在同一 latent 空间建模
- Video generation 能力对 policy 有益
- 推理时可 skip video decoding
- 统一框架支持多种功能 (policy/dynamics/planning)

### 被演进的关键设计

| 维度 | UVA (2025-02) | Cosmos Policy (2026-01) | 演进方向 |
|------|-------------|----------------------|---------|
| 骨干选择 | MAR-B (0.5B), 从零预训练 | Cosmos-Predict2-2B, 复用工业级预训练 | "站在巨人肩上" |
| 训练策略 | 两阶段 (video → joint) | 单阶段 post-training | 简化流程 |
| 推理模式 | Policy-only (skip video head) | Direct + Planning 两种模式 | 增加 test-time planning |
| Value estimation | 无 | 生成 value function + future images | 增加 model-based planning |

### 被抛弃的核心设计 (最关键转变)

**UVA 的 Decoupled Decode 被彻底抛弃。**
- UVA: joint latent Z → 两个独立轻量 diffusion head
- Cosmos: 单一 DiT, action 作为额外 latent frame 参与同一 denoising trajectory

---

## 3. The Key Architectural Fork: Decoupled vs Monolithic

### UVA Decoupled 架构
```
Obs → VAE Encoder → MAR Transformer (0.5B) → joint latent Z
                                                /         \
                                        Action Head    Video Head
                                     (lightweight diff) (lightweight diff)
```
- Skip video head → 只跑 action head (15ms@16 steps)
- 总计 ~55-95ms on RTX 3080

### Cosmos Policy Monolithic 架构
```
Obs → VAE Encoder → [z_obs, z_T_video, z_T_action]
                              ↓
                    Cosmos-Predict2-2B (DiT)
                    50-step flow matching denoise
                              ↓
                    [z_video_clean, z_action_clean]
                         /              \
                   VAE Decode      Linear Projection
```
- 即使 direct mode 仍需完整 50-step 2B DiT forward
- 没有轻量 action head 可以 shortcut

### 为什么同一组人从 Decoupled → Monolithic?

1. **Scale 改变游戏规则**: 0.5B 需要 decouple 给不同 modality 专用通道; 2B capacity 充裕, 单体即可
2. **预训练力量**: Cosmos video 能力内置于每层 attention, 加 head 反而破坏预训练知识
3. **系统同质性**: Monolithic DiT = FastVideo STA/VSA/step distillation 无摩擦迁移

### Trade-off 总结

| 维度 | UVA (Decoupled) | Cosmos Policy (Monolithic) |
|------|-----------------|--------------------------|
| Action-only 推理延迟 | ~55-95ms | ~400-4000ms (full 2B DiT) |
| 架构复杂度 | 高 (backbone + 2 heads) | 低 (单一 DiT) |
| 预训练复用 | 困难 (自训 backbone) | 天然 (直接 finetune) |
| 加速术迁移 | 需分别优化 | FastVideo 全套直接可用 |
| Quality ceiling | LIBERO-Long 90.0% | LIBERO avg 98.5% SOTA |
| Step distillation 潜力 | 16 步, 压缩空间小 | 50 步 → 4-5 步, 巨大压缩空间 |

---

## 4. Inference Profiling 视角对比

### UVA E/C/A 估算 (RTX 3080)

| Phase | 延迟 | 占比 |
|-------|------|------|
| E (VAE encode) | ~40ms | 42-73% |
| C (MAR forward) | ~30-40ms | 31-42% |
| A (action head) | ~15ms (16-step) / ~93ms (100-step) | 16-53% |
| Total | ~85-173ms | — |

特征: **均衡型** (类似 exp03a LingBot-VLA)

### Cosmos Policy E/D/X 估算

| Phase | 延迟估算 | 占比 |
|-------|---------|------|
| E (VAE encode) | ~20-40ms | 5-10% |
| D (DiT denoise) | ~400ms (@5-step) | 88-92% |
| X (action extract) | <1ms | <1% |
| Total | ~420-440ms | — |

特征: **Denoise 绝对 dominate** (类似 exp04a Fast-WAM)

### 跨实验 Latency Spectrum 定位

| 模型 | 总延迟 | 频率 | 主导阶段 | 架构类型 |
|------|--------|------|---------|---------|
| ACT (exp02a) | 3ms | 333Hz | Encode 80% | 轻量 VA |
| LingBot-VLA (exp03a) | 74.5ms | 13Hz | E~C~A 均衡 | Flow VLA |
| **UVA (论文)** | **85-173ms** | **6-12Hz** | **E~C~A 均衡** | **Decoupled video-action** |
| Pi-Zero (exp07a) | 200.5ms | 5Hz | Action 82% | Dual-stream flow VLA |
| Fast-WAM (exp04a) | 407ms | 2.5Hz | Action 89% | WAM (skip-imagination) |
| **Cosmos Policy (估算)** | **~440ms** | **~2.3Hz** | **Denoise 90%+** | **Monolithic video-policy** |
| LingBot-VA (exp04b) | 2518ms | 0.4Hz | Action 69% | Full WAM |

---

## 5. Scale: 预训练复用 vs 从零训练

两篇论文合在一起验证递进 thesis:
1. **UVA**: video + action 联合建模 > 纯 action (0.5B 规模下)
2. **Cosmos Policy**: 工业级 video 预训练 backbone + 最小改动 finetune 可以登顶 (2B 规模)

Step distillation 潜力估算:
- Cosmos 50 步 → 5 步: ~440ms (~2.3Hz)
- Cosmos 50 步 → 5 步 + STA 2x: ~220ms (~4.5Hz)
- 接近 Pi-Zero exp07a 水平, 但 quality 更高

---

## 6. Bridge to Our Research

### 共同揭示的 insights
1. **Video-action 共享表征是确定性趋势** — 两种架构都验证了这个 thesis
2. **Decoupled vs Monolithic 是 scale 的函数** — <1B: decoupled 更高效; 2B+: monolithic 更简洁
3. **Step distillation 是 monolithic 的杀手级优化** — 50 → 5 步 = 10x, 恰好是 FastVideo 最擅长的领域

### 哪种架构更适合加速?
- **短期 (exp09a)**: Cosmos Policy (monolithic) — 优化方向清晰, FastVideo 直接迁移
- **中长期 (极限 latency)**: UVA-style decoupled — 轻量 action head 可达 ~15ms, 但需自训 backbone

### 建议
exp09a 先做 Cosmos Policy, 验证 step distillation 效果。如果 5-step 后仍无法突破 5Hz, 再考虑 UVA-style decoupled 作为 plan B ("大 backbone + 轻量 decoupled action head")。

### Open Questions
1. UVA decoupled head + Cosmos backbone = 同时获得高 quality 和低 latency?
2. Cosmos 的 50-step 有 quality-step sharp transition 吗? (exp09a step sweep)
3. Action latent frame 在 Cosmos attention 中是什么角色? (exp05a 式 attention 分析)
4. 如果 Yihuai Gao 做第三代: 4B+ monolithic, edge (<1B) 回到 decoupled?

---

## One-liner

UVA → Cosmos Policy 是同一组人 (Gao, Song) 从 "decoupled video-action diffusion" 到 "monolithic latent frame injection" 的演进: scale 改变了架构选择, 但核心 thesis (video+action 共享表征) 不变。对 profiling 的意义: UVA 是均衡型 (适合 pipeline 优化), Cosmos 是 denoise-dominated (适合 step distillation) — 两条不同的加速路径。

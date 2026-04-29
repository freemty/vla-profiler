# Experiment Summary — TLDR

Cross-experiment flight recorder. Per-exp **目的 / 方法 / 结果 / 下一步** 四行卡片，底部保留单行汇总表。
每个 exp 的完整 README、配置、原始数据见 `exp/<id>/`。

---

## exp01a — Qwen2.5-VL-7B E/P/D baseline · **done**

- **目的**：为所有下游 VLA 对比建立 pure-VLM latency 基线。
- **方法**：Hydra + PhaseTimer，三种输入 (text_only / single_image / multi_image)，RTX 5880 Ada, bf16/sdpa, warmup=15 iter=20。
- **结果**：text P=20/D=18 ms；single_img E=253/P=156/D=18.6 ms；multi_img (3) E=541/P=332/D=21 ms。**Encode 与图片数线性**，Decode per-token 稳定。
- **下一步**：作为 exp03a/05a/07a/08 的锚点持续引用。

## exp01b — Qwen2.5-VL-7B attention analysis · **done** (无独立目录，写在 exp01a 配置下)

- **目的**：看 vanilla VLM 的 attention pattern 是否支持 visual token pruning。
- **方法**：Gini / attention sink 检测 / per-layer entropy，5 层采样 (0,7,14,21,27)。
- **结果**：**Pos 2 = universal attention sink**（12K–18K received，比 Pos#2 高 12–28x）。Text→Visual Gini >0.91（极稀疏 → pruning 可行）。Layer 21 entropy 最低 (3.44)。
- **下一步**：exp05a/b 测 VLA fine-tune / model size 是否改变该结论。

## exp02a — ACT (LeRobot) E/A · **done**

- **目的**：测经典 lightweight VLA 下限，锁定 "最快 VLA" anchor。
- **方法**：ResNet18 + transformer action head，3 种分辨率，CUDA-timed。
- **结果**：Total ~3 ms (~850x faster than VLM)；Encode 2.5–2.8 ms (80%)，Action 0.4–0.8 ms。
- **下一步**：作为 flow-VLA / full-WAM 对比下限。

## exp03a — LingBot-VLA-4B (Qwen2.5-VL-3B + flow) · **done**

- **目的**：量化 flow VLA fine-tuning 相对 pure VLM 的 overhead。
- **方法**：E/C/A breakdown (encode, context, 10-step flow denoise)，single_img + multi_view。
- **结果**：single_img E=35.7/C=38.3/A=0.48 → total 74.5 ms (~13 Hz)。**3B backbone 比 7B 快 7x**，Context ≈ Encode，flow action 0.48 ms 与 ACT 同量级。Multi-view 仅 +1.5% (patchify 前聚合)。
- **下一步**：作为 flow-VLA Pareto 前沿的"中等权重"数据点，与 Pi-Zero (exp07a) 做对比。

## exp04a — Fast-WAM (skip-imagination) · **done**

- **目的**：测"跳过 test-time video imagination"的 WAM 延迟分布。
- **方法**：E/C/A per-step sweep (k=10/20)，ActionDiT 350M。
- **结果**：@10step E=7.6/C=36.7/A=362 → total 407 ms (2.5 Hz)。**Action 占 89–94%**（30 层 MoT cross-attn）。Per-step ~32 ms。论文自报 190ms 疑似 A100 + k=5。
- **下一步**：与 exp04b (full WAM) 做 skip-imagination 代价量化。

## exp04b — LingBot-VA (full WAM, video imagination) · **done**

- **目的**：测"有 video imagination" 的 full WAM 延迟，补全 WAM 光谱。
- **方法**：E/V/A (video 20 步 + action 50 步)，warmup=15 iter=20 median。
- **结果**：E=84.7/V=697/A=1708 → total 2518 ms (0.40 Hz)。**Full WAM 比 skip-imagination 慢 ~6x**，action 占 69%。Encode CV ~20% 即便充分 warmup → VAE 本质 variance。Legacy warmup=3 数据系统性低估 ~18%。
- **下一步**：DreamZero (NVIDIA) 替代 profiling 候选。

## exp05a — LingBot-VLA-4B attention analysis · **done**

- **目的**：VLA fine-tuning 后 attention pattern 是否仍支持 token pruning？
- **方法**：复用 exp01b 三个 analysis task，6 层采样 (0,7,14,21,28,35)。
- **结果**：**VLA fine-tuning 彻底重塑 attention**。Sink 从 Pos 2 迁移到 Pos 64 (boundary)。Text→Visual Gini 从 >0.91 崩塌到 0.07–0.45 (接近均匀)。Entropy V-shape 变成 flat。**VLM token pruning 方法不可直接迁移到 VLA**。
- **下一步**：exp05b 消歧 model-size 混淆。

## exp05b — Qwen2.5-VL-3B vanilla attention (消歧) · **done**

- **目的**：exp05a 的 Gini 崩塌归因于 VLA fine-tune 还是 3B backbone？
- **方法**：跑 vanilla 3B（未经 VLA 微调）相同 attention task。
- **结果**：**消歧成功**：3B vanilla Gini 0.80–0.98（与 7B 一致），Pos 9 sink，entropy V-shape。**Gini 崩塌 = VLA fine-tuning，非 model size**。
- **下一步**：attention structure 是 training-objective property 这一结论可写成 mechanism note。

## exp06a — NitroGen 500M DiT k-sweep · **done**

- **目的**：补 DiT per-step latency vs size scaling curve 的中间点 (174M)。
- **方法**：E/C/A + k ∈ {1,2,4,8,16} sweep。
- **结果**：@k16 E=8.7/C=2.0/A=115.4 → total 126 ms (7.9 Hz)。**Per-step DiT 7.2 ms，完美线性**。k=1 total 17.9 ms (55.9 Hz)。vs Fast-WAM (350M, 32 ms/step)：**2x params 却 4.4x latency → memory-BW-bound 区间转折**。
- **下一步**：与 Pi-Zero 300M Action Expert 比对，看 cross-attn 是否多付 2x。

## exp07a — Pi-Zero (dual-stream flow VLA) · **done**

- **目的**：补 flow VLA Pareto 前沿的 PaliGemma + 300M Expert 数据点。
- **方法**：E/C/A profiling，stable-window (runs 13–20)，warmup=15。
- **结果**：E=9.32/C=26.40/A=164.76 → total 200.5 ms (~5 Hz)。**Action Expert 占 82%**，per-step ~16.5 ms。DiT scaling: 174M=7.2 ms < **300M=16.5 ms** < 350M=32 ms → cross-attn 使 300M Expert 比纯 DiT 贵 ~2.3x。**Bimodal 污染**：runs 1–12 慢 1.25x (GPU 功率爬坡)，定为后续 warmup=15 + `nvidia-smi -pm 1` 默认。
- **下一步**：作为 exp08 co-location 的 "A 阶段 300M" payload 候选。

## exp08a — EPDA pair-wise interference pilot · **done**

- **目的**：验证 roofline 预测（D+A strong, P+A weak），动机 EPDA disaggregation。
- **方法**：两 Python thread + 两 CUDA stream，strategy α (无 mid-loop barrier)，DA / PA pair。
- **结果**：DA: D ×3.52, A ×1.58；PA: P ×2.27, A ×1.15。**Roofline 严重低估 2–28x**（PA 预测 <10%，实测 127%）。GPU kernel dispatch contention 是 roofline 未捕捉的新瓶颈。
- **下一步**：扩到 full 6-pair matrix (exp08b)，拟合 contention 模型 (exp08c)。

## exp08b — EPDA 6-pair interference matrix · **done (pairs)** / **triples+quad pending**

- **目的**：完成 EPDA 所有 2-combo，量化 phase-pair 非对称干扰。
- **方法**：对 EP/ED/EA/PD/PA/DA 六对分别跑 warmup=15 iter=40 coloc，得 inflation ratio。
- **结果**：EP E×1.70/P×2.42；ED E×1.04/D×2.59；EA E×0.97/A×1.23；PD P×2.42/D×2.48；PA P×2.88/A×1.19；DA D×2.65/A×1.16。**D/P 极度脆弱 (2.4–2.9x) vs E/A 鲁棒 (<1.3x)**。
- **下一步**：跑 triples (EPD/EPA/EDA/PDA) + quad (EPDA) 以验证 exp08c M4 模型外推能力。

## exp08c — GPU contention model fit · **done (v1)**

- **目的**：拟合一个能从 pair 外推 triple/quad 的解析 contention 模型。
- **方法**：4 候选模型（M1 additive / M2 bottleneck / M3 empirical lookup / **M4 asymmetric v×a**），交替最小二乘 (ALS) 在 12 个 pair 观测上拟合。
- **结果**：**M4 R²=0.94**（M1=0.21，M2=-3.44，M3=1.0 查表上界）。公式 `inflation(X|Y) = 1 + v_X·a_Y`。学到 v=(D:1.52, P:1.61, E:0.23, A:0.20)；a=(A:1.12, E:0.96, P:1.02, D:0.87)。**A 是隐形破坏者**（自身鲁棒但对他人干扰最大）；**E 最安全 co-locate**。部署建议：{E,A} 同卡，{P,D} 必须 disaggregate。
- **下一步**：用 exp08b triple/quad 数据验证 M4 外推 → 若 R² 依旧 >0.9 即可作为 mechanism-study 论文核心。

## exp09a — Cosmos Policy direct-mode latency profiling · **ready**

- **目的**：测量 Cosmos Policy (Cosmos-Predict2-2B, LIBERO 98.5%) 的 direct-mode 推理延迟。论文 (arXiv:2601.16163) 零报告该数字，只说 planning 5s/chunk。
- **方法**：在 cosmos-policy Docker 环境里加 CUDA event timing 包裹三相位 (E/D/X)，合成 LIBERO 输入。canonical = warmup=15, iter=20, 5 步 denoise。附带 step sweep (1/2/5/10/20) 测 quality-speed frontier。
- **结果**：待跑。
- **下一步**：在 xdlab23 跑完后，和 Pi-Zero (200ms/5Hz) / Fast-WAM (407ms/2.5Hz) / LingBot-VA (2518ms/0.4Hz) 对比。若 pure DiT 每步 ~30-100ms，则 5 步 = 150-500ms → 候选 A (FastVideo 加速) 的 baseline。

---

## Quick glance (1 行 per exp)

| Exp ID | Motivation | Status | Key Finding |
|--------|-----------|--------|-------------|
| exp01a | Qwen2.5-VL-7B E/P/D baseline | **done** | text P=20/D=18; single_img E=253/P=156/D=18.6; multi_img E=541/P=332/D=21 ms |
| exp01b | Qwen2.5-VL-7B attention (5 layers) | **done** | Pos 2 universal sink (12–28x). Gini >0.91 → pruning viable. Layer 21 entropy min 3.44. |
| exp02a | ACT lightweight VLA E/A | **done** | Total ~3 ms (850x faster than VLM). Encode 80%, Action 20%. |
| exp03a | LingBot-VLA-4B flow VLA E/C/A | **done** | E=35.7/C=38.3/A=0.48 → 74.5 ms (~13 Hz). 3B backbone 7x faster than 7B. |
| exp04a | Fast-WAM (skip-imagination) E/C/A | **done** | @10step: 407 ms (2.5 Hz), Action 89%. Per-step ~32 ms (350M ActionDiT). |
| exp04b | LingBot-VA (full WAM) E/V/A | **done** | 2518 ms (0.40 Hz). Full WAM 6x slower than skip. Action 69%. |
| exp05a | LingBot-VLA attention (6 layers) | **done** | VLA 彻底重塑 attention: Gini 0.91→0.07, sink Pos 2→64, entropy flat. VLM pruning 不可迁移。 |
| exp05b | Qwen2.5-VL-3B vanilla attention (消歧) | **done** | 3B vanilla Gini 0.80–0.98 (≈7B) → Gini 崩塌归因于 VLA fine-tune 非 model size。 |
| exp06a | NitroGen 500M DiT k-sweep | **done** | Per-step 7.2 ms 完美线性, k=1 total 17.9 ms (55.9 Hz). 2x params → 4.4x latency。 |
| exp07a | Pi-Zero flow VLA E/C/A | **done** | E=9.32/C=26.40/A=164.76 → 200.5 ms (~5 Hz). Action Expert 82%. Per-step 16.5 ms. |
| exp08a | EPDA pair-wise pilot (DA/PA) | **done** | DA: D×3.52, A×1.58. PA: P×2.27, A×1.15. Roofline 低估 2–28x。 |
| exp08b | Full 6-pair EPDA matrix | **done (pairs)** | D/P 脆弱 (2.4–2.9x) vs E/A 鲁棒 (<1.3x)。Triples/quad pending。 |
| exp08c | GPU contention model fit | **done (v1)** | M4 asymmetric R²=0.94. inflation=1+v_X·a_Y. 部署: {E,A} 同卡 / {P,D} 必 disagg。 |
| exp04c | Fast-WAM 5-step paper-aligned | **done** | 257ms / 3.9Hz @ 5-step (paper 190ms on A100). Per-step ~41ms. |
| exp04d | LingBot-VA real-weight LIBERO eval | **planned** | ckpt found: robbyant/lingbot-va-posttrain-libero-long (22.7GB) |
| exp06b | NitroGen full 500M real-weight | **done** | 7.1ms/step — identical to exp06a (DiT=181M, not 500M). |
| exp07b | Pi-Zero real-weight profiling | **done** | 225ms total. vs exp07a 200ms (+12%). Random-weight timing ≈ faithful. |
| exp03b | LingBot-VLA LIBERO-4 eval | **planned** | 4B real ckpt, 20 ep/task × 4 suites |
| exp04e | Fast-WAM LIBERO-4 eval | **done** | 94.5% avg (spatial 91.5 / object 100 / goal 97 / 10 89.5), 800 ep, real ckpt 5-step |
| exp07c | Pi-Zero LIBERO-4 eval | **planned** | pi0-base real ckpt, 20 ep/task × 4 suites |

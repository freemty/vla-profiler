# Hao Zhang 工作风格综合分析：从 vLLM/DistServe/FastVideo 到我的下一步实验

> **目的**：整合三份 hao-style-*.md 文档，抽象出 Hao 的 co-design 方法论，并映射到我 VLM/VLA 研究方向的下一步实验选择。
> **编写日期**：2026-04-26
> **前置文档**：`hao-style-vllm.md`, `hao-style-distserve.md`, `hao-style-fastvideo.md`

---

## 1. Hao 的 Co-Design 方法论（三个工作共享的 DNA）

### 1.1 工作模板（每篇都遵循的五步法）

```
Step 1: Profile → 找到当前最大 bottleneck（必须量化到数字）
Step 2: 结构性论证 → 证明这个 bottleneck 不是调参能解决的
Step 3: 借用异学科抽象 → 找到已有成熟解法的"同构"问题
Step 4: 最小 co-design primitive → 算法 + 系统必须同时改
Step 5: 新指标 reframe → 重新定义评估边界
```

| 步骤 | vLLM (SOSP'23) | DistServe (OSDI'24) | FastVideo (ICML'25) |
|------|----------------|---------------------|---------------------|
| 1. Profile 数字 | "60-80% KV 内存浪费" | 测得 prefill vs decode 资源特征 | "attention alone: 800 of 945 seconds" |
| 2. 结构性论证 | 连续内存 vs 动态 KV 增长 | compute-bound vs memory-BW-bound | O(n²) 3D attention 不可避免 |
| 3. 借用抽象 | OS 虚拟内存分页 | 数据中心 disaggregation | 稀疏线性代数 / tile 计算 |
| 4. Co-design primitive | PagedAttention（内存+kernel） | Placement algo ↔ KV transfer | STA tile kernel ↔ GPU SRAM |
| 5. 新指标 | Throughput under latency | **Goodput**（SLO-attained throughput） | MFU + VBench@latency |

### 1.2 三个"反面"特征（Hao 不做什么）

理解 Hao 的工作风格，反面特征同样重要——这些是**不够 Hao-style** 的工作类型：

1. **纯 profiling 报告**：单独测 latency breakdown 不是 contribution，profiling 只是 motivation 的素材
2. **纯算法新架构**：光有"新模型更准"不够，必须解决系统层的结构性问题
3. **纯系统工程**：光优化 kernel 或调度，没有算法层配套改动，做不出 SOSP/OSDI-级别工作
4. **渐进式改进**：+5% throughput 的工作不写，必须是 2-10x 级别的 step change
5. **热门跟风**：他的每个工作都在领域 bottleneck 转移时第一个入场（LLM serving 火起来前做 vLLM、视频生成火起来前做 FastVideo）

### 1.3 三个工作的连贯性：bottleneck 转移的螺旋

```
Alpa (2022) → 训练时自动并行，serving 实验暴露推理内存问题
   ↓
vLLM (2023) → 解决单 GPU 的 KV 碎片化，但暴露了多 GPU prefill/decode 争用
   ↓
DistServe (2024) → 解决多 GPU PD 争用，但仅针对 LLM，未覆盖 video/diffusion
   ↓
FastVideo (2025) → 解决 video DiT 的 attention overhead
   ↓
VLM/VLA real-time systems (2026+)  ← 这是 Hao 正在做/你要对齐的下一步
```

**关键观察**：每一步都是**上一步工作暴露的下一个最大 bottleneck**。这不是规划出来的路线图，而是"认真做完当前工作，下一步自然显现"的研究态度。

---

## 2. 对 VLM/VLA Real-Time Systems 的映射

### 2.1 Hao 下一步可能做什么？（推测）

从三个工作的轨迹推断，Hao 的 VLM/VLA real-time 工作**大概率**围绕以下方向之一：

| 候选方向 | 类比 | 可能性 | 为什么 |
|---------|------|-------|-------|
| **EPD/EPDA Disaggregation** | DistServe → VLM | 高 | 三阶段异构更严重，goodput 潜力大 |
| **Visual KV Sparse Compression** | PagedAttention → VLM visual | 高 | Gini 0.91 极度稀疏是明显的结构矛盾 |
| **Diffusion Action Caching** | DiT caching ≈ PagedAttention for diffusion | 中-高 | WAM 类 VLA 的 A 阶段 89% latency 主导 |
| **VLA Attention Sparsification** | STA → VLA action DiT | 中 | 需要先验证 attention 局部性（exp05a 显示 Gini 崩塌 → 不确定） |
| **VLA Speculative Rollout** | Speculative decoding → Flow VLA | 中 | 少步蒸馏 + speculative verification |

### 2.2 我当前实验的诊断（exp01-06）

**现状**：全部实验都停留在方法论 **Step 1（Profile）**。

| 实验 | 完成度 | Hao-style 距离 |
|------|-------|---------------|
| exp01a/01b (Qwen2.5-VL profiling + attention) | Step 1 ✓ | 缺 Step 2-5 |
| exp02a (ACT baseline) | Step 1 ✓ | 缺 Step 2-5 |
| exp03a (LingBot-VLA-4B) | Step 1 ✓ | 缺 Step 2-5 |
| exp04a/04b (Fast-WAM / LingBot-VA) | Step 1 ✓ | 缺 Step 2-5 |
| exp05a/05b (VLA attention) | Step 1+2 部分 ✓ | 有结构性观察（Gini 崩塌），但未导向解法 |
| exp06a (NitroGen DiT) | Step 1 ✓ | 缺 Step 2-5 |

**诊断**：我有丰富的 Step 1 素材（这是**好事**——Hao-style 工作的前提就是扎实 profiling），但缺从"是什么"跳到"能不能"的那一步。下一个实验必须是 **Step 2→5 的跨越**。

### 2.3 三个候选的下一步实验（exp07 方向选择）

**按 Hao-style 方法论打分**：

#### 候选 A：**EPD 干扰量化 + Disaggregation motivation**（DistServe-style）
- **Step 1**: exp01a 已有单请求 E/P/D 数据
- **Step 2 (结构性论证)**: 需要量化 E+P+D 共置时的**相互干扰**——这是关键缺失数据
- **Step 3 (异学科抽象)**: 直接用 DistServe 的 disaggregation
- **Step 4 (co-design)**: EPD placement + visual feature transfer（不是 KV transfer）
- **Step 5 (新指标)**: EPD-goodput under VLM SLO
- **障碍**: 需要多 GPU serving 框架（vLLM 或自写），工程量大
- **对话 Hao 的价值**: 直接对齐 DistServe 主线，差异化空间（VLM 三阶段 > LLM 两阶段）
- **打分**: ⭐⭐⭐⭐ 正统延伸

#### 候选 B：**Visual KV 稀疏度 → 压缩机会**（vLLM-style）
- **Step 1**: exp01b 已有 Gini 0.91，但没连到显存占用
- **Step 2 (结构性论证)**: 证明 visual KV 中 top-k% token 的 KV 承载 >90% 有效信息，其余是 "memory fragmentation" 的新形式
- **Step 3 (异学科抽象)**: 稀疏表示 / learned compression / KV quantization
- **Step 4 (co-design)**: 稀疏感知的 visual KV block 重组 + 支持 sparse KV 的 attention kernel
- **Step 5 (新指标)**: KV-bytes-per-effective-visual-information
- **障碍**: exp05a 显示 VLA fine-tuning 后 Gini 崩塌 → 只适用于 VLM，不适用 VLA
- **对话 Hao 的价值**: PagedAttention 在 VLM 的直接类比，但范围有限
- **打分**: ⭐⭐⭐ 限于 VLM

#### 候选 C：**DiT 跨步 Feature 稳定性 → Action Caching**（FastVideo-style + DreamZero）
- **Step 1**: exp04a (Fast-WAM 362ms A) + exp06a (NitroGen 7.2ms/step) 已有素材
- **Step 2 (结构性论证)**: 测量 DiT 在相邻去噪步骤间的 layer activation 变化量——假设 late steps 高度冗余
- **Step 3 (异学科抽象)**: DiT caching（DreamZero 已有）+ consistency distillation
- **Step 4 (co-design)**: Step-aware feature cache + 可切换 kernel（dense vs cached-sparse）
- **Step 5 (新指标)**: Control-frequency@success-rate（类比 VBench@latency）
- **障碍**: DreamZero 已做了 DiT caching，需要差异化——可能的角度是**实时约束下**（< 50ms control loop）的 cache scheduling
- **对话 Hao 的价值**: 直接对齐 FastVideo + DreamZero 两条线，VLA 场景独特
- **打分**: ⭐⭐⭐⭐ 与 Hao 实验室现有工作最贴近

#### 候选 D：**EPDA 四阶段干扰量化**（DistServe + Diffusion 双引擎）
- 把 A 阶段（action denoising）作为第四个独立阶段，测量它与 E/P/D 的资源干扰
- 数据基础：exp04a 显示 A 占 89% → 是最值得分离的新阶段
- 这是候选 A 和候选 C 的融合版
- **打分**: ⭐⭐⭐⭐⭐ 最雄心但也最有效

### 2.4 推荐：候选 D（EPDA 四阶段）+ 候选 C 作为技术 enabler

**为什么是 D**：
1. 它同时继承 DistServe（分离思想）和 FastVideo（diffusion 优化）两条主线
2. A 阶段的独立性（independent compute graph, 89% latency）是**你独有的**观察，这是"Hao 还没做、但方法论上必然做的下一步"
3. EPDA 是对 EPD（arXiv:2501.05460 已做）的天然扩展——延伸"DistServe → VLM → VLA"这条线

**具体最小可行实验（exp07 proposal）**：

| 阶段 | 内容 | 对应方法论步骤 |
|------|------|---------------|
| exp07a | 测量 EPDA 四阶段在同 GPU 共置时的 SM / memory BW 干扰（用 nvtx + DCGM） | Step 1+2：motivation figure |
| exp07b | 测量 A 阶段（DiT）的 step-level feature stability（给 caching 铺垫） | Step 1+2：caching 的 structural evidence |
| exp07c | 提出 EPDA goodput 定义 + baseline co-located 违反分析 | Step 5：新指标 |
| exp07d (stretch) | 实现 minimal A-stage disaggregation + DiT caching | Step 4：co-design primitive |

exp07a-c 是 **写 position paper / workshop paper** 级别的产出；加上 exp07d 就是 **full systems paper** 级别。

---

## 3. 见 Hao 时的"一页话"

如果要压缩到一页纸（或 slide 开场）：

> **观察**：我 profile 了 VLA 全栈（VLM + VA + WAM + DiT），发现 WAM 类 VLA 的 action 阶段（diffusion DiT 去噪）占 89% latency，但它与 E/P/D 是完全独立的计算图，被串行执行。
>
> **结构性论证**：这是 DistServe 在 LLM 场景发现的 "PD 干扰" 问题的 VLA 四阶段版本。E/P 是 compute-bound，D 是 memory-BW-bound，A 是 iterative compute-bound——四者资源特征和 SLO 要求都不同。
>
> **类比与抽象**：
> - DistServe 提出 PD disaggregation + goodput → EPDA disaggregation + control-frequency-goodput
> - FastVideo 提出 STA + 蒸馏 → 实时 VLA 约束下的 DiT caching + step-skip
>
> **下一步**：exp07 系列，量化 EPDA 干扰和 DiT 跨步冗余，motivation figure 直接对标 DistServe Figure 1 和 FastVideo Figure 1。
>
> **想请教**：EPDA disaggregation 的关键难点是 visual feature 和 DiT latent 的高带宽传输（不是 LLM 的小 KV tensor），这在 Hao Lab 的 roadmap 里是否已经有布局？

这一段话的目的不是 "pitch 我的工作"，而是**证明我理解了你实验室的方法论连贯性**——这比做出任何 trivial 实验都重要。

---

## 4. 诚实标注

- **对 Hao "下一步做什么" 的推测** 基于公开工作轨迹，不等于内部规划。见面时应该问而不是断言
- **候选 D 的工程量很大**，exp07a-c（只做 motivation）是现实的，exp07d（完整 disaggregation）是 6-12 月的工作
- **候选 B 的 VLA 不适用性** 来自 exp05a 的 Gini 崩塌现象，但只测了 LingBot-VLA-3B 一个模型，不能强推广到所有 VLA
- **EPD Disaggregation (arXiv:2501.05460)** 已经存在，EPDA 是其自然扩展，差异化需要更仔细的比较（那篇论文是否已经隐含 A 阶段？需要精读）

---

## 附录：三份文档交叉引用

- **vLLM 视角**：`hao-style-vllm.md` §4.4 提出 "visual KV 静态存储 vs 稀疏利用" 矛盾（候选 B 的来源）
- **DistServe 视角**：`hao-style-distserve.md` §5 提出 EPD → EPDA 分解（候选 A+D 的来源）
- **FastVideo 视角**：`hao-style-fastvideo.md` §5.1 提出 Fast-WAM MoT cross-attn 的 STA 机会（候选 C 相关分析）

三份文档彼此补完，综合产出候选 D。

# Hao Zhang 方法论解析：vLLM / PagedAttention

> **调研目的**：深度分析 vLLM/PagedAttention 的 algorithm-system co-design 方法论，为 VLM/VLA real-time systems 研究方向对齐。这是 Hao Zhang 技术路线的"第一个里程碑"（Parameter Server → Alpa → **vLLM** → DistServe → FastVideo → VLM/VLA）。
> **编写日期**：2026-04-25
> **平行文档**：`hao-style-distserve.md`，`hao-style-fastvideo.md`

---

## 0. 论文清单

| 论文 | arXiv ID | 年份 | Venue | 核心 claim |
|------|----------|------|-------|------------|
| Efficient Memory Management for Large Language Model Serving with PagedAttention | 2309.06180 | 2023 | SOSP 2023 | near-zero KV 碎片化，吞吐提升 2-4x vs FasterTransformer/Orca |

**完整作者列表**（顺序）：Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, **Hao Zhang**, Ion Stoica

Hao Zhang 是倒数第二作者（第 8 位，共 9 人），Ion Stoica 压阵（通讯作者），Hao 与 Ion Stoica 共同指导（UCSD + UC Berkeley 联合项目）。一作 Woosuk Kwon 和二作 Zhuohan Li 是主要执行学生。

---

## 1. Core Contribution

### 1.1 Motivation：KV Cache 碎片化危机

vLLM 的起点是一个 profiling 观察：**现有 LLM serving 系统（FasterTransformer、Orca）中，KV cache 内存有高达 60-80% 被浪费**，浪费来源：

- **Internal fragmentation**：每个请求预分配最大序列长度的连续内存（worst case），但实际生成长度未知，导致预分配的内存中大量未使用。
- **External fragmentation**：不同请求的 KV cache 大小不同，随着请求的进出，物理内存出现大量碎片，无法分配给新请求。
- **Redundant duplication**：beam search、parallel sampling 等解码算法需要为同一前缀维护多份 KV cache 拷贝，无法共享。

这三种浪费共同导致 GPU 显存严重不足，batch size 被迫压低，吞吐量远未达到 GPU 的理论上限。

**为什么是结构性问题**：KV cache 的"动态增长"特性（AR 生成每步增加一个 token 的 KV，无法预知终止时长）与传统的"静态连续内存分配"范式存在根本矛盾。这不是调优参数能解决的，必须改变内存管理的基本抽象。

### 1.2 PagedAttention：OS 分页的直接借用

**核心洞察**：OS 虚拟内存管理中，进程不需要连续的物理内存，page table 负责 logical page → physical frame 的映射。KV cache 面临完全相同的问题结构——序列逻辑上是连续的，但物理上完全可以分散存储。

**算法层**：

```
Block Table（每请求的 KV 管理）:
  logical_block_id → physical_block_id
  每个 block 存储 block_size 个 token 的 KV（默认 16）

三种操作：
  Allocate: 按需分配 physical block，写入 block table
  Free:     请求完成时归还 physical block
  CoW:      Copy-on-Write，beam search 共享前缀 block，
            写时复制（reference count > 1 时触发）
```

**关键设计选择**：
- Block size 为 16（可配置）：太小则 block table 开销大，太大则内部碎片增加；16 是 attention CUDA kernel 的 tile 大小对齐边界
- Logical block 是序列的连续视图，physical block 可以任意散布——attention kernel 必须支持"非连续 KV"访问

**为什么是 co-design**：PagedAttention 的算法正确性（分页 KV）早有人想到，但实现效率取决于 attention kernel 是否能高效处理非连续的 KV block。vLLM 自写了支持 block table 寻址的 CUDA kernel（后来演进为接入 FlashAttention/FlashInfer），这是系统层的核心贡献。**单独改内存管理（不改 kernel）→ attention 计算大量随机内存访问，性能灾难；单独改 kernel（不改内存管理）→ 碎片问题依然存在。**

### 1.3 系统层：Block Manager + Scheduler

```
BlockManager（物理层）:
  - free_physical_blocks: 空闲 block 池
  - block_tables: per-request 的 logical→physical 映射表
  - ref_count: 每个 physical block 的引用计数（CoW 依赖）

  核心方法：
  can_allocate(seq_group) → 判断是否有足够空闲 block
  allocate(seq_group)    → 分配 block，建立映射
  free(seq_group)        → 归还 block，更新 ref_count
  fork(parent, child)    → beam 扩展：子序列共享父 block（ref_count++）
  copy_on_write(block)   → 写时复制（ref_count > 1 时新分配 block）
```

```
Scheduler（逻辑层）:
  三个队列：waiting → running → swapped
  
  调度策略：
  1. Preemption（抢占）：显存不足时，暂停低优先级请求，
     将其 KV cache swap 到 CPU（recompute 或 swap 策略）
  2. Continuous batching：请求完成即从 waiting 队列补充，
     不等待整个 batch 完成（与 Orca 相同，但 PagedAttention 使其更高效）
  3. Priority：FCFS 默认，可配置
```

**Preemption 机制**是 vLLM 显存管理的安全阀：当 KV block 池耗尽时，通过 swap 或 recompute 回收 block，保证系统不 OOM。这在原始 FasterTransformer 中是不存在的——系统要么 OOM 崩溃，要么保守地拒绝请求。

### 1.4 后续演进：哪些延续了 co-design 套路？

| 特性 | 引入版本 | Co-design 套路 | 说明 |
|------|---------|---------------|------|
| **Continuous batching** | v0.1 | 否（来自 Orca，纯调度） | 调度算法，不需要改 kernel |
| **Chunked prefill** | v0.3+ | 是 | 把长 prefill 切块，需要 kernel 支持部分 prefill 后的 KV 状态续算 |
| **Speculative decoding** | v0.4+ | 是 | Draft model + verifier 的 KV 分配联动，需要 block manager 支持投机分配和回退 |
| **Prefix caching** | v0.3+ | 部分 | 算法层（hash-based block 复用）+ block manager（跨请求共享 block） |
| **LoRA serving** | v0.3+ | 否（纯 adapter 权重管理） | 与 PagedAttention 解耦 |
| **PD Disaggregation** | v0.6+ | 是（→ DistServe 思路）| KV transfer connector + 调度拓扑改变 |

**结论**：Chunked prefill、Speculative decoding、Prefix caching 延续了 PagedAttention 的 co-design 套路（算法设计与 block 管理机制同步改）；Continuous batching 和 LoRA 是相对独立的工程改进。

---

## 2. 论文 + 代码实现细节

### 2.1 代码演进：v0 → v1 引擎重构

**v0（原始引擎，2023-2024）**：

```
vllm/core/block_manager.py      # BlockSpaceManager，管理 physical blocks
vllm/core/scheduler.py          # 调度逻辑（waiting/running/swapped 三队列）
vllm/attention/                  # Attention backend 抽象层（直接被 FastVideo 复用）
vllm/worker/                     # GPU worker，执行 forward pass
```

**v1（重构引擎，2024+）**：

```
vllm/v1/core/kv_cache_manager.py      # 重构后的 KV block 管理
vllm/v1/core/block_pool.py            # Physical block pool 抽象
vllm/v1/core/kv_cache_coordinator.py  # 多 KV cache group 协调
vllm/v1/core/kv_cache_metrics.py      # Metrics 收集（prefix cache hit rate 等）
vllm/v1/core/sched/scheduler.py       # 重构后的调度器
vllm/v1/attention/                    # v1 attention 系统（见下）
vllm/v1/kv_cache_interface.py         # KVCacheConfig / KVCacheBlocks 接口抽象
```

**v0 → v1 重构动机**（基于代码分析推断）：
1. 原 block manager 的数据结构不支持多 KV cache group（Multi-head Latent Attention 等变体需要不同 block size 的 group）
2. Prefix caching 在 v0 中是 patch-in 的，v1 将其作为一等公民设计
3. v0 的调度器和 block manager 有较强耦合，难以支持 disaggregated prefill 场景
4. v1 引入 `KVCacheBlocks` 作为调度器-管理器之间的清晰接口边界（减少内部数据结构泄漏）

### 2.2 Attention Backend 体系

```
vllm/v1/attention/
├── backend.py          # AttentionBackend ABC（抽象基类）
├── selector.py         # AttentionSelectorConfig + get_attn_backend()
│                       # 支持 use_mla, has_sink, use_sparse,
│                       #        use_mm_prefix 等特性 flag
└── backends/
    ├── flash_attn.py       # FlashAttention 2/3 backend（主流路径）
    ├── flashinfer.py       # FlashInfer backend
    ├── flex_attention.py   # PyTorch FlexAttention
    ├── triton_attn.py      # Triton kernel backend
    ├── tree_attn.py        # Tree attention（speculative decoding 专用）
    ├── flash_attn_diffkv.py # Diff KV size backend
    ├── mla/                # Multi-head Latent Attention (DeepSeek-V2/V3)
    └── ...（共 15+ backends）
```

**Attention kernel 演进**：
- **初期（v0.1-0.2）**：vLLM 团队自写 PagedAttention CUDA kernel（`csrc/attention/`），支持 non-contiguous KV block 访问
- **中期（v0.3+）**：接入 FlashAttention-2，但仍需要将 paged KV 转换为 FA2 兼容格式（有 reshape overhead）
- **现在（v1）**：FlashInfer 原生支持 paged KV（FIX: PagedKVCache），可以直接传入 block table，避免数据搬运；Triton backend 支持自定义稀疏 pattern

**这个 backend 插件体系被 FastVideo 直接移植**：`fastvideo/attention/selector.py` 的第一行注释明确指向 `vllm/v0.7.3/attention/selector.py`。这是 Hao 实验室在不同项目间复用基础设施的典型模式。

### 2.3 Prefix Caching 实现机制

Prefix caching 是 PagedAttention 的关键推论——如果 block 可以跨请求共享，那么相同前缀的 block 可以永久缓存：

```python
# v1/core/kv_cache_utils.py（概念）
# 每个 block 被 token hash 索引
# hash(tokens[0:block_size]) → physical_block_id

# 命中时：直接复用 physical block（ref_count++）
# 未命中时：正常分配 + 填写 hash 索引
```

**VLM 含义**（见 §5）：vision encoder 输出的 visual tokens 可以作为 prefix cache 的天然候选——相同图像的 KV 可以完全复用，无需重新 encode 和 prefill。exp01a 数据：single_img encode=253ms，若命中 prefix cache 则此部分归零。

---

## 3. Hao Zhang 的角色

### 3.1 作者位置与贡献

Hao Zhang 是第 8 作者（倒数第二），与 Ion Stoica 共同指导。主要执行者是 Woosuk Kwon（当时是 UCB PhD student，现 Google）和 Zhuohan Li（现在 Google DeepMind）。

从技术基因角度，Hao 的具体贡献领域：
- **系统设计方向判断**：将 OS 分页抽象引入 KV cache 管理——这是研究品味判断，而非工程执行
- **Serving 系统经验**：Alpa（2022）做的是训练时的自动并行化，但 Alpa 的 serving 实验暴露了推理阶段的内存问题；vLLM 是 Hao 从 Alpa serving 方向延伸的自然结果
- **评估框架设计**：Throughput vs Latency 的系统性 profiling，确立了 "KV 碎片化占比 60-80%" 这个可量化的 motivation

### 3.2 从 vLLM 到 DistServe 到 FastVideo 的延续

```
vLLM (SOSP 2023)
  → 发现：单 GPU 上 KV 内存碎片化是主要 bottleneck
  → 解法：PagedAttention（内存管理 primitive 重设计）
  → 遗留问题：多 GPU 场景下 prefill/decode 争用未解决

DistServe (OSDI 2024)
  → 发现：prefill/decode 在同 GPU 上有结构性资源争用
  → 解法：PD disaggregation（调度拓扑重设计）
  → 遗留问题：Video/Diffusion 模型根本没有 AR KV 增长问题

FastVideo (ICML 2025)
  → 发现：3D full attention 占 85% latency，具有空间局部性
  → 解法：STA tile-wise sparse attention（kernel + algorithm co-design）
```

每一步都是"用 profiling 找到当前最大 bottleneck，设计解决该 bottleneck 的最小必要系统-算法改动"。

---

## 4. Co-Design 方法论提炼

这是三份文档共享的方法论分析，从 vLLM 视角独立提炼。

### 4.1 借用其他学科成熟抽象——一致的套路

| 工作 | 借用的抽象 | 来源学科 | 为什么合理 |
|------|----------|---------|----------|
| vLLM / PagedAttention | 虚拟内存分页、page table | OS | KV cache 动态增长的问题结构与 OS 进程内存管理同构 |
| DistServe | Disaggregation（存储领域，如 DRAM-NVMe 分层） | 数据中心系统 | Prefill/decode 资源需求异构，与存储访问模式异构完全类比 |
| FastVideo / STA | Block-sparse / tile-based sparse attention | 稀疏线性代数 | GPU 的 SRAM tile 计算结构与 tile-wise 窗口天然对齐 |

**共同特征**：不是从头发明新算法，而是**识别出 ML 系统中某个子问题与已有学科中某个经典问题同构，然后迁移该学科中成熟的解法**。这种识别能力本身是方法论的核心——它要求研究者对 ML 以外的系统有足够的知识广度。

Hao 的技术路线从 Parameter Server（分布式系统）→ Alpa（编译器/自动并行）→ vLLM（OS 内存管理）→ DistServe（数据中心 disaggregation）→ FastVideo（稀疏线性代数）横跨了极宽的学科范围，这不是偶然的。

### 4.2 Bottleneck 识别：Profile vs Insight

三个工作的 bottleneck 识别方式有细微差异：

| 工作 | Bottleneck 识别方式 | 论文中的关键句子 |
|------|-------------------|---------------|
| vLLM | Profile 测量 + 理论分析 | "60-80% of memory is wasted due to fragmentation and redundant duplication" |
| DistServe | 计算特征分析（compute vs memory-BW bound），实测验证 | "prefill is compute-bound, decode is memory-bandwidth-bound" |
| FastVideo | Profile 直接测量 | "attention alone takes 800 out of 945 seconds" |

vLLM 的 bottleneck 识别稍微更理论性——"60-80% 碎片化"需要对内存分配模式做建模分析（不只是跑 profiler）。这是因为碎片化本身不直接显示在 profiling trace 中，需要对内存使用模式做推断。DistServe 和 FastVideo 的 bottleneck 更容易直接测量。

**共同点**：**bottleneck 必须是结构性的（structural），不是调参能解决的**。vLLM 的"碎片化"是内存分配范式决定的，DistServe 的"prefill/decode 干扰"是物理资源特征决定的，FastVideo 的"attention overhead"是 O(n²) 计算量决定的。三者都不是超参数问题。

### 4.3 Contribution 结构：算法 Primitive + 配套系统 + 新指标

这是 Hao 工作的标准模板：

```
贡献 = 核心 primitive（改最关键的那一个点）
     + 配套系统（使 primitive 在生产环境可用）
     + 新指标（量化 primitive 解决的问题）
```

| 工作 | 核心 Primitive | 配套系统 | 新指标 |
|------|--------------|---------|-------|
| vLLM | PagedAttention（分页 KV 管理） | BlockManager + Scheduler + Preemption | Throughput under latency constraint（KV 碎片化率 → 吞吐提升） |
| DistServe | PD Disaggregation（调度拓扑分离） | Placement Algorithm + KV Transfer + 双调度器 | Goodput（SLO 约束下的吞吐） |
| FastVideo | STA/VSA（tile-wise 稀疏 attention） | Triton kernel + backend 插件体系 | MFU + VBench@latency（质量约束下的加速比） |

**关键观察**：每个工作的"新指标"都是为了逼迫读者正视该工作解决的真实问题。vLLM 的"throughput under latency constraint"使"我的系统跑 10 个 req/s 但延迟是 30 秒"不再被视为成功；DistServe 的 goodput 使"吞吐高但违反 SLO"变得不可接受；FastVideo 的"VBench@latency"使"速度快但质量差"也不算数。**重新定义指标就是重新定义研究问题的边界**。

### 4.4 对 VLM/VLA 研究的 Punchline

**要做 PagedAttention 级别的 co-design，应该找什么样的结构性问题？**

PagedAttention 的成功来自：**找到了一个"已有系统设计决策与实际计算特征不匹配"的矛盾**（连续静态分配 vs 动态增长的 KV），并用一个成熟的异学科抽象（OS 分页）来解决它。

**VLM/VLA 里哪些现象是"类 KV 碎片化"级别的机会？**

从我们的实验数据出发，以下是候选的结构性矛盾：

| 现象 | 矛盾描述 | 类比 |
|------|---------|------|
| Visual KV cache 静态持续 | Visual tokens 完成 prefill 后 KV 不再更新，但每个 decode step 仍需读取全部 visual KV（exp01a: decode 18ms 中的大量内存带宽）| KV 碎片化：已分配的 block 中有大量"活着但无用"的数据 |
| VLA attention 结构突变 | VLM attention 的极度稀疏性（Gini 0.91）在 VLA fine-tuning 后完全崩塌（Gini 0.07），导致 VLM 的所有 attention 优化都无法迁移（exp05a/05b） | PagedAttention 前的系统：好的解决方案存在，但基础假设被打破 |
| WAM action denoising 独立 | Action DiT（exp04a: 89% latency）与 LLM backbone 是完全独立的计算图，但被串行执行，互相等待 | PD co-location：两个资源特征不同的阶段被强行绑定 |
| Diffusion 无 KV cache | Flow VLA / DiT 类模型根本没有 KV cache 概念（每步重新计算），但 vLLM 的整套 serving 基础设施都假设 KV cache 存在 | 将 paging 应用于不需要内存管理的系统：wrong abstraction |

**最有价值的切入口**（个人判断，不确定）：**"Visual KV cache 的低信息密度与其占用的显存不成比例"**——exp01b 显示 Gini >0.91，极度稀疏，但 visual KV 仍然占据完整的 KV block。如果能设计一个"稀疏度感知的 visual KV 压缩 + 重组"机制（类比 PagedAttention 对碎片化的解决），可能是 VLM 场景下的 PagedAttention 类比工作。

---

## 5. 到 VLM/VLA 的迁移启示

### 5.1 Visual KV 与 PagedAttention 的匹配性分析

PagedAttention 的设计假设：KV cache 随 AR 生成**动态增长**，内部+外部碎片化严重。

VLM visual tokens 的实际情况：
- **Visual tokens 是 prefill-only**：encode 完成后，visual token 的 KV 一次性写入，不再增长
- **无 AR growth 问题**：碎片化的"动态"来源消失，PagedAttention 针对 AR 增长的核心动机减弱
- **外部碎片仍然存在**：不同请求的 visual token 数量差异巨大（单图 vs 多图 vs 视频），不同大小的 visual KV block 组合同样产生外部碎片——这里 PagedAttention 仍然直接适用

**结论**：PagedAttention 对 VLM 的 text KV（AR 生成部分）完全适用，对 visual KV 的 AR growth 动机减弱但外部碎片动机仍然有效。vLLM 和 SGLang 的当前实现都把 visual token KV 和 text token KV 统一用 block manager 管理，这是正确的。

**新机会**：visual KV 的"prefill-only + 内容不变"特性带来了一个 PagedAttention 没有的机会——**visual KV 可以永久 prefix cache**（只要图像相同），而 text KV 的 prefix cache 只能 cache 相同 prompt 前缀，实际命中率远低于 visual KV。

### 5.2 Prefix Caching 在 VLM 中的价值

exp01a 数据：
- single_img encode = 253ms，prefill = 156ms
- 这 253ms 包括 ViT encode + visual feature 传给 LLM 后的 prefill attention

**Prefix caching 可以节省多少？**

```
场景 1：同一张图像 + 不同问题（多轮对话）
  → visual prefix 完全命中 → 节省 ~253ms encode + 156ms prefill（visual 部分）
  → 首次请求后，后续问题的 TTFT 从 ~409ms 降至 ~50ms（仅 text prefill）
  → 约 8x TTFT 改善（在高命中率下）

场景 2：系统 prompt（固定说明） + 变化图像
  → 文本系统 prompt prefix cache 命中
  → visual prefix 不命中（图像不同）
  → 节省部分较少
```

**现实限制**：prefix cache 命中率取决于相同图像的请求频率。在实时机器人控制（每帧新图像）的场景下命中率几乎为零；在交互式 AI 助手（反复问同一张图的不同问题）场景下命中率很高。对 VLM serving 系统研究，visual prefix caching 的收益分析是重要工作方向。

### 5.3 Flow VLA / Diffusion 类模型：vLLM 哪些不适用，哪些仍适用

**不适用的部分**：

| vLLM 机制 | 不适用原因 |
|----------|----------|
| PagedAttention（KV block 管理） | Diffusion DiT 在每步重新计算 Q/K/V，不存在跨步的 KV cache 积累 |
| Prefix caching | Diffusion 的 conditioning（文本/视觉）通过 cross-attention 注入，不是自回归 prefix，无法按 token hash 缓存 |
| AR Continuous batching | Diffusion 是固定步数去噪，没有"某个请求提前结束"的情况，batching 策略需要重新设计 |
| Preemption（KV swap） | 无 KV cache → 无 swap，预演（preemption）等于重算整个 denoising trajectory |

**仍然适用的部分**：

| vLLM 机制 | 在 Diffusion VLA 中的对应 | 具体场景 |
|----------|------------------------|---------|
| Attention backend 插件体系 | 直接复用（FastVideo 已示范） | 切换稀疏/密集 attention 实现 |
| Layerwise profiler | 直接复用 | DiT 每层耗时分解（exp06a 正是这样做的） |
| Batch scheduling | 需要重设计但概念沿用 | 多个并发机器人请求的 action denoising batching |
| Speculative 思想（草稿模型加速） | Speculative rollout（FastVideo Self-Forcing 的变体） | 少步蒸馏 = LLM speculative decoding 的 Diffusion 类比 |

**DiT caching 是 PagedAttention 的真正 Diffusion 类比**（参考 `dreamdojo-dreamzero-deep-dive.md`）：
- DreamZero 的 DiT caching 在去噪步骤间缓存中间 feature（而非 KV），在"feature 变化不大"的步骤间复用
- 这与 PagedAttention 的"跨请求共享 block"思想类似：找到可以安全共享/复用的中间状态，减少重复计算
- 但实现层面完全不同（block table vs timestep-aware feature cache）

### 5.4 具体实验建议

基于 vLLM 方法论，以下实验路径最直接有价值：

**实验 A：Visual KV Prefix Cache 命中率测量**
- **目标**：量化 VLM serving 中 visual prefix cache 的实际收益
- **方法**：在 Qwen2.5-VL 上模拟多轮对话 workload（同图不同问题 vs 不同图），测量 TTFT 在 cache hit vs miss 场景下的分布
- **数据基础**：exp01a 已有 single-request baseline（encode 253ms）
- **意义**：类比 vLLM prefix caching 论文中的 "cache hit rate vs TTFT" 分析，这是 EPD disaggregation 工作的配套实验

**实验 B：Visual KV 稀疏度与显存占用分析**
- **目标**：量化 visual KV block 中"有效信息"与"总占用"的比例
- **方法**：exp01b 已有 Gini 系数（0.91），进一步分析：top-k% 的 attention-heavy visual tokens 的 KV 对 decode 质量的贡献，以及其在 KV block 中的物理占比
- **假设**：visual KV 中 ~10% 的 token（高注意力权重）贡献了 90%+ 的查询值，剩余 90% 的 visual KV block 空间可以压缩
- **意义**：如果成立，这是"Visual-aware KV compression"工作的 motivation figure，类比 vLLM 的"60-80% KV 碎片化"

**实验 C：Flow VLA DiT 的跨步 Feature 稳定性分析**
- **目标**：测量 DreamZero/NitroGen DiT 在相邻去噪步骤间的 layer activation 变化量
- **方法**：对 exp06a 的 NitroGen DiT，记录每个 transformer layer 在 k=1,2,3,4 步的 hidden state L2 norm 变化
- **假设**：早期去噪步骤（t 大）的 feature 变化剧烈；后期步骤（t 小）变化趋缓，可以 cache
- **意义**：这是"DiT caching（类 PagedAttention）"工作的 profiling motivation，直接类比 vLLM 的 KV 碎片化量化

---

## 6. 方法论总结（三份文档对比）

| 维度 | vLLM | DistServe | FastVideo |
|------|------|-----------|-----------|
| **借用的抽象** | OS 虚拟内存分页 | 数据中心存储 disaggregation | GPU tile 计算 + 稀疏线性代数 |
| **Bottleneck** | KV 碎片化（理论分析 + 测量） | Prefill/decode 资源争用（计算特征分析） | Attention FLOPs 主导（直接测量） |
| **Co-design 核心** | PagedAttention kernel 支持非连续 KV | Placement algorithm ↔ KV transfer 带宽 | Tile-wise SWA ↔ SRAM 对齐 kernel |
| **新指标** | Throughput under latency | Goodput（SLO 约束吞吐） | MFU + VBench@latency |
| **主要贡献者** | Woosuk Kwon + Zhuohan Li (UCB/UCSD) | Yinmin Zhong (PKU→UCSD) | Peiyuan Zhang (UCSD) |
| **Hao 角色** | 共同指导（第 8 作者） | 通讯作者（最后一位） | 通讯作者（最后一位） |

**核心 punchline（对我研究的最重要启示）**：

> vLLM 的成功不在于 PagedAttention 算法有多复杂（思路来自 OS，30 年前就有），而在于**精确识别出"连续内存分配"这个隐含假设与"动态增长的 KV"这个现实之间的矛盾**，然后用最小的系统改动（换内存管理 primitive）解决了这个矛盾。
>
> 对 VLM/VLA 研究的类比启发：**先找到"系统现有的隐含假设与 VLM/VLA 的实际计算特征最不匹配的那一个点"，那个点就是下一个 PagedAttention 的位置。** 基于目前实验，我的候选是：visual KV cache 的"静态完整存储"假设与其极度稀疏的注意力利用率（Gini 0.91）之间的矛盾。

---

## 附录：术语速查

| 术语 | 含义 |
|------|------|
| PagedAttention | vLLM 的 KV cache 分页管理机制，logical block → physical block 映射 |
| Block Table | Per-request 的 logical block ID → physical block ID 映射表 |
| CoW | Copy-on-Write，beam search 场景下 KV block 的写时复制机制 |
| Block Size | 每个 KV block 存储的 token 数，默认 16（与 GPU tile size 对齐） |
| Continuous Batching | 请求完成即补充新请求，不等整个 batch 完成（Orca 提出，vLLM 采用） |
| Chunked Prefill | 将长 prefill 切块执行，减少对 decode 请求的延迟影响 |
| Prefix Caching | 相同前缀的 KV block 跨请求复用（hash-based） |
| Preemption | 显存不足时暂停低优先级请求（swap 到 CPU 或 recompute） |
| Goodput | DistServe 引入：满足 SLO 约束下每 GPU 能服务的最大请求率 |
| Visual Prefix Cache | VLM 中，相同图像的 visual token KV 的跨请求缓存 |
| EPD | Encode-Prefill-Decode，VLM 的三阶段执行流 |
| DiT Caching | Diffusion Transformer 去噪步骤间的 feature 复用（Diffusion 版 PagedAttention） |

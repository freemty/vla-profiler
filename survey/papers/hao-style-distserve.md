# Hao Zhang 方法论解析：DistServe

> **调研目的**：深度分析 DistServe 的 algorithm-system co-design 方法论，为 VLM/VLA real-time systems 研究方向对齐。
> **编写日期**：2026-04-25
> **平行文档**：`hao-style-fastvideo.md`，`hao-style-vllm.md`

---

## 0. 论文清单

| 论文 | arXiv ID | 年份 | Venue | 核心 claim |
|------|----------|------|-------|------------|
| DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving | 2401.09670 | 2024 | OSDI 2024 (pp. 193-210) | PD disaggregation 消除干扰，goodput 提升 7.4x，SLO 收紧 12.6x |

作者列表（完整）：Yinmin Zhong, Shengyu Liu（Peking University）; Junda Chen（UC San Diego）; Jianbo Hu（Peking University）; Yibo Zhu（StepFun）; Xuanzhe Liu, Xin Jin（Peking University）; **Hao Zhang**（UC San Diego，最后一位，通讯作者）。

---

## 1. Core Contribution

### 1.1 问题发现：Prefill-Decode 干扰

DistServe 的起点是一个 profiling 观察，而非先验猜测：**prefill 和 decode 两个阶段在同一 GPU 上共存时存在严重干扰**。

- **Prefill**：compute-bound。处理用户 prompt 的所有 token，做一次完整的 attention forward pass。GPU 算力 100% 满负荷，是 TTFT（Time To First Token）的决定因素。
- **Decode**：memory-bandwidth-bound。每步只生成一个 token，但需要读取所有 layer 的 KV-cache。GPU 算力利用率低，是 TPOT（Time Per Output Token）的决定因素。

这两者的计算特征**结构性不同**——不是调参能解决的，而是硬件资源需求从根本上就是异构的。现有系统（如原始 vLLM）将两者 co-locate 在同一 GPU batch 中，导致：
1. Prefill 请求因为 decode 请求占用资源而延迟（TTFT 变差）
2. Decode 请求因为 prefill 突然加入 batch 而被打断（TPOT 抖动）
3. 两者的并行配置（tensor parallelism 的 degree）被强行绑定，无法各自最优

### 1.2 核心解法：Disaggregation

DistServe 将 prefill 和 decode **分配到不同的 GPU 实例**：
- **Prefill 实例**：高算力配置（tensor parallel degree 高），专职处理 prompt
- **Decode 实例**：高内存带宽配置（pipeline parallel 优化 KV 读取），专职生成 token

核心机制：请求进来后，先在 prefill 实例上计算，生成 KV-cache；然后 KV-cache 通过 NVLink 或 InfiniBand **传输到 decode 实例**，在那里继续 autoregressive 生成。

### 1.3 Goodput 新指标

这是 DistServe 的方法论贡献之一，不只是一个系统 trick。

传统 LLM serving 用 **throughput**（tokens/s）衡量性能，但 throughput 可以通过堆 batch 来提升，代价是延迟飙升，用户体验变差。

DistServe 引入 **goodput**：
> "the maximum request rate a system can process while meeting all SLO constraints on TTFT and TPOT"

即：在满足延迟 SLO 的前提下，每 GPU 每秒能处理多少请求。这个指标把**延迟约束变成了分子的一部分**，而不是事后检验。

这种指标设计本身就是一个 claim：
- 证明了两个指标（TTFT 和 TPOT）之间存在 trade-off
- 使 7.4x goodput 提升这个数字变得有意义（单纯的 throughput 提升可能只是以违反 SLO 为代价）

### 1.4 Placement Algorithm

给定集群的 GPU 数量和带宽（NVLink vs IB），DistServe 需要决定：
- 多少 GPU 分配给 prefill？多少分配给 decode？
- 每个阶段内部用什么 parallelism 配置（TP/PP degree）？

论文采用的是**基于 profiling 的搜索（search over configurations）**：
1. 离线 profile 不同 parallelism 配置下的 prefill/decode 吞吐量
2. 给定 TTFT 和 TPOT SLO，搜索最优的 GPU 分配比例
3. 考虑 KV-cache 传输的带宽开销（NVLink 内置节点之间传输比 IB 快 10x+）

代码层面：`distserve/context_stage_scheduler.py` 和 `distserve/decoding_stage_scheduler.py` 分别管理两个阶段的调度逻辑，底层执行用 C++ 库 SwiftTransformer（支持 FlashAttention、PagedAttention、Continuous Batching）。

### 1.5 代价与局限

PD disaggregation 不是免费午餐：
- **KV-cache 传输开销**：KV-cache 从 prefill GPU 传到 decode GPU 需要网络时间。NVLink 约 600GB/s，IB 约 50-100GB/s，传输一个 7B 模型的 single request KV-cache 需要 1-5ms（取决于带宽和 sequence 长度）
- **资源碎片**：prefill GPU 在 decode 阶段是空闲的，反之亦然；利用率不如 co-locate 方案
- **部署复杂度**：需要维护两个实例池，调度更复杂

但论文证明：在 SLO 严格的场景下（例如 TTFT < 200ms，TPOT < 50ms），goodput 的提升远远超过了这些代价。

---

## 2. 论文 + 代码实现细节

### 2.1 仓库结构

```
DistServe/
├── distserve/
│   ├── context_stage_scheduler.py    # Prefill 阶段调度器
│   ├── decoding_stage_scheduler.py   # Decode 阶段调度器
│   ├── api_server/
│   │   └── distserve_api_server.py   # 统一入口，透明路由
│   └── ...
├── SwiftTransformer/                  # C++ 推理后端（git submodule）
│   ├── src/model/                     # GPT/OPT/LLaMA 实现
│   └── src/kernel/                    # FlashAttention, PagedAttention
└── examples/
    ├── offline.py                      # 离线推理示例
    └── online.py                       # 在线推理 + 性能测试
```

关键设计：**Ray 作为分布式 worker 框架**，prefill 和 decode 实例作为独立 Ray actor 运行，KV-cache 传输通过 Ray object store 或直接 RDMA 实现。

### 2.2 Placement 算法的实现细节

DistServe 的 placement algorithm 核心是一个双层搜索：

**层 1：给定 prefill-to-decode GPU 比例，分别优化各阶段内部的并行配置**
- 枚举 TP degree ∈ {1, 2, 4, 8}，PP degree ∈ {1, 2, 4}
- 对每种配置，用实测的 throughput 数据（不是理论模型）验证 SLO 是否满足

**层 2：搜索 prefill/decode 的 GPU 分配比例**
- 目标：最大化 goodput（满足 SLO 的情况下单 GPU 请求吞吐）
- 约束：KV-cache 传输带宽不成为瓶颈

这是一个**启发式搜索**而非精确优化，但实验表明在常见配置下搜索空间较小（几十到几百种候选），可以在分钟级完成。

### 2.3 DistServe 思想在 vLLM 中的落地演进

DistServe 论文发表时（2024 年 1 月），vLLM 不支持 PD disaggregation。到 2025-2026 年，vLLM 已经原生支持此功能：

**vLLM v0.6.x+ 的 disaggregated prefill**：
- 路径：`vllm/distributed/kv_transfer/`，包含多个 KV 传输 connector（p2p, mooncake, nixl, lmcache, moriio）
- API：`vllm serve --disagg` 子命令，配合 `vllm/entrypoints/serve/disagg/` 入口
- KV connector 抽象允许不同后端（NVLink、IB、甚至 CPU 内存）通过统一接口传输

**与 DistServe 原版的关键差异**：
1. DistServe 使用独立的 SwiftTransformer C++ 后端；vLLM 在自己的 Python/CUDA 体系内实现，复用率高
2. DistServe 的 placement algorithm 是离线搜索；vLLM 目前更偏向用户手动配置 prefill/decode 实例比例
3. vLLM 的 `--enable-chunked-prefill` 是另一个相关特性：将长 prefill 分块，减少对 decode 的突发影响，是 co-locate 场景下的 softer 版本

---

## 3. Hao Zhang 的角色

Hao Zhang 是 DistServe 的最后一位作者（UC San Diego），即指导教授 / 通讯作者角色。

**技术 DNA 视角：**

| 工作 | 时间 | Hao 的贡献层次 |
|------|------|---------------|
| Parameter Server（CMU） | 2014 | 分布式 ML 训练中的资源调度 |
| Alpa（CMU/UCB） | 2022 | 自动并行化，跨 device 资源分配 |
| vLLM | 2023 | LLM 内存管理（PagedAttention），**单点内存优化** |
| DistServe | 2024 | **分布式资源分离**，prefill/decode 解耦 |
| FastVideo | 2025 | 算子级稀疏化（STA/VSA）|

从 vLLM 到 DistServe 的演进是一个自然延伸：vLLM 优化了单 GPU 上的 KV-cache 内存管理，DistServe 发现下一个瓶颈是多 GPU 场景下两个阶段的资源争用。这种**分析-发现-解决-再分析**的螺旋式递进是 Hao 实验室一贯的风格。

主要学生贡献者：Yinmin Zhong（PKU 一作，当时是 PKU→UCSD 联合培养）。PKU 的 Xuanzhe Liu 和 Xin Jin 提供系统方向指导。合作院校：PKU（主要执行方）+ UCSD（Hao Zhang 指导）+ StepFun（Yibo Zhu 提供资源支持）。

---

## 4. Co-Design 方法论提炼

### 4.1 从 Profiling 发现"结构性干扰"

DistServe 最重要的贡献不是技术实现，而是**命名并量化了一个此前被混淆的问题**。

在 DistServe 之前，LLM serving 社区普遍知道 prefill 和 decode 性能有矛盾，但没有人明确说：**这两者的资源需求从物理上就是不可调和的，必须物理分离**。

DistServe 的论证链：
1. Profile → prefill 是 compute-bound，decode 是 memory-BW-bound
2. 两者的最优并行配置不同：prefill 喜欢大 TP（计算并行），decode 喜欢小 TP 大 PP（减少 attention 计算，流水线 KV 读取）
3. 因此：任何共存方案都是对两者的妥协

这种"**从 profiling 到 structural argument**"的推理路径，在 FastVideo 的 STA 中再次出现（attention score 局部性 → tile-wise 计算必然性）。这是 Hao 工作的核心方法论特征。

### 4.2 指标 Reframing 是 Contribution 的一部分

"Goodput" 这个词在 networking 领域早已存在，DistServe 将它引入 LLM serving 并重新定义，是一种**研究品味的体现**：

> 不是"我的系统吞吐量是 X tokens/s"
> 而是"在 TTFT < T1 且 TPOT < T2 的约束下，每 GPU 可以服务多少请求/秒"

这两种表述讲的是同一件事，但第二种逼迫评估者正视延迟约束。**重新定义指标就是重新定义问题边界**，这是 top-venue systems 论文常见的贡献形式。

类比：vLLM 的核心贡献其实也是"重新定义 KV-cache 管理"——把"OOM 导致请求 reject"转化为"memory fragmentation loss"这个可量化的指标。

### 4.3 算法（Placement）与系统（KV Transfer）的耦合

DistServe 的 placement algorithm 和 KV transfer 机制是**强耦合**的：
- placement 算法搜索配置时，把 KV 传输带宽作为约束条件
- KV 传输机制决定了哪些配置是可行的（NVLink 节点内 vs IB 跨节点）
- 集群拓扑（几台机器、什么互联）直接影响最优的 prefill/decode 比例

这不是"先设计算法、再实现系统"的串行思路，而是算法和系统在设计阶段就互相约束——这是 co-design 的本质。

### 4.4 Disaggregation 作为通用范式

回顾 Hao 的工作，"分离"是一个反复出现的主题：

| 工作 | 分离的对象 | 动机 |
|------|---------|------|
| vLLM | KV-cache pages vs model weights | 减少内存碎片 |
| DistServe | prefill GPUs vs decode GPUs | 消除计算特征不匹配的干扰 |
| FastVideo | full-attention vs tile-attention | 对齐计算粒度与 GPU 内存层次 |

这种模式可以总结为：**找到两个被错误绑定在一起的计算阶段，证明它们的资源需求本质不同，然后设计新的抽象层把它们分离**。

---

## 5. 到 VLM/VLA 的迁移启示

### 5.1 从 PD 到 EPD：三阶段分离的动机

DistServe 解决了 LLM 的 P/D 二阶段问题。VLM 在此基础上增加了 **Encode 阶段**（视觉编码器），使问题变成三阶段，且三者的计算特征异构程度更严重：

| 阶段 | 计算特征 | 我们的实验数据 | 瓶颈类型 |
|------|---------|-------------|---------|
| **E（Encode）** | Vision encoder（ViT-style），dense compute，无 KV-cache | exp01a: single_img=253ms, multi_img=541ms | compute-bound（GEMM 密集） |
| **P（Prefill）** | LLM attention 对 visual+text tokens，prefill 阶段 | exp01a: single_img=156ms, multi_img=332ms | compute-bound（attention 主导） |
| **D（Decode）** | Autoregressive token 生成，KV-cache 读取 | exp01a: 18-21ms/token | memory-BW-bound |

从 exp01a 数据：
- **E 占 E+P 总时 61%**（single_img: 253/(253+156)=62%，multi_img: 541/(541+332)=62%）
- E 的计算强度（FLOPS/Byte）与 P 相近但略高，两者都是 compute-bound
- D 的强度远低于 E 和 P，是典型 memory-BW-bound

**EPD 三阶段分离的 goodput 提升潜力**：

DistServe 在 PD 两阶段场景下实现了 7.4x goodput 提升，核心是消除了两个阶段各自约占 50% 时间的资源争用。VLM 的 EPD 场景中：
- E 和 P 都是 compute-bound，可以共享高算力 GPU（相互干扰相对小）
- D 是 memory-BW-bound，与 E/P 的干扰程度类比 DistServe 中的 P/D 干扰
- 因此 EPD 的潜力略小于 PD（因为 E 和 P 可合并为"大 Prefill"），但 E 的引入带来了**新的 batching 机会**

关键洞察：**E 阶段的 batching 效率极低**——每张图像的 ViT 计算是独立的，不同请求的 visual token 数量差异巨大（单图 253ms，多图 541ms）。把 E 单独分离出来，可以对 E 做更激进的批处理（例如同类图像 size 的请求合并）。

### 5.2 已有 EPD 工作调研

landscape.md §1.3.2 已经列出了若干 EPD 工作。这里做 DistServe 视角的解读：

| 系统 | 年份 | 与 DistServe 的关系 | 关键差异 |
|------|------|-------------------|---------|
| EPD Disaggregation (arXiv:2501.05460) | 2025 | **最直接的 DistServe → VLM 迁移** | ICML 2025，提出 multimedia token caching，15x 内存降低，90-100% SLO 提升 |
| HydraInfer (arXiv:2505.12658) | 2025 | 混合 EPD 调度 | 支持 stage-level batching，4x 吞吐于 vLLM，8xH800 验证 |
| RServe (arXiv:2509.24381) | 2025 | E/P 的 pipeline overlap | 不完全分离，而是 overlap——更保守但实现更简单 |
| EPD-Serve (arXiv:2601.11590) | 2026 | EPD on Ascend (华为) | 57-69% 吞吐提升，国产芯片适配 |
| xLLM (arXiv:2510.14686) | 2025 | 动态 EPD 策略 | 根据负载动态切换 co-locate/disaggregate |

**关键发现**：确实存在明确的 "DistServe 推演到 VLM" 的工作，最重要的是 arXiv:2501.05460（EPD Disaggregation，ICML 2025）。搜索"DistServe VLM"并无同名论文，但上述工作已经是实质等价的迁移。

**EPD Disaggregation (2501.05460) 的具体新贡献**（超越 DistServe 的部分）：
1. **Multimedia token caching**：encode 出的 visual token 可以缓存，相同图像无需重新 encode（类比 vLLM 的 prefix caching）
2. **Intra-request encode parallelism**：单个请求内，多张图像可以并行 encode（不同于 LLM 的 prefill 必须串行处理 token）
3. **Role-switching**：负载变化时，decode 实例可以动态切换角色为 encode 实例（比 DistServe 更弹性）

### 5.3 VLA 扩展：第四阶段问题

VLA 在 EPD 基础上引入 **Action 阶段**，形成 EPDA 四阶段问题：

```
E (Vision Encode) → P (LLM Prefill) → D (LLM Decode / 生成 action token 或 latent) → A (Action Head)
```

从我们的实验数据分析各阶段：

| 阶段 | 代表实验 | 延迟 | 占比 | 特征 |
|------|---------|------|------|------|
| E | exp03a (LingBot-VLA-4B) | 35.7ms | 48% | compute-bound，与 VLM 相同 |
| P/C | exp03a (Context attention) | 38.3ms | 51% | compute-bound |
| A | exp03a (Flow action head) | 0.48ms | 1% | **可忽略（3B 场景）** |
| E | exp04a (Fast-WAM, @10step) | 7.6ms | 2% | 已 overlap |
| A | exp04a (action denoising) | 362ms | **89%** | memory/compute-bound（10步去噪） |

**关键差异**：
- **Autoregressive VLA（OpenVLA）**：A 阶段只是 D 阶段的几个额外 token decode，不需要单独分离
- **Flow VLA（LingBot-VLA, Pi-Zero）**：A 阶段是单步 flow head（~0.5ms），极轻量，不构成新瓶颈
- **WAM（Fast-WAM, DreamZero）**：A 阶段是多步 DiT 去噪（exp04a: 362ms，89% 占比），是**主导瓶颈**

**结论**：对于 WAM 类 VLA，A 阶段（diffusion action denoising）是最值得做 disaggregation 的新阶段，因为它：
1. 独立于 E/P/D，有自己的 DiT 架构和计算特征
2. 是系统的主要瓶颈（89%）
3. 可以做独立的 caching（DiT KV cache，类比 DiT caching 论文）

### 5.4 给当前实验的启示

基于 DistServe 的方法论，以下实验最能推进 EPD / EPDA 研究方向：

**实验 A：EPD 干扰量化**
- **目标**：测量 E、P、D 阶段在 co-locate 场景下的相互干扰程度
- **方法**：对 Qwen2.5-VL 做并发 serving，记录 TTFT 和 TPOT 随 batch_size 的变化；分别用纯 encode-heavy 和纯 decode-heavy 负载做基准
- **期望发现**：像 DistServe 那样，展示 co-location 导致两个指标此消彼长，无法同时优化
- **实验价值**：这是做 EPD disaggregation 工作的必要前提实验

**实验 B：GPU Utilization Breakdown**
- **目标**：用 `torch.profiler` 或 DCGM 可视化 E/P/D 阶段的 SM utilization、memory bandwidth 利用率
- **方法**：在 exp01a 数据基础上增加 per-phase GPU 利用率记录（加 `nvtx` range markers）
- **期望发现**：E 阶段 SM ~90%（compute-bound），D 阶段 SM ~20%（memory-BW-bound）
- **实验价值**：直接可视化为 DistServe-style 的"为什么分离"的 motivation figure

**实验 C：SLO 违反分析（Co-located Baseline）**
- **目标**：在不同负载下，展示 co-locate 方案的 SLO 违反率
- **方法**：设定 TTFT < 500ms，TPOT < 30ms 的 SLO；用 vLLM default 配置做在线 serving，记录违反率
- **期望发现**：在中等负载（>4 concurrent requests）下，TTFT 或 TPOT 必然有一个开始违反 SLO
- **实验价值**：直接对标 DistServe Figure 1 的 motivation figure，可作为 EPD 工作的开场图

**实验 D：E 阶段 Batching 效率分析**
- **目标**：测量 vision encode 的 batching 效率（是否存在 batch-friendly 的场景）
- **方法**：exp01a 已有 batch_size=1 数据；增加 batch_size=2,4,8 的 encode 时间测量
- **期望发现**：encode 的 batching efficiency 高（compute-bound，batch 利用 GPU 算力），但 KV-cache 后续 prefill/decode 因 batch size 增大而干扰加剧
- **实验价值**：量化 E 阶段独立 batching 的潜力（EPD 中 E 可以用大 batch 提效）

### 5.5 从 PD 到 EPD 的 goodput 提升预估

基于 exp01a 数据做 back-of-envelope 分析：

- VLM single_img：E=253ms，P=156ms，D=18ms/tok（decode 50 tokens = 900ms）
- co-locate 场景：E+P 阶段（409ms）与 decode 阶段（900ms）存在争用
- E 阶段比 DistServe 的 prefill 多出一个 253ms 的 compute-bound 开销

如果按 DistServe 的方法论推算，EPD disaggregation 的 goodput 提升预期：
- **保守估计**：2-3x（仅消除 D 和 E/P 的干扰，类比 DistServe 的 2-3x 基础提升）
- **乐观估计**：4-8x（如果 E 阶段能做独立批处理，进一步摊薄 encode 延迟）

这个估计比 DistServe 的 7.4x 可能略低，因为 VLM 的 decode 阶段相对较短（18ms vs 数百ms），D 对 E/P 的干扰相对小；但 E 阶段的独立优化（caching、batching）带来额外增益。

---

## 6. 方法论总结

DistServe 对我的研究的最重要启示可以压缩为三条：

**1. 先证明干扰是结构性的（不是参数问题）**
- EPD 的研究不能从"设计一个更好的调度器"开始，要从"证明 E、P、D 阶段在同一 GPU 上有不可调和的资源冲突"开始
- 实验 A（干扰量化）是先决条件

**2. 用新指标 reframe 问题**
- "VLM throughput"没意义；"在 TTFT < 500ms 约束下每 GPU 能处理多少带图请求"才有意义
- Goodput 指标直接从 DistServe 迁移到 VLM serving 是完全合法的

**3. 算法（placement）和系统（E/P/D 传输）必须协同设计**
- EPD 中 E → P 的 visual feature 传输（不是 KV-cache，而是 encoder output）是新的通信原语
- 如何在 encode 实例和 prefill 实例之间高效传输 visual features（大 tensor，压缩？量化？），是 DistServe KV transfer 机制的 VLM 版本

---

## 附录：术语速查

| 术语 | 含义 |
|------|------|
| TTFT | Time To First Token，从请求发出到第一个 token 返回的延迟，prefill 阶段决定 |
| TPOT | Time Per Output Token，每个生成 token 的平均延迟，decode 阶段决定 |
| Goodput | 满足 TTFT+TPOT SLO 的情况下，每 GPU 能处理的最大请求率 |
| PD Disaggregation | Prefill-Decode 分离，DistServe 的核心机制 |
| EPD | Encode-Prefill-Decode，VLM 三阶段分离 |
| EPDA | Encode-Prefill-Decode-Action，VLA 四阶段（WAM 场景） |
| KV Transfer | KV-cache 从 prefill 实例传输到 decode 实例的机制 |
| Placement Algorithm | 决定 GPU 资源如何在 prefill/decode 实例间分配的搜索算法 |
| SwiftTransformer | DistServe 使用的 C++ 推理后端（现已被 vLLM 整合） |

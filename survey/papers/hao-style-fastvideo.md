# Hao Zhang 方法论解析：FastVideo

> **调研目的**：深度分析 FastVideo 的 algorithm-system co-design 方法论，为 VLM/VLA real-time systems 研究方向对齐。
> **编写日期**：2026-04-25
> **平行文档**：`hao-style-distserve.md`，`hao-style-vllm.md`

---

## 0. 论文清单

| 论文 | arXiv ID | 年份 | Venue | 核心 claim |
|------|----------|------|-------|------------|
| Fast Video Generation with Sliding Tile Attention | 2502.04507 | 2025 | ICML 2025 | STA 替换 3D full attention，attention 加速 2.8-17x，HunyuanVideo 945s→268s |
| VSA: Faster Video Diffusion with Trainable Sparse Attention | 2505.13389 | 2025 | NeurIPS 2025 | 可训练稀疏注意力，training FLOPS 削减 2.53x，Wan2.1 端到端 31s→18s |

两篇论文共享 `hao-ai-lab/FastVideo` 代码库，Hao Zhang 均为通讯/共同通讯作者，主要学生贡献者为 Peiyuan Zhang。

---

## 1. Core Contribution

### 1.1 背景：3D Full Attention 的成本危机

Video DiT（HunyuanVideo、Wan2.1、CogVideoX）以三维时空 token 序列作为 attention 输入。生成一段 5 秒 720P 视频，3D full attention 耗时 **800 秒（占总 945 秒的 85%）**。这是 FastVideo 整个工作的出发点——profiling 观察确立了唯一目标：attention 就是 bottleneck。

### 1.2 STA（Sliding Tile Attention）

**论文**：arXiv 2502.04507，ICML 2025

**Motivation / Profiling 观察**：  
预训练视频 DiT 的 attention score 在 3D 空间内具有极强的局部性——attention mass 主要集中在时空邻域内。这是一个 trained model 上的实验性观察，而非先验假设。

**改在哪一层**：  
- 算法层：局部 3D sliding window 替换 global attention  
- Kernel 层：**tile-by-tile 而非 token-by-token** 的新 sliding window 设计，配合 Triton 自定义 kernel

**核心设计**：  
传统 SWA（sliding window attention）以 token 为粒度移动窗口，无法利用 GPU 的 block-level 计算效率。STA 的关键创新是以 **tile**（一组 token 的矩形块）为移动粒度，确保每次 kernel 调用处理的 KV 区域与 GPU SRAM 大小对齐，实现 **58.79% MFU**（FlashAttention 级别）。

**加速比**：
- 相比 FA2：2.8-17x attention 加速
- 相比 FA3：1.6-10x
- HunyuanVideo 端到端：945s → 685s（无需训练），fine-tuning 后 → 268s（VBench 下降 0.09%）

**代价**：  
STA 是局部近似，远距离时空依赖被截断。无 fine-tuning 时质量下降，fine-tuning 后接近无损（0.09% VBench gap）。

### 1.3 VSA（Video Sparse Attention）

**论文**：arXiv 2505.13389，NeurIPS 2025

**Motivation / Profiling 观察**：  
大部分 attention mass 集中在少量"关键 token"上。不同于 STA 的固定窗口，VSA 动态识别每个 query tile 对应的高权重 key tiles。

**改在哪一层**：
- 算法层：两阶段 coarse-to-fine 稀疏化
- Kernel 层：可微分 Triton kernel，支持端到端训练

**核心设计**：
1. **Coarse stage**：将 token pool 成 tile，通过 pooled QK 相似度识别"critical tokens"
2. **Fine stage**：在 critical token 范围内做 token-level attention，利用 block computing layout 保证硬件效率
3. **可微分**：整个过程端到端可训练，无需 post-hoc profiling 选 sparsity pattern

**加速比**：
- Training FLOPS 削减 2.53x（diffusion loss 无下降）
- Wan2.1 attention 加速 6x，端到端 31s → 18s
- 85% FlashAttention3 MFU

**代价**：  
需要联合训练，无法即插即用。Coarse stage 有额外计算开销（但远小于节省的 attention 计算）。

### 1.4 蒸馏 Recipe（Step-Parallel 去噪）

FastVideo 将 **step-parallel denoising** 与 **distillation** 结合，通过减少 denoising steps 来实现端到端加速。

两种蒸馏策略：
- **DMD2（Distribution Matching Distillation）**：few-step 蒸馏，teacher 多步去噪 → student 少步匹配分布
- **Self-Forcing（因果蒸馏）**：视频生成的 autoregressive distillation，每帧生成条件化于已生成的前帧，减少全局去噪步数

代码位置：`fastvideo/training/wan_distillation_pipeline.py`，`fastvideo/training/self_forcing_distillation_pipeline.py`

蒸馏组合加速：Sparse Distillation（STA/VSA + 步数压缩）声称 **>50x** 端到端 denoising 加速（相比原始 full-step + full-attention baseline）。

### 1.5 与 vLLM/DistServe 方法论延续

| 工作 | bottleneck 识别 | 系统改动 | 算法改动 |
|------|----------------|---------|---------|
| vLLM | KV cache 碎片化 → OOM → 低吞吐 | PagedAttention（block-level KV 管理） | beam search 调度 |
| DistServe | prefill/decode GPU 争用 | prefill-decode disaggregation（新调度拓扑） | goodput 新指标 |
| FastVideo | 3D attention → 85% latency | STA tile-based kernel / VSA sparse kernel | 局部注意力 + 蒸馏 |

共同特征：**profiling 驱动**，先量化 bottleneck，再同时改算法和内核。

---

## 2. 论文 + 代码实现细节

### 2.1 仓库结构

```
fastvideo/
├── attention/
│   ├── backends/           # 所有 attention backend 实现
│   │   ├── abstract.py     # AttentionBackend ABC（直接从 vLLM 适配！）
│   │   ├── flash_attn.py   # FA2/FA3 baseline
│   │   ├── sla.py          # SLA: Sparse-Linear Attention (Triton kernel)
│   │   ├── video_sparse_attn.py  # VSA backend
│   │   ├── vmoba.py        # VMoBA backend
│   │   ├── bsa_attn.py     # BSA: Bidirectional Sparse Attention
│   │   └── sage_attn.py    # SageAttention backend
│   ├── layer.py            # DistributedAttention / DistributedAttention_VSA
│   └── selector.py         # 环境变量驱动的 backend 选择（直接从 vLLM 适配！）
├── training/
│   ├── wan_distillation_pipeline.py         # DMD2 蒸馏
│   ├── wan_self_forcing_distillation_pipeline.py  # Self-Forcing
│   ├── distillation_pipeline.py             # 基类
│   └── ...
├── pipelines/              # 推理 pipeline 抽象
├── distributed/            # Sequence Parallelism (SP)
└── models/
```

### 2.2 Attention Backend 抽象（vLLM DNA）

`fastvideo/attention/selector.py` 的注释第一行：

```python
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/attention/selector.py
```

`fastvideo/attention/backends/abstract.py` 同样注明来自 vLLM。这不是偶然——FastVideo 直接移植了 vLLM 的 attention backend 插件化设计，通过 `FASTVIDEO_ATTENTION_BACKEND` 环境变量切换 kernel 实现，而不需要改模型代码。

关键 metadata 设计：

```python
@dataclass
class AttentionMetadata:
    current_timestep: int  # diffusion step，用于 step-conditional sparsity
```

这个 `current_timestep` 字段是视频 DiT 独有的，vLLM 中没有——diffusion 的 sparsity pattern 可以随 timestep 变化（early steps 更稀疏，later steps 更密集）。

### 2.3 VSA 实现关键细节

```python
# video_sparse_attn.py
VSA_TILE_SIZE = (4, 4, 4)  # time × height × width tiles

def get_tile_partition_indices(dit_seq_shape, tile_size, device):
    """将 3D 时空 token 重排为 tile 优先顺序"""
    T, H, W = dit_seq_shape
    # 枚举所有 (t_tile, h_tile, w_tile) 组合
    # 使得同一 tile 内的 token 在内存中连续
    ...
```

VSA 的核心工程创新：**token reordering**。通过 `get_tile_partition_indices` 将时空 token 从 (T, H, W) 线性顺序重排为 tile-first 顺序，使 GPU block-level attention 计算边界与 tile 边界对齐，避免 padding 浪费。`fastvideo_kernel.video_sparse_attn` 是独立的 CUDA/Triton 扩展包。

### 2.4 蒸馏流水线设计

蒸馏采用 teacher-student 框架，一份 transformer 权重同时承担三个角色（student / teacher-ema / critic），通过 `deepcopy` 分叉。`WanDistillationPipeline` 继承 `DistillationPipeline`，在 `initialize_validation_pipeline` 中实例化带 `dit_cpu_offload=True` 的验证管道，避免显存不够。

```python
# wan_distillation_pipeline.py
validation_pipeline = WanDMDPipeline.from_pretrained(
    training_args.model_path,
    loaded_modules={"transformer": self.get_module("transformer")},  # 共享权重
    dit_cpu_offload=True,  # 训练时评估节省显存
    ...
)
```

---

## 3. Hao Zhang 的角色

Hao Zhang 在两篇论文中均为最后一位作者（通讯/指导），一作均为 Peiyuan Zhang（UCSD PhD student）。FastVideo 是 Hao 在 UCSD 的 Hao AI Lab 的旗舰系统项目。

从技术 DNA 角度看，FastVideo 是 vLLM 方法论在视频生成域的直接延伸：
- vLLM → 解决 LLM serving 的 KV cache 问题
- DistServe → 解决 LLM serving 的 prefill-decode 争用问题  
- FastVideo → 解决 Video DiT serving 的 attention 成本问题

Hao 在 CMU 期间主导的 Alpa（自动并行化）背景，使 FastVideo 在分布式设计（Sequence Parallelism、all-to-all 通信）上相当扎实，这是普通 diffusion 加速工作做不到的系统深度。

---

## 4. Co-Design 方法论提炼

### 4.1 Profile 先行，永远

FastVideo 最重要的句子在 STA 论文引言：
> "attention alone takes 800 out of 945 seconds"

这个数字的出现本身代表了一种工作哲学：**在提出任何方案之前，先把 profiling 数字写在纸上**。这与 vLLM 的 "KV cache memory waste > 60%"、DistServe 的 "prefill/decode resource contention" 是同一套范式。

### 4.2 Bottleneck 必须是结构性的，不是偶然的

STA 的核心 insight 不只是"attention 慢"，而是"attention score 的局部性在预训练模型上已经存在"——这是一个可以通过 profiling attention score distribution 来验证的结构性观察。类似地，DistServe 的 insight 是 prefill/decode 的资源需求从根本上不同，不是调参能解决的。

Hao 的选题标准可以总结为：**"这个 bottleneck 是否有 structural reason 导致它只能通过系统-算法 co-design 来解决？"**

### 4.3 Algorithm + System 必须同时改

| 单独改算法 | 单独改系统 | Co-Design |
|-----------|-----------|-----------|
| 局部 attention（SWA）早已有人提，但 token-wise 实现 GPU 效率极差 | 单独优化 kernel 无法减少 FLOPs | STA：tile-wise sliding window + hardware-aware kernel，两者缺一不可 |

STA 的 insight：SWA 的算法正确性早已知晓（局部 attention），但没有人做到高效实现，因为 token-wise 移动窗口在 GPU 上极度不连续。STA 的贡献是找到了让算法与 GPU 内存层次结构对齐的 tile abstraction。

### 4.4 借用已有抽象框架

FastVideo 的 `attention/selector.py` 和 `attention/backends/abstract.py` 直接来自 vLLM。这不是懒惰——这是复用经过验证的插件化抽象，让注意力 backend 可以像更换零件一样替换。同样，`profiler.py` 完全委托 PyTorch profiler，不重新发明轮子。

**模式：复用基础设施（vLLM/PyTorch），创新在最关键的那一个 primitive 上（tile attention / sparse pattern）。**

### 4.5 从 vLLM → DistServe → FastVideo 保留的套路

1. **新指标驱动**：vLLM 的 "goodput"，DistServe 的 "attained goodput under SLO"，FastVideo 的 "MFU" + "VBench@latency"。新指标本身就是 co-design 的 claim。
2. **Disaggregation**：vLLM disaggregates memory，DistServe disaggregates prefill/decode，FastVideo disaggregates full-attention → tile-attention（空间粒度的 disaggregation）。
3. **零运行时开销的配置接口**：环境变量驱动（`FASTVIDEO_ATTENTION_BACKEND`），无需修改模型代码。
4. **实证为先**：每个 claim 都配有端到端 benchmark，speedup 数字明确标注 baseline 是什么。

---

## 5. 到 VLM/VLA 的迁移启示

### 5.1 STA/VSA → Flow VLA Action Denoising

我们的实验数据：
- **exp04a (Fast-WAM)**：@10step: action 占 89%（362ms），per-step ~32ms，30 层 MoT cross-attn
- **exp06a (NitroGen 500M DiT)**：per-step 7.2ms（174M DiT），线性 scaling，k=1: 55.9Hz

**FastVideo-style 加速机会分析**：

| 场景 | 当前 bottleneck | STA 适用性 | 预期收益 |
|------|----------------|-----------|---------|
| Flow VLA action head（NitroGen 174M DiT, k=1） | 7.2ms/step，已经很快 | 低（序列短，attention 不是主要 overhead） | 边际 |
| Fast-WAM @10step（MoT cross-attn） | 362ms action，29ms/layer | **高**：30 层 cross-attn，若序列有局部性可 2-3x | 30-50% 端到端 |
| LingBot-VA full WAM（2091ms） | Action 68%（1423ms），可能有 video DiT 级别 attention） | 高（与 Wan2.1 结构最相似） | 3-5x |

**关键前提**：STA 的适用性取决于 action DiT 的 attention score 是否也有局部性。这需要对 action denoising step 做 attention analysis（类似 exp05a 做 VLA attention 的方式）——**这就是一个 FastVideo-style 的实验机会**。

### 5.2 Step-Parallel 蒸馏 → Diffusion Policy 加速

VLA 的 diffusion policy（如 LingBot-VLA flow head、NitroGen DiT）使用多步去噪。FastVideo 的 self-forcing distillation（步数 50 → 4）直接可以作为方法论参考：
- **Student**：少步 VLA action head
- **Teacher**：原始多步 VLA action head
- 目标：保持 action quality（success rate），压缩 steps

**数字对比**：NitroGen k=1（单步）已经 55.9Hz，若 k=4（当前默认）是 ~14Hz。若 steps 从 4 → 1 不损质量，则直接达到实时控制频率（50-100Hz）。

### 5.3 蒸馏 Recipe → VLA 的独特挑战

视频生成的蒸馏比 VLA 简单，因为 VLA 的 reward 是物理任务成功率，而不是视觉质量。但方法论层面：
- Sparse Distillation = sparsity（STA/VSA）+ step 压缩，两者独立可组合
- VLA 可以先做 step 蒸馏，不需要同时引入 sparse attention

### 5.4 对当前研究阶段的诊断

**当前实验偏 trivial 的根因**：profiling 实验（exp01-06）记录了"是什么"（latency breakdown），但还没有问"能不能"（能否用已知 technique 改变这个 breakdown）。

FastVideo 的核心贡献不是 profiling，而是用 profiling 发现的 structural property 来设计新算法：
- 发现：attention score 有局部性 → 设计：STA/VSA
- 发现：多步去噪冗余 → 设计：蒸馏

**对应的 VLA 升级方向**：
1. 测量 Flow VLA action DiT 的 attention score 分布（局部性？稀疏性？）
2. 若有局部性 → 设计 VLA-adapted STA（action token 的时序局部性与视频空间局部性类比）
3. 测量 step-by-step 的 noise residual 变化（是否大部分 steps 贡献微小？）
4. 若是 → few-step distillation for VLA action head

**最有可能有回报的实验**：对 exp04a（Fast-WAM MoT cross-attn）的 attention score 做 tile-wise 局部性分析，这是 FastVideo 方法能否迁移的关键验证。

---

## 附录：术语速查

| 术语 | 含义 |
|------|------|
| STA | Sliding Tile Attention，tile 粒度 3D 局部 attention |
| VSA | Video Sparse Attention，动态稀疏 attention，可训练 |
| SLA | Sparse-Linear Attention（FastVideo backends 中还有 SLA，来自 TurboDiffusion） |
| MFU | Model FLOPS Utilization，GPU 实际 FLOPS / 峰值 FLOPS |
| DMD2 | Distribution Matching Distillation v2，few-step 蒸馏 |
| Self-Forcing | Causal distillation，autoregressive 视频生成蒸馏 |
| SP | Sequence Parallelism，跨 GPU 的序列维度并行 |
| DiT | Diffusion Transformer |
| MoT | Mixture-of-Transformers（Fast-WAM 的 cross-attn 模块） |

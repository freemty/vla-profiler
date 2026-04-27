# Learning Plan — GPU Systems for exp08

> 为做 exp08 决策所需的最小补课路径。不追求体系，追求"解锁决策"。
> Created: 2026-04-27. Owner: @freemty.

## 背景

exp08 横跨 GPU 执行模型 (L1) / LLM serving 概念 (L2) / GPU contention 机制
(L3) / co-location 策略 (L4) 四层。目前在 L1–L4 都有缺口，导致对实验
motivation 和预期结果解读感到 lost。本计划按"先决策、后学习"原则排布。

## 分层诊断

| 层 | 典型问题 | 当前状态 |
|----|---------|---------|
| L0 CUDA 基础 | kernel / stream / SM 是什么 | 模糊 |
| L1 GPU 执行模型 | 一次 forward 变成多少 kernel、如何排队 | 缺口 |
| L2 Serving 概念 | prefill/decode, KV-cache, continuous batching | VLA 侧懂，LLM 侧半懂 |
| L3 Contention 机制 | SM scheduler / mem controller / tensor core 如何 share | 全缺 |
| L4 Co-location 策略 | CUDA MPS / MIG / stream priority / TP | 全缺 |

## 决策分叉 (决定学到哪一层)

| exp08 下一步选项 | 必补深度 | 时间预算 |
|------------------|---------|---------|
| (C) pair-level 发现收工 | **L0+L1** | 半天 |
| (A) triple/quad 外推验证 M4 | L0+L1 + 一点 L3 | 1–2 天 |
| (D) 机制判别实验 (MPS / SM quota) | L0–L4 全要 | 1–2 周 |

**策略**：先走 (C) 的半天预算，做完再判断是否升级。

## 半天学习计划

### 上午 2h — L0+L1 GPU 执行模型

- [ ] Horace He, "Making Deep Learning Go Brrrr From First Principles"
      https://horace.io/brrr_intro.html
      核心: compute-bound vs memory-bound vs overhead-bound
- [ ] CUDA MODE Lecture 1 (YouTube) 前 30 分钟
      演示: kernel launch / stream / grid

**完成判据 (能回答)**:
- decode 为什么 memory-bound?
- "kernel dispatch contention" 具体指什么?

### 下午 2h — L2 LLM serving 概念 (与项目直接相关)

- [ ] vLLM 官方 blog "Efficient Memory Management for LLM Serving with PagedAttention"
      核心: KV-cache 为何占大内存, 为什么 prefill/decode 要分开
- [ ] DistServe 论文前 3 节 (你组自己的工作, exp08 的直接前身)
      核心: PD disaggregation, goodput 指标

**完成判据 (能回答)**:
- 为什么 P 和 D 不能共卡?
- exp08 在 DistServe 2-phase 基础上加了什么?

## 预备动作 (无外部依赖，可立即做)

- [ ] Socratic 模式: 对着 `scripts/exp08a_interference.py` 逐行提问, 把"让你 lost 的术语" 用 exp08a 的实测数字解释清楚 (不引入新术语)
- [ ] 写 `docs/knowhow/exp08-mental-model.md` (一页), 用 exp08 的数字锚点解释 L0–L2 关键概念, 作为外部材料阅读的"导入"

## 学完后的行动

- [ ] 重新评估 exp08 下一步: (C) 收工 / (A) 外推 / (D) 机制实验
- [ ] 如果升级到 (D), 补 L3+L4 材料 (CUDA MPS docs, NVIDIA Grace Hopper SM scheduler whitepaper)

## 参考材料汇总

| 资源 | 层 | 状态 |
|------|----|------|
| Horace He "Brrrr" blog | L0+L1 | pending |
| CUDA MODE Lecture 1 | L0+L1 | pending |
| vLLM PagedAttention blog | L2 | pending |
| DistServe paper (组内) | L2 | pending |
| CUDA MPS docs (NVIDIA) | L3+L4 | 仅在走 (D) 时需要 |
| NVIDIA H100 whitepaper (SM scheduler 章节) | L3 | 仅在走 (D) 时需要 |

# TODO

> Completed items moved to CHANGELOG (audit trail preserved there).
> Last refresh: 2026-04-27 — closed exp08b pairs + exp08c v1.

## P0 — 学习补课 (unblocks 所有 exp08 决策)

- [ ] **P0** 执行 `docs/learning-plan.md` 半天计划 — 上午 Horace He Brrrr + CUDA MODE L1 (L0+L1 解锁); 下午 vLLM PagedAttention + DistServe (L2 解锁)。完成判据: 能回答 "decode 为何 memory-bound" / "exp08 在 DistServe 基础上加了什么"。学完再决定 exp08 走 (C)/(A)/(D) 哪条路 — 2026-04-27 (new)

## P0 — 见 Hao 前必做 (学习后启动)

- [ ] **P0** 与 Hao 面谈准备 — 用 `slides/epda-roofline-motivation.html` 开场。**数据到齐版本**: exp08a 3.15×/3.52× pilot → exp08b 完整 6-pair 矩阵 (D/P 脆弱 2.4–2.9×, E/A 鲁棒 <1.3×) → exp08c M4 asymmetric model R²=0.94 (v=(D:1.52, P:1.61, E:0.23, A:0.20))。请教方向: 候选 C (DiT caching) vs D' (mechanism study + VLA SLO)，承认 vLLM-Omni 占据 framework 空间 — 2026-04-27

## P1 — exp08 主线 (blocked by P0 学习)

> 学完 learning-plan 后再决定是否启动, 以及走哪条路 (C/A/D)。

- [ ] **P1** exp08b triple/quad combos (EPD/EPA/EDA/PDA/EPDA) — 脚本已就绪 (`launch_exp08b.sh 0 --multi-only`)，~3-4h server time。**目的: 验证 M4 从 pair 外推到 N>2 的能力**。仅在学完 P0 后、决定走 (A) 或 (D) 时启动 — 2026-04-27 (new)
- [ ] **P1** M4 → M5 升级 — exp08c EP(E) 残差 -0.46 为最大误差，提示 tensor-core specific interaction 项缺失。若 triple R²<0.9 即启动 M5 (加 compute-channel-specific 二阶项) — 2026-04-27 (new)
- [ ] **P1** exp08 mechanism-study 论文骨架 — 若 M4/M5 triple 外推 R²>0.9, 按 "EPDA interference 非对称 + 可预测 contention model" 动笔 outline，目标 workshop / short paper — 2026-04-27 (new, 后置)
- [ ] **P1** 写 `docs/knowhow/exp08-mental-model.md` — 用 exp08 实测数字解释 L0–L2 关键概念 (kernel dispatch / SM scheduler / memory-bound vs compute-bound), 作为自己和 future Claude 的 anchor 参考 — 2026-04-27 (new)

## P2 — 候选方向细化 / 补充研究

- [ ] **P2** 候选 C 细化 spec (DiT caching for VLA) — FastVideo-style step caching 迁移到 Fast-WAM / DreamZero / NitroGen Action DiT; 测量 **DiT layer activation variance** (每层变化率 → 最优 cache 策略); 设计实时约束下的 step-aware cache scheduling — 2026-04-27 (合并了原 DreamZero DiT layer variance 项)
- [ ] **P2** DreamZero baseline profiling on RTX 5880 Ada — 候选 C 的 WAM baseline 之一。需 Wan2.1-I2V-14B (~45GB) + DreamZero-DROID (~28GB) 下载。**仅在候选 C 确定为主线时启动** — 2026-04-21 (降档 2026-04-27)
- [ ] **P2** vLLM-Omni / SGLang Diffusion 接触 — 阅读 vLLM-Omni paper 细节, 尝试在仓库 issue 或 slack (#diffusion) 联系维护者讨论 VLA/robotics SLO 方向是否有合作空间 — 2026-04-27

## P2 — 工程清洁 (可绕开)

- [ ] **P2** phase 命名标准化 — exp04b JSON key `video_denoise`/`action_denoise` 统一到 `video`/`action`, 或在 exp08 代码加映射层 — 2026-04-26
- [ ] **P2** wall-clock vs phase-sum gap tracking — timing_validation task 补充这个差值指标, 为 exp08 干扰量化提供 baseline noise floor — 2026-04-26

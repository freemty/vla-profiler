# TODO

> Completed items have been moved to CHANGELOG (audit trail preserved there).

## P0 — 见 Hao 前必做

- [ ] **P0** 与 Hao 面谈准备 — 用 `slides/epda-roofline-motivation.html` 开场, 展示 exp08a pilot 3.15×/3.52× inflation, 请教 C (DiT caching) vs D' (mechanism study + VLA SLO) 方向选择, 承认 vLLM-Omni 占据 framework 空间 — 2026-04-27

## P1 — exp08 降档后的主线

- [ ] **P1** exp08b 完整 6×6 EPDA 干扰矩阵 — 扩展 exp08a pilot (2 pair → 12 pair, 含 E+P/E+D/E+A/P+D 等), 配合 stream-aware PhaseTimer + nvtx markers。预计 server 跑 4-6 小时。产出: 完整 inflation heatmap + CV 矩阵 — 2026-04-27
- [ ] **P1** exp08c GPU kernel-level contention model — 拟合 exp08a/b 数据, 建立 "给定两 phase 的 kernel shape/size 预测 coloc inflation" 的解析模型。候选抽象: kernel launch queue / SM scheduler / HBM controller queue。补 roofline gap — 2026-04-27

## P2 — 候选方向细化 / 补充研究

- [ ] **P2** 候选 C 细化 spec (DiT caching for VLA) — FastVideo-style step caching 迁移到 Fast-WAM / DreamZero / NitroGen Action DiT; 测量 **DiT layer activation variance** (每层变化率 → 最优 cache 策略); 设计实时约束下的 step-aware cache scheduling — 2026-04-27 (合并了原 "DreamZero DiT layer variance" 项)
- [ ] **P2** DreamZero baseline profiling on RTX 5880 Ada — 作为候选 C 的 WAM baseline 之一。需 Wan2.1-I2V-14B (~45GB) + DreamZero-DROID (~28GB) 下载，降档: 仅在候选 C 确定是主线时启动 — 2026-04-21 (降档 2026-04-27)
- [ ] **P2** vLLM-Omni / SGLang Diffusion 接触 — 阅读 vLLM-Omni paper 细节, 尝试在仓库 issue 或 slack (#diffusion) 联系维护者讨论 VLA/robotics SLO 方向是否有合作空间 — 2026-04-27

## P2 — 工程清洁 (可绕开)

- [ ] **P2** phase 命名标准化 — exp04b JSON key `video_denoise`/`action_denoise` 统一到 `video`/`action`, 或在 exp08 代码加映射层 — 2026-04-26
- [ ] **P2** wall-clock vs phase-sum gap tracking — timing_validation task 补充这个差值指标, 为 exp08 干扰量化提供 baseline noise floor — 2026-04-26

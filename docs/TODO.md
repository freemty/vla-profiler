# TODO

- [x] ~~NitroGen 500M DiT profiling~~ — 2026-04-21 (done 2026-04-22)
- [ ] DreamZero profiling on RTX 5880 Ada (exp05?) — 验证 DiT caching 在 memory-bandwidth bound regime 下的真实收益; 对比 exp04a/04b 的 per-step baseline (~28-32ms); 需要先下载 Wan2.1-I2V-14B + DreamZero-DROID checkpoint (~45GB+28GB) — 2026-04-21
- [ ] DreamZero DiT layer activation variance 分析 — 测量哪些层可以 cache, 每层 activation 变化率, 最优 cache 策略 — 2026-04-21
- [ ] **P0** exp08 roofline 分析 — 用已有 profile 数据画 E/P/D/A 四阶段在 RTX 5880 Ada 上的 compute-intensity (FLOPS/Byte) 坐标, 判断是否落在不同象限 (0 GPU 成本, 半天). 决定 EPDA 方向 go/no-go — 2026-04-26
- [ ] **P0** exp08 stream-aware PhaseTimer 设计 — 扩展 `_CudaTimerBackend` 接受 stream 参数, 支持多 stream 并发测量. exp08a 干扰矩阵的必需基础设施 (2-3 天) — 2026-04-26
- [ ] **P1** exp04b LingBot-VA 重跑 — 当前 warmup=3, CV=21%, 无分位数. 改为 warmup=15 + `nvidia-smi -pm 1`, 补 raw iteration data + P10/P90 — 2026-04-26
- [ ] **P1** exp04a/04b 统计口径统一 — standalone scripts 补 `all_ms` + median + P10/P90/P99, 对齐 Hydra 路径的 PhaseTimer 格式 — 2026-04-26
- [ ] **P2** phase 命名标准化 — exp04b JSON key `video_denoise`/`action_denoise` 统一到 `video`/`action`, 或在 exp08 代码加映射层 — 2026-04-26
- [ ] **P2** wall-clock vs phase-sum gap tracking — timing_validation task 补充这个差值指标, 为 exp08 干扰量化提供 baseline noise floor — 2026-04-26

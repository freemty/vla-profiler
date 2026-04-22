# TODO

- [ ] NitroGen 500M DiT profiling — 2026-04-21
- [ ] DreamZero profiling on RTX 5880 Ada (exp05?) — 验证 DiT caching 在 memory-bandwidth bound regime 下的真实收益; 对比 exp04a/04b 的 per-step baseline (~28-32ms); 需要先下载 Wan2.1-I2V-14B + DreamZero-DROID checkpoint (~45GB+28GB) — 2026-04-21
- [ ] DreamZero DiT layer activation variance 分析 — 测量哪些层可以 cache, 每层 activation 变化率, 最优 cache 策略 — 2026-04-21

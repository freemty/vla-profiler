# TODO

- [x] ~~NitroGen 500M DiT profiling~~ — 2026-04-21 (done 2026-04-22)
- [ ] DreamZero profiling on RTX 5880 Ada (exp05?) — 验证 DiT caching 在 memory-bandwidth bound regime 下的真实收益; 对比 exp04a/04b 的 per-step baseline (~28-32ms); 需要先下载 Wan2.1-I2V-14B + DreamZero-DROID checkpoint (~45GB+28GB) — 2026-04-21
- [ ] DreamZero DiT layer activation variance 分析 — 测量哪些层可以 cache, 每层 activation 变化率, 最优 cache 策略 — 2026-04-21
- [x] ~~**P0** exp08 roofline 分析~~ — 2026-04-26 done (`docs/specs/2026-04-26-epda-roofline-analysis.md`). **结论: GO** — E/P/D/A 跨越 3 类 bottleneck (compute / BW-saturated / BW-moderate / latency-bound), A 阶段 latency-bound 是 LLM 域未研究的新 class. D+A 预测最强干扰, P+A 可互补
- [x] ~~**P0** exp08 stream-aware PhaseTimer 设计~~ — 2026-04-27 done. `_CudaTimerBackend.record_start/end(stream=)` + `PhaseTimer.mark_start/end(stream=)`, backward-compatible. CPU backend accepts + ignores stream kwarg. Tests: 14/14 pass
- [x] ~~**P1** exp04b LingBot-VA 重跑~~ — 2026-04-26 (done 2026-04-27, `results_lingbot_va_rerun_warmup15.json`): **canonical** E=84.7/V=697/A=1708ms, total 2518ms (0.40Hz). Legacy warmup=3 系统性低估 ~18%. Encode CV 保持 20% → VAE 固有方差
- [x] ~~**P1** exp04a/04b 统计口径统一~~ — 2026-04-26 (done 2026-04-26-27, `scripts/_profiling_stats.py` + profile_fastwam.py + profile_lingbot_va.py): all_ms + median + p10/p90/p99/cv 已对齐 Hydra PhaseTimer
- [ ] **P2** phase 命名标准化 — exp04b JSON key `video_denoise`/`action_denoise` 统一到 `video`/`action`, 或在 exp08 代码加映射层 — 2026-04-26
- [ ] **P2** wall-clock vs phase-sum gap tracking — timing_validation task 补充这个差值指标, 为 exp08 干扰量化提供 baseline noise floor — 2026-04-26

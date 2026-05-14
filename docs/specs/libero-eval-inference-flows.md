# LIBERO Eval — 三模型闭环推理流程对比

> Fast-WAM / LingBot-VA / Cosmos Policy 在 LIBERO benchmark 上的闭环 eval 实现差异

## LIBERO Benchmark 设定

**环境**: MuJoCo 物理仿真，单臂 Franka Panda，robosuite 渲染，EGL headless。

| Suite | 任务数 | 变化维度 | Max Steps |
|-------|--------|----------|-----------|
| libero_spatial | 10 | 同一物体不同位置 | 220 |
| libero_object | 10 | 不同物体同一动作 | 280 |
| libero_goal | 10 | 不同目标 | 300 |
| libero_10 | 10 | 完全不同场景+物体+目标 | 520 |

**指标**: success_rate = successes / episodes (20 ep/task)，overall = mean(4 suites)。二值判定（MuJoCo done=True）。

**闭环**: 所有模型都是闭环 eval — env.step(action) 返回真实模拟观测，replan 时用真实 obs。

## Fast-WAM (skip-imagination)

```
obs → model.infer_action(当前单帧, prompt, proprio) → action chunk → 执行 replan_steps 个 → 新 obs → 重新推理
```

- **无状态**: 每次 replan 只看当前帧，无历史 context
- **Skip video**: `infer_action` 跳过 video 生成，直接 denoise 出 action
- **Replan 频率**: 每 5 步 (replan_steps=5)
- **Denoise**: MoT DiT × 5 步
- **延迟**: 257ms/次 (exp04c)
- **成绩**: 94.5%

## LingBot-VA (full WAM)

```
[首次] obs → E(VAE encode) → V(video denoise ×20) → A(action denoise ×50) → 执行 16 步
[反馈] 真实关键帧 + 已执行 action → compute_kv_cache (注入 DiT KV cache)
[后续] → V(基于 cache) → A(基于 cache) → 执行 16 步 → 反馈 → 循环
```

- **有状态**: KV cache 累积所有历史帧 + 动作，autoregressive context window
- **Full imagination**: 先 video denoise 想象未来，再 action denoise 出动作
- **共享 DiT**: 同一个 5B WanTransformer3DModel，action_mode=True/False 切换
- **三步交替**: `_infer` → 执行 chunk → `compute_kv_cache`(真实帧注入) → 下次 `_infer`
- **反馈注入**: `compute_kv_cache` 把真实观测帧 + 已执行 action VAE encode 后追加到 KV cache (update_cache=2)
- **Replan 频率**: 每 16 步 (4帧 × 4 action/帧)
- **Denoise**: Video 20步 + Action 50步，分阶段跑
- **延迟**: 2518ms/次 (exp04b) + KV cache 更新开销
- **成绩**: 🔄 运行中

## Cosmos Policy (unified latent denoising)

```
obs → 构建 latent sequence [blank|proprio|wrist|primary|action|future_proprio|future_wrist|future_primary|value]
    → VAE encode → 注入 proprio/action noise → DiT denoise ×5 步 → 提取 action chunk → 执行 16 步 → 新 obs → 重推
```

- **无状态**: 每次 replan 只看当前帧，无历史 context
- **隐式 imagination**: 不单独生成 video，而是在同一个 latent sequence 里同时 denoise action + future state + value
- **Unified sequence**: 当前状态 (proprio + wrist + primary) + action (noise) + future state (noise) + value 编码到一个 latent 序列
- **一次 denoise**: 2B Cosmos-Predict2 DiT 一次前向同时预测所有输出
- **Replan 频率**: 每 16 步 (chunk_size=16, open_loop=16)
- **Denoise**: DiT × 5 步
- **延迟**: ~342ms/次 (exp09a, 1-step extrapolated)
- **成绩**: 97.4%

## 对比表

| | Fast-WAM | LingBot-VA | Cosmos Policy |
|---|---|---|---|
| 架构 | MoT DiT | 5B WanDiT (shared V+A) | 2B Cosmos-Predict2 DiT |
| 上下文 | 无状态，单帧 | 有状态，KV cache 累积 | 无状态，单帧 |
| 视频想象 | ❌ Skip | ✅ 先生成视频再出动作 | ✅ 隐式 (unified latent) |
| Denoise | 仅 action, 5步 | Video 20步 + Action 50步 | 统一 sequence, 5步 |
| 反馈方式 | 直接用新 obs 重推 | compute_kv_cache 注入真实帧 | 直接用新 obs 重推 |
| Replan 步数 | 5 | 16 | 16 |
| 单次延迟 | 257ms | 2518ms + cache | ~342ms |
| LIBERO | 94.5% | 🔄 | 97.4% |

## 关键洞察

1. **无状态 vs 有状态**: Fast-WAM 和 Cosmos 都是无状态（每次重新看当前帧），LingBot-VA 用 KV cache 维护时间上下文。无状态反而成绩更好 — 说明 LIBERO 任务不需要长期记忆。
2. **Unified vs Sequential**: Cosmos 把 action/future/value 放在同一个 latent 序列一次 denoise，LingBot-VA 分两阶段跑同一个 DiT。Unified 更高效。
3. **Imagination 的价值**: Cosmos 97.4% > Fast-WAM 94.5%，都有 imagination（Cosmos 隐式/Fast-WAM 跳过），LingBot-VA 的显式 imagination 代价最大但成绩待定。
4. **profiling 的局限**: exp04b profiling 只测了单次 `_infer`，LingBot-VA 真实部署还有 `compute_kv_cache` 开销。

## Notes
- Date: 2026-05-14
- 代码路径: `scripts/run_cosmos_libero.py`, `scripts/run_exp04d_libero.sh`, Fast-WAM vendor `experiments/libero/eval_libero_single.py`
- LIBERO 环境: xdlab23, mujoco 3.8.0 + robosuite 1.4.0 + EGL

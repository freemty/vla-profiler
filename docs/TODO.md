# TODO

> Completed items moved to CHANGELOG (audit trail preserved there).
> Last refresh: 2026-05-13 — LIBERO eval 开始调试，发现 lingbotvla 依赖不兼容。

## 战略判断

VLA 推理现在卡在**单请求太慢** (Pi-Zero 200ms=5Hz, 需要 10-50Hz)，不是并发不够。
这是 FastVideo 阶段 (单次加速)，不是 vLLM 阶段 (多用户 serving)。
→ **候选 A (VLA 推理加速) 是最高优先级方向。**
→ exp08 (contention/serving) 降档为 side project / 备用论文素材。

## P0 — 学习补课 (unblocks 方向决策)

- [ ] **P0** 执行 `docs/learning-plan.md` 半天计划 — 上午 Horace He Brrrr + CUDA MODE L1; 下午 vLLM PagedAttention + DistServe 前 3 节。完成判据: 能回答 "decode 为何 memory-bound" / "PD disaggregation 为什么有效"。**学完后决定 Hao meeting 前是否还需要补更多** — 2026-04-27

## P0 — 见 Hao 前必做

- [x] **P0** 画 7-model Pareto 图 (一页, 有 Hz 刻度) — `viewer/static/design-space.html` Section 01 散点图 ✅
- [x] **P0** 画 DiT scaling curve 图 (174M / 300M / 350M per-step, 含 cross-attn 标注) — `viewer/static/design-space.html` Section 03 ✅
- [x] **P0** 读 FastVideo / DistServe — 已有综合 survey (`survey/papers/hao-style-fastvideo.md`, `hao-style-distserve.md`)，蒸馏为口头速查 `docs/meeting-cheatsheet.md` ✅
- [x] **P0** 新 slide deck 按四幕叙事重做 — `slides/hao-meeting-2026-04-28.html` (10 页, Title/Opening/4 Jumps/Jump 3 deep dive/Spectrum bars/VLA attention/Priorities/Questions/exp08 backup/Closing) ✅
- [x] **P0** 准备 exp08 一页总结 (备用) — `docs/meeting-cheatsheet.md` 末尾口头版 ✅
- [x] **P0** 详见 `docs/hao-meeting-prep.md` — 四幕叙事结构已完成 ✅

## P0 — LIBERO Eval (补 quality 数据)

> **Blocker**: lingbotvla 包和 vit-probe env 不兼容 (transformers 4.57 缺 `LossKwargs`, lerobot 0.5.1 路径 `lerobot.common.policies` → `lerobot.policies`)。已 patch lerobot import，但 transformers `LossKwargs` 无法 patch。

- [ ] **P0** exp03b: LingBot-VLA LIBERO-4 eval
  - **脚本**: `scripts/run_exp03b_libero.py` 已上传到 xdlab23
  - **Blocker**: `lingbotvla.models.vla.pi0.modeling_lingbot_vla` import `LossKwargs` from transformers → 4.57 没有
  - **Fix 方案**: 重写 `load_policy()`，不 import lingbotvla 包，改用我们 `LingBotVLAController` 的加载路径 (PI0Config + safetensors 直接加载)。Controller import 已确认 OK。
  - **已完成的 patch**: xdlab23 上 lingbotvla 的 `lerobot.common.policies` → `lerobot.policies` (8 处, sed 批量修)
  - **下一步**: 重写 load_policy 绕过 lingbotvla → smoke test → full 20 ep × 4 suites
- [ ] **P0** exp04d: LingBot-VA LIBERO eval
  - server-client 模式 (`run_libero_all.sh` 里有)
  - **Blocker**: lingbot-va 也可能有同样的 transformers/lerobot 兼容问题；之前 cuDNN crash 过
  - **下一步**: 先等 exp03b 跑通再看
- [ ] **P0** exp07c: Pi-Zero LIBERO-4 eval
  - openpi server + client (`run_libero_all.sh` 里有)
  - **Blocker**: flax/JAX 依赖，uv env 需要 GitHub (被墙)
  - **下一步**: 检查是否可以用我们的 PiZeroController + LIBERO env loop 绕过 openpi
- [ ] **P0** Cosmos Policy LIBERO eval
  - 需要新写 eval 脚本
  - **下一步**: 参考 exp09a 的 `get_action()` 接口 + LIBERO env loop

## P0 — 实验补强 (补 depth)

- [ ] **P0** Memory profiling — 每个 exp 加 `torch.cuda.max_memory_allocated()`, 补全 VRAM 数据 (目前只有 Cosmos 8816MB)
- [ ] **P0** Batch size sweep — batch 1/2/4/8, 看 throughput scaling (目前全是 batch=1)
- [ ] **P0** Kernel-level trace — 拿 Pi-Zero 跑一次 `torch.profiler`, 展示 action phase 里哪些 kernel 最重

## P1 — 候选方向 (Hao meeting 后启动)

> 候选排序: **A > B > C >> D (too early) > E (side project)**

- [ ] **P1** 候选 A: VLA 单次推理加速 — FastVideo 思路迁移到 VLA Action DiT (STA/蒸馏/step caching)。**直接 bottleneck**: Action 占 80-94% 延迟, DiT 越大越贵呈超线性增长。Hao meeting 后细化 spec
- [ ] **P1** 候选 B: VLA inference benchmark — 统一 profiling 框架 + SLO benchmark (填 wild-west 空白)。配套 A，低风险。可复用 exp01-07 的 framework
- [ ] **P1** 候选 C 细化 spec (DiT caching for VLA) — FastVideo-style step caching 迁移; 测量 DiT layer activation variance; 设计实时约束下的 step-aware cache scheduling。A 的子方向

## P2 — exp08 收尾 (降档, 可选)

> exp08 已完成 6-pair 干扰矩阵 + M4 模型 (R²=0.94)。Triple/quad 验证和论文动笔**暂停**。

- [ ] **P2** exp08b triple/quad combos — 脚本已就绪 (`launch_exp08b.sh 0 --multi-only`)。仅在 Hao 认为 mechanism study 值得做时启动
- [ ] **P2** M4 → M5 升级 — EP(E) 残差 -0.46 提示 tensor-core interaction 缺失。仅在 triple R²<0.9 时启动
- [ ] **P2** exp08 mechanism-study 论文骨架 — workshop / short paper。仅在 M4 triple 外推 R²>0.9 且 Hao 支持时动笔

## P2 — 补充研究 (可选)

- [ ] **P2** DreamZero baseline profiling on RTX 5880 Ada — 候选 A/C 的 WAM baseline。需 Wan2.1-I2V-14B + DreamZero-DROID 下载
- [ ] **P2** vLLM-Omni / SGLang Diffusion 接触 — 了解他们在 multimodal serving 的进展, 确认不重复

## P2 — 工程清洁 (可绕开)

- [ ] **P2** phase 命名标准化 — exp04b `video_denoise`/`action_denoise` → `video`/`action`
- [ ] **P2** wall-clock vs phase-sum gap tracking — timing_validation 补差值指标
- [ ] **P2** 写 `docs/knowhow/exp08-mental-model.md` — 用 exp08 数字解释 L0-L2 关键概念, 作为自己的 anchor 参考

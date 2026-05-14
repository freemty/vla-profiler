# Changelog

## v0.11.0 @freemty — 2026-05-13/14

### 新增 (LIBERO eval pipeline + Cosmos 97.4%)
- **Cosmos Policy LIBERO eval**: **97.4% avg** (spatial 96.0 / object 100 / goal 98.0 / 10 95.5), 800 ep. 论文 98.5%, 差 1.1pp (RTX 5880 vs A100 + seed)
- **exp04d LingBot-VA LIBERO eval**: 4-GPU 并行 (GPU 3/5/6/7), server-client 模式, 20 ep × 4 suites (运行中)
- **lerobot_stub**: 最小 PI0Config + PreTrainedPolicy stub, 避免 lerobot 0.5.1 与 transformers 4.51 版本冲突 (`scripts/lerobot_stub/`)
- **LIBERO eval 脚本**: `run_exp03b_libero.py` / `run_exp07c_libero.py` / `run_exp04d_libero.sh` / `run_exp04d_parallel.sh` / `run_cosmos_libero.py`
- **knowhow**: cuDNN 版本冲突修复, LIBERO OffScreenRenderEnv API, lingbotvla transformers 4.57 兼容 7 处 patch

### 修复
- **cuDNN 9.1 → 9.10**: `LD_LIBRARY_PATH` 前置 pip cuDNN lib, 解除所有 DiT 模型 (Cosmos/LingBot-VA) 的 core dump
- **LossKwargs TypedDict shim**: transformers 4.57 移除了 LossKwargs, 用 TypedDict stub 注入
- **lingbotvla 7 处 patch**: apply_rotary_emb shim / TypedDict 继承冲突 / use_flash_attention_2 废弃 / sdpa→eager / rotate_half 定义 / expert 直接构造 / vision attn_implementation
- **LIBERO env 创建**: `OffScreenRenderEnv` + `set_init_state` 替代不存在的 `bench.get_env()`
- **Cosmos vendor 5 处 bug**: EvalCfg scoping / get_model tuple unpack / load_dataset_stats string arg / get_image_resize_size / norm_stats guard

### 变更
- **run_libero_all.sh**: 从 3 模型扩展到 5 模型并行 (GPU 0-4), 含 cuDNN LD_LIBRARY_PATH
- **exp03b/07c 搁置**: VLA 无 LIBERO finetune checkpoint, Pi-Zero checkpoint 需 Physical Intelligence gated 授权

### 关键发现
- **Cosmos Policy 97.4%**: 首次在 RTX 5880 上复现, 与论文 98.5% 接近
- **exp03b 0% 预期**: lingbot-vla-4b 是 pretrained foundation model, 非 LIBERO finetune
- **cuDNN 是 DiT 统一 blocker**: 系统 9.1.1 vs torch 需 9.10, pip 包已有但 LD_LIBRARY_PATH 未设

## v0.10.0 @freemty — 2026-05-11/12

### 新增
- **exp11a OpenVLA-OFT**: E=16.8ms / C=92.3ms / A=0.24ms → 109ms / 9.2Hz. Llama-2 7B prefill 占 84%.
- **exp11b StarVLA-OFT**: E=34.7ms / C=28.5ms / A=0.13ms → 63ms / 15.8Hz. OFT MLP 0.13ms (1270x faster than flow).
- **OpenVLA-OFT controller** + **StarVLA-OFT controller**: 共享 `_get_or_create_oft_head` helper
- **exp03b LIBERO eval 脚本**: `scripts/run_exp03b_libero.py` (LingBot-VLA closed-loop eval)
- **Hao meeting v2 slides**: Swiss Knife dark design, 14 slides, 中英混排, lean text. 部署: freemty.github.io/slides/vla-design-space.html
- **slides-voice skill**: `~/.claude/skills/slides-voice/` — slide 文案审核 (去 overclaim/filler/大词)
- **Survey**: Rhoda DVA + Generalist GEN-1 对比精读, Genesis GENE-26.5 ($105M seed)
- **knowhow**: OpenVLA/StarVLA OFT 集成 5 陷阱 (AutoModelForVision2Seq / eager attn / 6ch / grid_thw / embed_tokens)

### 变更
- **Inline probe_core**: 删除 git submodule `src/core`, 5 个文件 (819行) 变普通目录. `sync_to_remote.sh` 简化.
- **README 重写**: 250→71 行, 核心 findings 表 + lean structure
- **Slides**: 690→450 行, 每页一句+数据, slides-voice 审过

### 关键发现
- **OFT 瓶颈翻转**: Action 165ms → 0.13ms, 瓶颈从 action (82%) 转到 backbone (84-99%)
- **两条加速路径**: Path A (压 Action DiT) vs Path A' (OFT + 压 backbone)
- **工业分野对齐**: VLA ~5Hz vs WAM ~0.4Hz, 和我们 9 模型实测一致

## v0.9.0 @freemty — 2026-04-29

### 新增 (Full reproducibility + LIBERO eval)
- **Reproducibility spec**: `docs/specs/2026-04-28-reproducibility-spec.md` — 7 模型官方配置合约 (权重/架构/步数/benchmark)
- **exp06b NitroGen 500M real-weight**: 7.1ms/step (vs exp06a 174M 7.2ms, DiT 实际 181M 非 500M)
- **exp04c Fast-WAM 5-step paper-aligned**: 257ms / 3.9Hz (paper 190ms on A100, 1.35x RTX 5880)
- **exp07b Pi-Zero real pi0-base weights**: 225ms total (vs exp07a random 200ms, Δ=12%)
- **exp04d LingBot-VA checkpoint found**: `robbyant/lingbot-va-posttrain-libero-long` (22.7GB, Apache-2.0)
- **exp04e Fast-WAM LIBERO-4 eval**: 真权重 5-step, **800 ep 完整结果: spatial 91.5% / object 100% / goal 97% / libero_10 89.5% / avg 94.5%** (paper 93.7%)
- **LIBERO 环境 + assets**: xdlab23 fastwam/vit-probe 两个 env 完整安装 (1370 asset files)
- **Wan2.2-TI2V-5B 下载**: 非 Diffusers 版 32GB (Fast-WAM base model)
- `viewer/static/reproducibility.html` — latency realignment + LIBERO success matrix dashboard
- `src/eval/consolidate_matrix.py` — JSON 结果聚合脚本
- **exp09a Cosmos Policy profiling**: Cosmos-Predict2-2B direct-mode e2e timing (论文零报告 latency)
- `viewer/static/reproducibility.html` — latency realignment + LIBERO success matrix dashboard
- `src/eval/consolidate_matrix.py` — JSON 结果聚合脚本
- `exp/exp10_aloha_deferred/` — ACT ALOHA eval 显式 deferred stub
- 4 条 knowhow 归档: LIBERO assets 部署 / hf-mirror 429 / conda env 兼容矩阵 / FastWAM Hydra eval 接口

### 変更
- Slides `hao-meeting-2026-04-28.html`: 新 slide 9 (Reproducibility backup) + slide 3/4/5 数据更新 (real weights + 5-step)
- 实验命名重构: exp09* → exp0{3b,4c,4d,4e,6b,7b,7c} 沿用父实验字母后缀

### 关键发现
- **Random-weight timing 可信**: Pi-Zero real vs random Δ=12%, NitroGen Δ<2%. 权重值不改变 CUDA kernel 计算图
- **DiT 真实参数量**: NitroGen "500M" 实为 DiT 181M + SigLIP 316M. Scaling curve 修正: 181M=7.1ms < 300M=18.3ms < 350M=41ms
- **Fast-WAM LIBERO 完整复现**: 94.5% avg (800 ep) vs paper 93.7% — 匹配在噪声范围内
- **exp09a Cosmos Policy**: direct-mode 1362ms / 0.73Hz (RTX 5880, 5-step EDM), per-step 272ms, VRAM 8816MB

### 修复 (Codex adversarial challenge)
- `run_fastwam_fullweight.sh`: `--steps` → `--num-inference-steps` (argparse mismatch = dead script)
- `consolidate_matrix.py`: `load_libero` 支持聚合 `libero_results.json` 格式 (was silently producing empty LIBERO rows)

### Deferred
- Pi-Zero LIBERO eval (openpi 依赖 flax/JAX, GitHub 被防火墙挡)
- LingBot-VA LIBERO eval (vit-probe cuDNN crash + fastwam env flash-attn build fail)
- ACT ALOHA sim eval → exp10

## v0.8.4 @freemty — 2026-04-28

### 修复 (Codex adversarial review 2026-04-27 — exp08 harness rescue)
在战略转向前把 exp08 harness 的 4 个 showstopper bug 全部修掉并加测试锁, 保证历史可复盘、未来若复活 exp08 无需从零重建
- **Bug 1 修复**: `build_nitrogen_action_payload` 不再静默 fallback 到 dummy `TransformerEncoder`。`require_real=True` 默认, `--allow-dummy-action` 显式 opt-in (commit `4f88b38`)
- **Bug 2 修复**: `run_loop` 把 `barrier.wait()` 从 pre-loop 单次改为 per-iteration, 消除并发 loop free-run drift (commit `30a9e48`)
- **Bug 3 修复**: `build_llm_decode_payload` 拆分为 `_load_decode_model` + `build_llm_decode_payload_testable`, KV cache 只 snapshot 一次, 每次 decode 工作量恒定 (commit `6f3edfb`)
- **Bug 4 修复**: `exp08c_contention_model.py` 新增 `resubstitution()` + `leave_one_out()` + `format_report()`, 暴露真 LOO R² = −12.69 (假 CV 0.94), 负 R² 自动标红 (commit `a476e2f`)
- **Launcher 加固**: `launch_exp08b.sh` 强制 `nvidia-smi -pm 1` + warmup≥15 (exit 3 if fail), 对齐 exp07a canonical 条件 (commit `dfb30e9`)
- **Self-consistency gate**: 新增 `verify_exp08b_baseline.py` — A 在 EA/PA/DA 三组合的 isolated median spread 必须 ≤5%, 否则 exit 1 (commit `a871ed4`)
- **Survey 清洁**: scrub `vla-wam-efficiency-systems-deep-research.md` 里 356 个伪造 `citeturn*` token (commit `0a19640`)
- **6 个单元测试全 pass** (`tests/test_exp08b_harness.py` × 4 + `tests/test_exp08c_contention_model.py` × 2)

### 归档
- 按 v0.8.3 "Fast VLA first, serving later" 战略转向, exp08 全部工件 (3 实验目录 + 5 脚本 + 2 测试 + 1 rescue plan) 移到 `_archive/` 保留历史 (commit `1c54cb1`)
- 活跃树聚焦 exp01-07 profiling, exp08 rescue 8 个 harness commit 保留在 git 历史, 未来若复活可 revert

### 备注
- 原 Codex review 指认的 4 个 showstopper 加 1 个 env 问题全部已修 + 测试覆盖, 归档是"修完之后的归档"而非"跑路式归档"
- SKILL.md v8 + exp/_archive/exp08*/INVALIDATED.md + FINDINGS.md banner 三处都已标明 INVALIDATED 2026-04-27, 防止未来误引用旧数字

## v0.8.3 @freemty — 2026-04-28

### 战略转向
- **"Fast VLA first, serving later"** — VLA 推理当前卡在单请求太慢 (Pi-Zero 5Hz vs 目标 10-50Hz, 2-10x gap)，不是并发不够。定位为 FastVideo 阶段 (单次加速), 不是 vLLM 阶段 (多用户 serving)。候选排序: **A (Action DiT 加速) > B (benchmark) > C (DiT caching) >> D (serving)**。exp08 (contention/serving) 降档为 side project / 备用论文素材
- 决策依据: exp07a 显示 Action Expert 占 82% 延迟; exp04a Fast-WAM 89%; exp04b LingBot-VA 69%。Action DiT 是 80-94% 的 bottleneck

### 新增 (Meeting prep 收尾, 见 Hao 2026-04-28)
- **`docs/hao-meeting-prep.md`** 重构为四幕叙事 — "从 3ms 到 2.5s: 四次跳跃"
  - Act 1 (10min 主菜): Single-Forward (3ms) → Flow Head (74ms) → Flow DiT (18-407ms) → Full WAM (2518ms)。每一步讲清加了什么能力、付了什么延迟代价
  - Act 2 (3min): VLM→VLA attention 崩塌 (Gini 0.91→0.07) 的意外发现
  - Act 3 (5min): 版图空白 + "Fast VLA first" 优先级判断
  - Act 4 (5min): 请教 — 漏了什么架构 / STA 哪些能迁移 / 入学前做什么
- **`viewer/static/design-space.html`** — Action Model Design Space 交互 dashboard (Chart.js + 深色主题)
  - Section 01: 7-model scatter (paradigm × log latency, bubble=params, 10-50Hz 绿带)
  - Section 02: 100% stacked horizontal bar (E/C/V/A 四阶段占比, 标 action %)
  - Section 03: DiT scaling curve (174M/300M/350M per-step + linear 外推线 + cross-attn tax 标注)
- **`docs/meeting-cheatsheet.md`** — 15 分钟面谈前口头速查
  - FastVideo 3 件事: 问题 (85% attention) / 三板斧 (STA + VSA + 蒸馏) / VLA 迁移适用性表
  - DistServe 2 件事: PD disaggregation 机制 / 为什么 "serving later"
  - 7 模型数据速查 (ms / Hz 对照) + DiT scaling + VLA attention 崩塌数字
  - exp08 口头版 (Hao 问才说)

### 変更
- **`docs/TODO.md`** 重新围绕 "Fast VLA first" 排序 — 战略判断置顶; P0 meeting 项多数已 ✅ (Pareto 图, DiT scaling, FastVideo/DistServe 精读→cheatsheet 替代, exp08 一页总结); P1 候选 A/B/C 待 meeting 后启动; P2 exp08 收尾降档
- **`CLAUDE.md`**: `current_exp` = "profiling complete (exp01-07 done), exp08 deprioritized"; `stage` = "meeting prep + learning"; `latest` = v0.8.2; Key References 加 `meeting-cheatsheet.md` 索引
- **`.claude/skills/project-skill/SKILL.md`** v7→v8: 头部加 "Fast VLA first" 战略转向说明; current stage "Strategic Pivot" v0.8.2; 四范式覆盖情况 (exp01-07 全部 done); 下一步 P0 学习 → P0 meeting → P1 候选 A

### 备注
- 本版本无实验跑新结果，专注 meeting prep 战略叙事 + 可视化交付。下一次实验等 Hao meeting 后启动候选 A spec

## v0.8.2 @freemty — 2026-04-27

### 新增 (exp08b/c 完成)
- **exp08b — 6-pair EPDA interference matrix** (6/6 pair complete, RTX 5880 Ada)
  - EP: E=1.70x/P=2.42x. ED: E=1.04x/D=2.59x. EA: E=0.97x/A=1.23x.
  - PD: P=2.42x/D=2.48x. PA: P=2.88x/A=1.19x. DA: D=2.65x/A=1.16x.
  - **D/P 极度脆弱 (2.4-2.9x) vs E/A 鲁棒 (<1.3x)** — 非对称干扰现象确认
  - 产出: `exp/exp08b/interference_matrix.json` + 6 份 `results_XX.json` + `run.log`
- **exp08c — M4 Asymmetric contention model** (R²=0.94)
  - 公式: `inflation(X|Y) = 1 + vulnerability(X) * aggressiveness(Y)`
  - 学到的参数: v=(D:1.52, P:1.61, E:0.23, A:0.20), a=(A:1.12, E:0.96, P:1.02, D:0.87)
  - Baseline 对比: M1 additive R²=0.21, M2 bottleneck R²=-3.44, M3 empirical R²=1.00 (查表上界)
  - 对称模型根本无法拟合该数据 — 不对称性是主信号
  - **A 是"隐形破坏者"**: v=0.20 自身不受影响 + a=1.12 干扰别人最强 (bursty kernel dispatch)
  - **E 是最安全 co-locate 候选**: v=0.23 + 中等 a=0.96
  - EPDA 部署建议: {E,A} safe 同卡; {P,D} 必须 disaggregate; 最优拆分 {E,A} | {P} | {D}
  - 产出: `scripts/exp08c_contention_model.py` (新增 `fit_m4_asymmetric` + 交替最小二乘); `exp/exp08c/FINDINGS.md`, `model_pairs.json`

### Survey 入库
- `survey/papers/vla-wam-efficiency-systems-deep-research.md` — VLA 双轨解读 (omni-multimodal serving vs robotics VLA)。
  **三层心智模型**: 通用 serving 主干 (vLLM/SGLang/TRT-LLM) vs workload 加速器 (vLLM-Omni/SGLang-Diffusion) vs 工程友好层 (LMDeploy)。
  **Robotics 层不同 bucket**: LeRobot (基座 23.6k★) / OpenVLA-OFT (recipe, LIBERO 76.5→97.1%, A100 4.2Hz→109.7Hz, 25-50x) / Fast-WAM (WAM baseline, 190ms/step)。
  **关键数字**: LMDeploy vs vLLM 1.8x 吞吐 + 4-bit 2.4x FP16; TRT-LLM H100 10k tok/s + 100ms TTFT。
  **统一 benchmark**: GenAI-Perf (TTFT/ITL/TPS/RPS 多模态 BYOD)。
  **anti-pattern**: 不要把 LeRobot/OpenVLA-OFT/Fast-WAM tok/s 与 vLLM 直接比 — 不同层, 正确指标是"同任务成功率下的 Hz / 单步 latency / 显存"

### 変更
- `exp/summary.md`: exp08b `scaffolded` → `done (pairs)`; exp08c `scaffolded` → `done (v1)`，填入 M4 参数表
- `.claude/skills/project-skill/SKILL.md`: survey 产出 5→6 份, §3.1.1 新增"系统选型三层心智模型"(serving 主干 / workload 加速器 / 工程层 + robotics 独立 bucket)
- `CLAUDE.md`: Key References 加 `vla-wam-efficiency-systems-deep-research.md` 条目 (完整三层分类 + OpenVLA-OFT/LMDeploy/TRT-LLM/GenAI-Perf 关键数字)

### 构建与工具链
- `.gitignore`: 新增 `.playwright-mcp/` — per-session console/page YAML dumps, 非源码不追踪
- `uv.lock`: 入库 — lingbot/pizero uv venv 复现锚

## v0.8.1 @freemty — 2026-04-27

### 新增
- **WAM demo reproduce 全覆盖** — Fast-WAM / LingBot-VA / NitroGen 三模型 forward pass 验证
  - `scripts/wam_demo_reproduce.py` — standalone 脚本，正确调用 FastWAM.infer_action + LingBot-VA transformer action loop
  - `configs/nitrogen/demo_reproduce.yaml` — NitroGen 真权重 demo config (SigLIP-large + ng.pt)
  - Fast-WAM PASS 2/2 [10,7], LingBot-VA PASS 2/2 [1,16,30], NitroGen PASS 3/3 [1,16,25]
- **exp01a rerun** (warmup=15, iter=20) — 数据已下载到 `exp/exp01a/results_rerun3/`
- **NitroGen ng.pt + SigLIP 权重下载** — via hf-mirror.com，NitroGen 真权重 demo reproduce 通过

### 修复
- **validation_task.py PyTorch 2.9 兼容** — `cuda_time_total` → `device_time_total` (renamed in torch 2.9)
- **demo_reproduce_task.py 支持 tensor 提取** — 从 controller 返回的 dict 中提取 actions tensor，fallback 到 shape key
- **nitrogen_controller.py** — `max_seq_len` 从硬编码 512 改为 config 读取 (默认 1024)；返回 dict 中加入 actions tensor

### 文档
- 4 个 knowhow 文件更新: PyTorch 2.9 API 变更, hf-mirror 下载方案, NitroGen full-weight config mismatch, WAM demo reproduce API

### Survey 入库 (同日产出 4 份，curl-based 一手源 + arXiv ID 全部反验)
- `survey/papers/vla-wam-serving-systems-2026.md` — VLA/WAM 专用 serving 现状。18 arXiv ID curl 校验零幻觉。唯二真 system：**OxyGen** (2603.14371, KV shared + continuous batching) + **VLAgents** (2601.11250, policy server 协议层)。Hao AI Lab 未涉足 → 候选 D' 的 concurrent-work baseline
- `survey/papers/vla-acceleration-tricks-2026.md` — 9 篇 model-level VLA 加速汇总 (PD-VLA / Discrete Diffusion VLA / SnapFlow / FASTER / StreamingVLA / A1 / NanoVLA / OpenVLA-OFT / Fast-WAM) 按 parallel-decoding / one-step-flow / model-shrinking / ft-recipe 四族分类。SnapFlow 独立验证 "A 阶段 80%" 与 exp07a 82% 吻合
- `survey/papers/pi-series-evolution.md` — Physical Intelligence π 系列 (π0 / π0.5 / π*0.6 / **π0.7**) + Generalist **GEN-1** 演进。**2026-04 同月两家都承认 "model is a system"** — π0.7 四件套 (HLP + WM + VLA + Action Expert), GEN-1 blog 原话 "GEN-1 is a system"。exp07a 测的 π0 已是 2024-era single-model picture
- `survey/papers/industrial-wam-landscape-2026.md` — 工业 WAM 全景。Binary 分类 (推理时 video-gen vs 仅 pretrain)。Runtime video-gen 少数派: **1X 1XWM** ("text-conditioned video generation") + **Rhoda FutureVision** ("video-predictive control", $450M Series A / $1.7B val, 2026-03-10 stealth exit) + NVIDIA DreamZero + World Labs RTFM。VLA 主流: PI / Figure Helix / Generalist / Wayve / Cosmos+GR00T。中国工业界无匹配旗舰。1XWM 原文显式点名 VLA 派为对立面

## v0.8.0 @freemty — 2026-04-27

### 新增 (exp08a pilot 完成)
- **exp08a — Pair-wise EPDA co-location interference pilot** (3 runs, RTX 5880 Ada, warmup=10, iter=30)
  - `scripts/exp08a_interference.py` — 两 Python thread + 两 CUDA stream 并发 probe，strategy α (no mid-loop barrier)。支持 DA/PA/SMOKE 三种 pair，自动 fallback dummy DiT 若 NitroGen import 失败
  - 真 NitroGen canonical 结果 (median-based)：
    - **PA** (Qwen-VL-3B P + NitroGen 174M A)：P 27.3→85.9ms (**inflation 3.15×**), A 47.7→65.1ms (1.37×)
    - **DA** (Qwen-VL-7B D + NitroGen 174M A)：D 22.1→77.6ms (**inflation 3.52×**), A 48.2→76.3ms (1.58×)
  - **Roofline 严重低估 2-28×** — "P+A 弱 (<1.1×) / D+A 强 (1.4-1.7×)" 预测中，只有 DA·A 命中
  - Contention 机制推断：GPU kernel launch queue / SM scheduler，不是 FLOPs 或 HBM BW peak
  - LLM 侧 (P, D) 膨胀 >> A 侧 — 非对称干扰。P CV 14.9%→46.7% (compute-bound) vs D CV 7.7%→15.8% (BW-bound)，反直觉
- **`exp/exp08a/FINDINGS.md`** — 完整 pilot 报告 + 见 Hao 一页话 (2026-04-27 版，诚实承认 vLLM-Omni)

### 新増 (multimodal-serving scope audit)
- **`survey/papers/multimodal-serving-systems-2026.md`** — vLLM-Omni + SGLang Diffusion 代码级调研
  - 读了 `github.com/vllm-project/vllm-omni` (4.5k⭐, arXiv:2602.02204, 2026-02 paper) 和 `github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen` (Hao 参与, blog 2025-11-07)
  - **结论**：写新 EPDA framework 的空间已关闭
    - vLLM-Omni 已有 `omni_ar_scheduler` + `omni_generation_scheduler` + `omni_scheduling_coordinator` + `omni_connectors` + `omni_coordinator/load_balancer`，JCT -91.4%
    - SGLang Diffusion 的 `runtime/disaggregation/roles.py` 已有 ENCODER/DENOISER/DECODER/SERVER/MONOLITHIC 五 role，`DiffusionServer` N:M:K
  - 剩余真空白：(B1) robotics/VLA SLO benchmark，(B2) GPU kernel-level contention model，(B3) FastVideo-style 加速迁移，(B4) Visual KV 压缩

### 変更 (exp08 方向降档)
- `survey/papers/hao-style-synthesis.md` — 候选 D **⭐⭐⭐⭐⭐ → ⭐⭐**，新增降档版候选 D' ⭐⭐⭐ (mechanism study + VLA SLO benchmark, 不做 framework)。推荐组合从 "D+C" 改为 **"C+D'"**。见 Hao 的"一页话"重写，开场承认 vLLM-Omni 占据 framework 空间
- `docs/specs/2026-04-26-epda-disaggregation-spec.md` — §0 新增 scope 调整声明；产出清单从 "position paper / framework" 改为 "benchmark + mechanism study + VLA SLO suite"；明确 "不与 vLLM-Omni 竞争"
- `slides/epda-roofline-motivation.html`:
  - 加 EPDA 字母图例条 (E/P/D/A 4 色徽章 + 名字 + 描述) + DistServe → EPD → exp08 lineage callout
  - 加 **exp08a pilot inflation bar chart** (4 pair × 蓝 alone + 红 coloc bar, color-coded ratio vs roofline prediction)
  - 加 **Related serving systems section** (vLLM-Omni / SGLang / DistServe / EPD 4 行对比表 + 2 个 callout)
  - Header 改为 "prediction & exp08a pilot falsification"

### 変更 (project skill v6 → v7)
- `.claude/skills/project-skill/SKILL.md`: current_exp=exp08a, stage=Phase 2 (Co-location Probing), survey 文档数 4→5, 新增重大方向变更段落, 核心数据表加 exp08a 行
- Engineering Lessons APPEND-ONLY **#52-58** (7 新 lesson): GPU power bimodal → warmup=15+pm1, two-thread coloc probe (strategy α), roofline kernel-launch-level 失败 (2-28× 低估), NitroGen import 需 repo root, HF_HOME 布局 `org/repo`, sync_to_remote 静默失败, claim 新 framework 前必须 code-scan vLLM/SGLang

### 実験データ (exp08a canonical median)

| Pair | Phase | Alone (ms) | Coloc (ms) | Inflation | Roofline predicted |
|------|-------|------------|------------|-----------|---------------------|
| PA | P | 27.30 | 85.95 | **3.15×** | <1.1× ❌ |
| PA | A | 47.70 | 65.13 | 1.37× | <1.1× ❌ |
| DA | D | 22.07 | 77.64 | **3.52×** | 1.4-1.7× ❌ |
| DA | A | 48.18 | 76.27 | 1.58× | 1.3-1.6× ✅ |

---

## v0.7.0 @freemty — 2026-04-26

### 新增
- **exp07a 完成** — Pi-Zero dual-stream flow VLA E/C/A profiling on RTX 5880 Ada (20 iterations)
  - **Canonical = stable-window (runs 13-20)**: E=9.32ms / C=26.40ms / A=164.76ms / Total=200.5ms (~5Hz)
  - Action Expert dominates 82% latency; per-step ~16.5ms (cross-attn to PaliGemma KV → 300M Expert ~2.3x pure DiT)
  - DiT scaling curve filled: 174M=7.2ms < **300M=16.5ms** < 350M=32ms
  - **Bimodal 污染**: runs 1-12 比 13-20 慢 1.25x (GPU 功率爬坡)，aggregated 20-run mean 不作为 canonical
  - 下游约定: 后续 profiling 默认 `warmup=15` + `nvidia-smi -pm 1` 锁定 persistence mode
- **PiZeroController** — allenzren/open-pi-zero backend, manual E/C/A phase decomposition
  - Vendor namespace collision solved: `src/` → `pizero_src/` rename + sed rewrite at setup time
  - `.pt` checkpoint loading support (state_dict `_orig_mod.` prefix stripping)
  - `register_profiling_hooks` overridden as no-op (dual-stream requires manual timing)
- **Pi-Zero uv environment** — `.venvs/pizero/` (Python 3.10, torch 2.5.0+cu121)
  - `scripts/setup_pizero.sh` — clone, rename, rewrite imports, install deps
  - `scripts/launch_pizero.sh` — dedicated launcher using pizero venv
  - `scripts/download_pizero_ckpt.sh` — HF checkpoint download (4 variants)
- `configs/pizero/profiling.yaml` — Pi-Zero profiling config with experiment metadata

### 変更
- Project skill v6 — exp07a findings, Pi-Zero in all tables, lessons #47-51
- CLAUDE.md — current_exp 更新到 exp07a, 新增 exp07a key findings
- docs/README.md — 新增 exp07a 行, 版本号更新
- exp/summary.md — exp07a 从 pending → done

### 削除
- Physical Intelligence (openpi) 相关文件清理 — 只保留 allenzren/open-pi-zero 后端

### 后续改进 (exp07a audit 驱动)
- `scripts/_profiling_stats.py` — 新增 standalone 脚本共享的统计 helper，产出与 Hydra PhaseTimer 一致的 JSON (mean/median/std/p10/p90/p99/cv/all_ms)
- `scripts/profile_fastwam.py` — 采用 shared helper，默认 warmup 5→15 以消除 exp07a 暴露的 GPU power-state bimodality
- `docs/TODO.md` — 新增 exp07a audit 衍生的 P0/P1/P2 行动项：exp08 roofline 分析、stream-aware PhaseTimer、standalone 脚本统计口径统一、exp04b 重跑、phase 命名标准化、wall-clock vs phase-sum gap 跟踪

### exp08 准备工作 (roofline 驱动)
- `docs/specs/2026-04-26-epda-roofline-analysis.md` — RTX 5880 Ada 上 E/P/D/A 四阶段的 AI/achieved-utilization 坐标与 ceiling 对比。结论 **GO**：四阶段横跨 3 类 bottleneck (compute / BW-saturated / BW-moderate / latency)，A 阶段 latency-bound 是 LLM 域未研究的新 class
- `slides/epda-roofline-motivation.html` — advisor meeting one-pager (SVG roofline + 4×4 预测干扰矩阵)
- `docs/specs/2026-04-26-epda-disaggregation-spec.md` §1 — motivation 段注入 roofline utilization 表 (E 10% BW / P 17% TF / D 73% BW / A 2-5%)
- `src/utils/timing.py` — PhaseTimer & `_CudaTimerBackend` 支持 `stream=` 参数，exp08a 多-stream 并发测量基础设施。CPU backend 接受并忽略 kwarg 保持 API 稳定。tests/test_timing.py +3 tests (14/14 pass)，现有调用点不受影响

---

## v0.6.1 @freemty — 2026-04-23

### 新增
- **Advisor meeting slides** — `slides/vlla-phase1-results.html` (12 pages)
  - Latency spectrum (9 models, 6 measured + 3 planned with dashed bars)
  - Open-source landscape table (10 models by paradigm)
  - Action Model Design Space 象限图 (latency × generalization, 14 data points, 5 paradigms)
  - Emerging paradigms: Dual-System, MoE-VLA, SSM-VLA, 1-Step Flow
  - 3 insight slides, prediction calibration, next steps

---

## v0.6.0 @freemty — 2026-04-22

### 新增
- **exp05a 完成** — LingBot-VLA-4B attention analysis (6 layers: 0,7,14,21,28,35)
  - VLA fine-tuning 彻底重塑 attention: sink Pos 2→Pos 64, Gini 0.91→0.07, entropy V-shape→flat
  - 结论: VLM token pruning (FastV, FlashVLM) 不可迁移到 VLA
  - Attention structure 是 training objective property 而非 architecture property
- **exp05b 完成** — Qwen2.5-VL-3B-Instruct attention analysis (vanilla VLM baseline, ablation for exp05a)
  - 消歧成功: Gini 崩塌归因于 VLA fine-tuning，非 model size
  - 3B vanilla Gini 0.80-0.98 (与 7B 一致)，Pos 9 sink, entropy V-shape
  - `configs/qwen_vl_3b/attention.yaml` — 新增 vanilla 3B VLM config
- **exp06a 完成** — NitroGen 500M DiT E/C/A profiling + k sweep (k=1..16)
  - Per-step DiT 7.2ms, perfectly linear scaling
  - 174M DiT params, compute-bound → 174M-350M 之间转为 memory-BW-bound
  - k=1: 17.9ms (55.9Hz), k=16: 126ms (7.9Hz)
  - `NitroGenController` — SigLIP→VL-SA→DiT manual timing, random weights mode

### 变更
- Project skill v5 — exp05a/05b/06a findings, lessons #38-46, NitroGen in latency spectrum
- CLAUDE.md — current_exp 更新到 exp06a, 新增 exp05a/05b/06a key findings
- docs/TODO.md — NitroGen profiling 标记完成

### 修复
- **NitroGenController config passthrough** — `init_pipeline()` 从 `controller_config` 读取 NitroGen 专属字段，YAML 配置不再被静默忽略
- **NitroGen profiling warmup/iterations** — 顶层 `num_warmup_runs: 5` / `num_benchmark_runs: 20` 覆盖 base.yaml 默认值 (3/10)

### 可视化
- `viewer/static/scaling-curve.html` — Chart.js 交互式 dashboard: DiT scaling scatter (log-log), latency spectrum (stacked bar), NitroGen k-sweep (dual y-axis), paradigm comparison matrix
- `viewer/static/index.html` — 新增第 4 张导航卡 "DiT Scaling & Latency"，版本号更新到 v0.5.1

### 文档
- `docs/knowhow/debug-solutions/nitrogen-controller-deployment.md` — NitroGen 部署 5 个问题 + Codex 审查发现
- `docs/knowhow/debug-solutions/lingbot-va-wam-integration.md` — LingBot-VA WAM 集成
- `docs/knowhow/toolchain/wam-standalone-profiling.md` — WAM standalone profiling 模式
- `docs/knowhow/runbooks/deploy-new-model-package.md` — sparse clone→tar→scp 部署流程

## v0.5.1 — 2026-04-21

### 新增
- **exp04a 完成** — Fast-WAM (skip-imagination WAM) E/C/A profiling on RTX 5880 Ada
  - @10step: E=7.6ms / C=36.7ms / A=362ms (total 407ms, 2.5Hz)
  - @20step: E=7.6ms / C=36.7ms / A=639ms (total 677ms, 1.5Hz)
  - Action phase 占 89-94%，30-layer MoT cross-attn × N steps
  - `scripts/profile_fastwam.py` — standalone profiler (random/full mode)
- **exp04b 完成** — LingBot-VA (full WAM, video imagination + action) E/V/A profiling
  - E=75.5ms / V=592.5ms (20 steps) / A=1423.1ms (50 steps), total 2091ms (0.5Hz)
  - Full WAM 比 skip-imagination 慢 ~5x
  - Video denoise ~29.6ms/step, action ~28.5ms/step (same DiT, similar per-step cost)
- **LingBotVAController** — 全新 WAM controller，E/V/A 三阶段手动 timer marks (同一 transformer 服务 video+action，module hooks 无法区分)
  - `scripts/profile_lingbot_va.py` — standalone profiler

### 修复
- Fast-WAM profiler: dummy_context 注释明确说明排除 text encoding 原因
- LingBot-VA init_pipeline: 使用独立 `repo_path` 参数而非混用 `model_name` (Codex review P1)
- LingBot-VA full mode: 不加载未使用的 4.7B text encoder，节省 ~9GB VRAM (Codex review P2)

### 文档
- Project skill v4 — exp04a/04b WAM findings, lessons #30-37, WAM benchmark baselines
- exp/summary.md — exp04a + exp04b rows done

## v0.5.0 — 2026-04-20

### 新增
- **Pi-Zero open-pi-zero 后端迁移** — 从 OpenPI (torch 2.7+) 切换到 open-pi-zero (allenzren)，与项目其他模型共享同一 uv venv
  - `PiZeroController` 全面重写：vendor path 校验、cuDNN probe-based 降级、random weights profiling 模式
  - Benchmark: 211ms total ≈ 4.7Hz (224x224, bf16, 10 denoise steps, RTX 5880 Ada)
  - Vendored code: `vendor/open_pi_zero/` (server-side, imports 从 `src.` 改为 `open_pi_zero.`)
- **uv 项目初始化** — `pyproject.toml` (torch>=2.5, transformers>=4.47, hydra, easydict)，替代 conda 作为 Pi-Zero/LingBot-VLA 的默认环境管理

### 变更
- `configs/pizero/profiling.yaml` — 简化为 open-pi-zero 后端配置 (model_name="" = random weights)
- README 更新：Pi-Zero 结果 (211ms/4.7Hz)、uv 安装说明、vendor/ 目录、server uv venv 信息

### 构建与工具链
- `pyproject.toml` — hatchling build backend, ruff config (line-length=100, py311)
- uv env on xdlab23: `.venv/` with torch 2.5+cu121, Python 3.12

## v0.4.4 — 2026-04-20

### 新增
- **exp03a 完成** — LingBot-VLA-4B E/C/A profiling (74.5ms ≈ 13Hz)
  - single_img: E=35.7ms / C=38.3ms / A=0.48ms
  - multi_view: E=36.3ms / C=38.3ms / A=0.48ms
  - 3B backbone 比 7B (exp01a) 快 ~7x
- `BaseVLAController._register_capture_hook()` — analysis hooks 支持 VLA 继承链

### 修复
- LingBotVLAController 继承链 AttributeError (`_register_capture_hook` 只在 VLM 基类)
- Empty safetensors guard — `init_pipeline` 找不到权重时抛 FileNotFoundError
- `scripts/run_remote.sh` command injection — shell 变量加引号
- `scripts/setup_lingbot_vla.sh` 补全 lingbotvla/lerobot 安装步骤，修正 model download 路径

### 文档
- `docs/knowhow/debug-solutions/lingbotvla-integration.md` — BaseVLAController 继承链问题
- Project skill v3 — exp03a findings, lessons #22-29, prediction calibration

## v0.4.3 — 2026-04-17

### 新增
- **Research Presentation Viewer** — 3 个新 HTML 页面，用于 advisor meeting 展示
  - `viewer/static/index.html` — 导航 hub，4 张卡片链接所有子页面
  - `viewer/static/presentation.html` — 5 section 汇报页 (hero, motivation, approach, experiments, roadmap)，含 timing bar chart、attention heatmap 可视化、architecture tree
  - `viewer/static/experiments.html` — 可展开的实验详情页，含数据表格和 prediction calibration
  - FARS dark slate 设计系统 (Instrument Serif + DM Sans + DM Mono, gold accent)
- `viewer/app.py` 新增 catch-all static file 路由

### 修复
- Flask catch-all 路由加 `api/` 前缀防护，避免未注册 API 路径被遮蔽
- multi_image total 数据修正 (`~1.1s` → `~3.6s` at 128 tokens)
- `index.html` body `overflow:hidden` → `overflow-x:hidden; overflow-y:auto` (移动端可滚动)
- Attention heatmap 添加 "illustrative pattern" 标注，避免与实际数据混淆

## v0.4.2 — 2026-04-17

### 文档
- **README.md 全面重写** — 新增 Status Overview (Done/TODO 表格)、3 个实验结果汇总、完整 Architecture 图、7 个 Hydra config 索引、Scripts 索引表、Adding New Models 指南
- TODO 清单覆盖 8 个待完成项（Attention overlay server run, OpenVLA, Pi-Zero, Gradient saliency 等），标注优先级和阻塞原因

### 构建与工具链
- `scripts/run_remote.sh` — 一键 SSH 到 xdlab23 启动实验（封装 launch_exp.sh）
- `scripts/run_local.sh` — 本地 GPU 实验启动（带 logs 输出）
- `scripts/run_viewer.sh` — Flask viewer 启动
- `scripts/run_tests.sh` — pytest 测试套件启动

## v0.4.1 — 2026-04-15

### 新增
- **Timing Cross-Validation Task** (`timing_validation`) — 用 torch.profiler 独立测量 E/P/D phase 时长，与 PhaseTimer CUDA Events 对比验证
  - `src/tasks/validation_task.py` — `_ProfilerPhaseTracker` 在相同 phase boundary 注册 `record_function` hooks，同时运行两套测量
  - 对比报告：每 phase 的 deviation %，PASS (<5%) / WARN (5-15%) / FAIL (>15%) verdict
  - Gap 分析：`sum(E+P+D)` vs end-to-end wall clock，量化 projection/sampling 等漏测时间
  - 自动集成：profiling config 加 `timing_validation` 即启用，无需改动其他代码

### 实验结果
- **exp01b: Qwen2.5-VL-7B attention analysis** (5 layers) — Pos 2 (first visual patch) 是 universal attention sink (12K-18K received, 12-28x vs #2)。Text→Visual Gini >0.91 (extreme sparsity, supports token pruning)。Layer 21 entropy 最低 (3.44)
- **exp02a: ACT (LeRobot) profiling** — Total ~3ms (850x faster than VLM)。Encode ~2.5-2.8ms (80%)，Action ~0.4-0.8ms。VLA latency 下界

### 修复
- 消除 controller 双重注册 (\_\_init\_\_.py + run_tasks.py 重复 import)
- `_aggregated_timing` 从动态属性改为 BaseVLM/VLAController 显式声明
- `head_dim` divisibility check — 防止非 128 head_dim 模型的静默错误
- Multi-image heatmap key collision + defensive detach before numpy

### 文档
- Knowhow 归档 (7 个文件):
  - `docs/knowhow/toolchain/cuda-profiling-patterns.md` — CUDA Event vs torch.profiler 对比
  - `docs/knowhow/debug-solutions/phasetimer-cpu-backend-bug.md` — CPU backend bug
  - `docs/knowhow/debug-solutions/act-action-queue-hooks.md` — ACT action queue 缓存
  - `docs/knowhow/debug-solutions/gqa-attention-analysis.md` — GQA Q/K head mismatch
  - `docs/knowhow/infrastructure/xdlab23-model-weights.md` — 更新 OpenVLA ModelScope 404
  - vision token mapping、Hydra ListConfig/device gotchas

## v0.4.0 — 2026-04-15

### 新增
- **Attention Overlay Visualization** — 将 VLM attention weights 映射回输入图像空间，渲染 heatmap 叠加可视化
  - `src/interpretability/` — Interpretability Mixin 体系：TokenSpatialMap、TokenType 数据结构，BaseInterpretabilityMixin 抽象接口
  - `src/interpretability/vlm_mixin.py` — Qwen2.5-VL 专用 token-to-image 空间映射，从 processor 的 `image_grid_thw` 读取 patch grid，扫描 `<|image_pad|>` token 定位 vision token 范围
  - `src/interpretability/vla_mixin.py` — Pi-Zero VLA placeholder（接口就绪，等模型权重）
  - `src/viz/overlay_renderer.py` — OverlayRenderer：JET/viridis colormap heatmap 叠加、multi-layer 横向对比 strip、GIF 动画输出
  - `src/tasks/attention_overlay_task.py` — 新 task：从 global_store 读 QK → 计算 attention → mixin 空间映射 → overlay 渲染
  - `configs/qwen_vl_7b/attention_overlay.yaml` — Hydra config，5 层 (0/7/14/21/27) attention overlay
- **E2E 验证通过** — Qwen2.5-VL-7B 单图输入，产出 5 层 overlay PNGs + multi_layer_strip.png + layers_sweep.gif + attention_data.json
- 灵感来源：[physical-AI-interpretability](https://github.com/villekuosmanen/physical-AI-interpretability) (overlay 渲染) + rope2sink (log-scale normalization, colormap 体系)

### 修复
- **GQA attention 计算** — `_compute_attention_scores()` 支持 Grouped Query Attention (K heads < Q heads)，通过 `repeat_interleave` 对齐
- **vision token ID 动态获取** — 从 tokenizer 解析 `<|image_pad|>` 而非硬编码 151655
- **layer 排序** — multi-layer strip 和 GIF 使用数字排序 (layer_7 在 layer_14 之前)
- **save_results 路径** — 使用 `base_output_path` 替代不存在的 `output_path`
- **OmegaConf 序列化** — save_results 中 ListConfig → plain list 再 JSON dump
- **launch_exp.sh** — 去掉 `device=cuda:0` override (base.yaml 已定义)
- **profiling_task.py** — 动态遍历所有 phase，不再硬编码 E/P/D
- **ACT controller** — model.forward() 直接调用 + observation.images 格式修复

## v0.3.1 — 2026-04-15

### 新增
- **ML Profiling 系统综合调研** — 8 个系统 (vLLM/SGLang/FastVideo/TensorRT-LLM/DeepSpeed/Triton/llama.cpp/MLC LLM) 的 profiling 实现对比 + CUDA timing 最佳实践
  - `survey/papers/ml-profiling-systems-comprehensive-survey.md` — 综合报告
  - `survey/papers/profiling-systems-survey.md` — 8 系统横向对比
  - `notes/sglang-profiling-deep-survey.md` — SGLang 深度调研
- **Profiling 统计增强** — median, P10/P90/P99, CV (变异系数)，CV>5% 自动警告 `[UNSTABLE]`

### 修复
- **PhaseTimer CPU backend bug** — `record_end()` 被 `if self._use_cuda:` 包裹导致 CPU backend 永远不会被调用
- **profiling_task.py mutation** — `single_run["decode"]` dict mutation 改为 immutable spread pattern

## v0.3.0 — 2026-04-15

### 新增
- **VLM Profiling Framework** (`src/`) — 完整的 VLM inference profiling + attention analysis 框架
  - `model-probe-core` 共享核心 package (git submodule at `src/core/`)，从 rope2sink 提取
  - BaseController → BaseVLMController → QwenVLController 三层继承链
  - PhaseTimer: CUDA event-based E/P/D timing (CPU fallback)
  - 4 个 analysis tasks: `epd_profiling`, `visual_text_attention`, `sink_detection`, `per_layer_stats`
  - Hydra 配置驱动 (`configs/base.yaml`, `qwen_vl_7b/profiling.yaml`, `qwen_vl_7b/attention.yaml`)
- **xdlab23 服务器部署**
  - `scripts/sync_to_remote.sh` — git bundle 同步 (绕过 GitHub firewall)
  - `scripts/launch_exp.sh` — GPU-pinned 实验启动
  - `scripts/download-results.sh` — rsync 结果下载
  - `docs/knowhow/runbooks/deploy-to-xdlab23.md` — 部署 runbook
- **exp01a: Qwen2.5-VL-7B E/P/D Profiling** — 首个实验完成
  - Per-input profiling: text_only / single_image / multi_image
  - 10 benchmark runs + 3 warmup，CUDA event sub-ms 精度
  - 关键发现: Encode 随 image 线性增长，decode per-token 稳定 ~18-21ms
- **Framework Design Spec** (`docs/superpowers/specs/2026-04-14-vlm-profiling-framework-design.md`)
- **Implementation Plan** (`docs/superpowers/plans/2026-04-14-vlm-profiling-framework.md`) — 12 tasks, 44 steps

### 修复
- PhaseTimer: decode timing 改为累加模式 (原 dict 覆盖只保留最后一个 step)
- PhaseTimer: CPU backend `record_end()` 改为立即记录时间 (原为 no-op)
- run_tasks.py: per-input profiling 隔离 (原混跑所有 inputs)
- run_tasks.py: Hydra struct mode + nested config unwrap
- QwenVLController: model path 修正 (`.model.language_model.layers`)
- QwenVLController: OmegaConf → plain dict 转换 for qwen_vl_utils
- probe_core hooks.py: extractor hook 错误改为 log 而非静默吞没

### 更新
- CLAUDE.md: 新增 framework 目录结构、server 信息、quick commands、current state
- project-skill SKILL.md: v0 → v1，新增 framework 架构、exp01a 数据、prediction calibration、7 条 engineering lessons
- .pipeline-state.json: stage dev → experiment, compute_env → xdlab23

## v0.2.1 — 2026-04-11

### 修复
- Dashboard: denoising chart inline styles 替换为 CSS classes
- Dashboard: 删除 canvas glow 死代码 (无效 hex→rgba)
- Dashboard: resize handler 加 rAF 节流
- Dashboard: IntersectionObserver 加 unobserve()
- Dashboard: header "8 Research Gaps" → "6 Research Entry Points" (与实际渲染一致)
- CLAUDE.md: Domain papers 路径从 `docs/papers/` 改为 `survey/papers/`

## v0.2.0 — 2026-04-11

### 新增
- **survey/papers/va-world-models-web.md** — WAM/Video WM/VA 最新论文 web 调研 (80+ 论文)，覆盖 DreamZero、NVIDIA Cosmos、DuoCore-FS 30Hz VLA、PocketDP3 等
- **viewer/static/survey-dashboard.html** — 交互式信息图 dashboard (Pareto frontier, pipeline comparison, maturity matrix, research gaps)
- **.claude/skills/project-skill/SKILL.md** — v0 bootstrap，7 章完整 context 文档 (overview, architecture, cognition, archive, experiments, lessons, reference)

### 修复
- 删除冗余 `experiments/` 目录，统一使用 `exp/`
- CLAUDE.md 目录结构描述补全实际目录
- Session startup 路径从空壳 `docs/papers/landscape.md` 改为 `survey/landscape.md`
- `.gitignore` 补充 Python/macOS/editor/ML artifacts 规则

## v0.1.0 — 2026-04-11

### 新增
- 项目初始化：LabMate 骨架 + git repo
- **survey/landscape.md** — VLM/VLA inference efficiency 全景 survey (80+ 论文)
- **survey/papers/recent-papers.md** — 2025-2026 最新论文 web 搜索 (40+ 论文)
- **survey/papers/va-world-models.md** — VA + World Action Model 深度分析 (60+ 论文)
- **notes/survey-plan.md** — 8 周调研行动计划
- CLAUDE.md 包含项目概述、survey 维度、LabMate workflow

### 其他
- selfOS wiki 同步更新：vlm-vla-real-time-systems concept 扩充、新建 world-action-model concept、方向启动 motivation source page

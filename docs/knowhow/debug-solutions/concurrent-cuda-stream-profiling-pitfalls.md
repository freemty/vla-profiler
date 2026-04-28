# Concurrent CUDA Stream Profiling — 4 Pitfalls

> Co-location 干扰测量 (两 thread + 两 CUDA stream) 极易产生 schedule artifacts 而不是真干扰信号。本文记录 Codex adversarial review (2026-04-27) 在 exp08b harness 暴露的 4 个 showstopper + 各自的修复模板。

## Problem

测"A 相 vs B 相并发运行时互相膨胀多少倍"时, 最 naive 的两 thread 两 stream 实现会报出看起来漂亮但其实是 schedule artifact 的数字 (exp08b: PA inflation 3.15x, DA 3.52x, M4 contention model resub R² 0.94 — 全部虚的)。

## Cause — 4 独立 bug

### Bug 1: Silent dummy fallback 污染 baseline

```python
try:
    from src.controllers.xxx import RealController
    ctrl = RealController(...); ctrl.init_pipeline()
    def step_fn(): ctrl.infer()
except Exception:
    # 静默回退到随机权重 TransformerEncoder — 于是所有 A 相数字都是假的
    model = nn.TransformerEncoder(...)
    def step_fn(): model(x)
```

**后果**: A 相在测 dummy model 而不是真模型, 所有 concurrent 膨胀比例都失去意义。

### Bug 2: 单次 pre-loop barrier → 并发 loop free-run

```python
# 错
if barrier: barrier.wait()         # 只同步一次
for _ in range(n_iter):
    e0.record(stream); step_fn(); e1.record(stream)
    e1.synchronize()
    times.append(e0.elapsed_time(e1))
```

线程启动瞬间同步, 之后各跑各的 — 快的 loop 会把慢的 loop 甩在身后几代 iteration, 后半段的 concurrent 根本不"并发"。exp08b raw array 表现:  E 从 150ms → 68ms 单次 run 内漂移, P 从 105ms → 29ms, median 就是 schedule 切面。

### Bug 3: Stateful payload 跨 iteration 工作量漂移

```python
# 错 — 每次 decode 让 KV 长一点
def step_fn():
    out = model(input_ids=state["token"], past_key_values=state["past_kv"])
    state["past_kv"] = out.past_key_values        # 单调增长
    state["token"]   = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)
```

KV cache 单调增长 → 每次 decode 工作量不同 → iteration N 的 latency 本身就是 N 的函数, concurrent 膨胀 = 干扰 + workload drift, 无法区分。

### Bug 4: 同一 `step_fn` 跨 warmup/isolated/concurrent/next-combo 复用

Warmup 阶段已经 mutate 过 payload state, isolated baseline 测的就不是"冷启动"workload, next combo 又继承上一 combo 的 state leakage。典型症状: A 的 isolated median 在 EA=51.2ms vs DA=36.8ms, 差 39% — 纯粹 order effect, 但 matrix 对比毫无意义。

### 附加 env 坑

- Launcher 漏 `nvidia-smi -pm 1` → GPU clock 在 warmup 中爬坡, 与 exp07a canonical 条件不一致 (见 `cuda-profiling-patterns.md`)
- Warmup<15 在 RTX 5880 Ada 上会被 P-state 爬坡污染出 bimodal 分布

## Solution — 修复模板

### 1. No silent fallback

```python
def build_payload(gpu, require_real=True):
    try:
        ctrl = RealController(...); return ctrl.infer
    except Exception as e:
        if require_real:
            raise RuntimeError(
                f"{ctrl} unavailable ({e}); refusing dummy fallback. "
                f"Pass --allow-dummy-action to explicitly opt in."
            ) from e
        # dummy path only when explicitly opted in
```

CLI 层用 `--require-real-models` / `--allow-dummy` flag 强制, 默认 raise。

### 2. Per-iteration barrier

```python
for _ in range(n_iter):
    if barrier is not None:
        barrier.wait()                 # ← 每次 iter 开头同步
    e0.record(stream); step_fn(); e1.record(stream)
    e1.synchronize()
    times.append(e0.elapsed_time(e1))
```

代价: barrier 把所有线程对齐到最慢那条, fast phase 会在 barrier 上等 — 但这就是 co-location 定义本身, 不是 bug。想测"两 phase 重叠的窗口内 fast phase 被拖慢多少", 就必须让窗口真正重叠。

### 3. Frozen state per call

```python
def build_llm_decode_payload_testable(...):
    proc, model = _load_model(...)
    with torch.no_grad():
        prefill_out = model(**inputs, use_cache=True)
    fixed_kv    = prefill_out.past_key_values    # snapshot 一次
    fixed_token = torch.argmax(prefill_out.logits[:, -1, :], -1, keepdim=True)
    state = {"past_kv": fixed_kv, "token": fixed_token}

    def step_fn():
        with torch.no_grad():
            model(input_ids=state["token"],
                  past_key_values=state["past_kv"],  # 从不更新
                  use_cache=True)
        # 故意 NOT 更新 state — 每次 call 都是恒定工作量
    return step_fn, state
```

测试钩子: 暴露 `_testable` 返回 `(step_fn, state)`, 监控 `state["past_kv"][0].shape[2]` 10 次调用后仍等于初始值。

### 4. Self-consistency gate

run 相同 phase 在 3 个不同 combo 的 isolated section, 相对 spread 必须 ≤5% 才可信。exp08b 里 A 的 isolated 在 EA/PA/DA 三组合 spread 39% 就是警报:

```python
# scripts/verify_exp08b_baseline.py
baselines = {c: median(run_combo(c)["isolated"]["A"]["all_ms"]) for c in ["EA","PA","DA"]}
spread = (max(baselines.values()) - min(baselines.values())) / mean(baselines.values())
assert spread <= 0.05, f"order effect / state leakage — spread {spread:.1%}"
```

Gate 不过就**不要**进入 matrix run, 不是"先跑再说"。

## Commands

```bash
# Dry-run self-consistency gate
python scripts/_archive/verify_exp08b_baseline.py --gpu 0 --iterations 20 --tol 0.05

# Env 对齐 exp07a canonical (launcher 已自动)
sudo nvidia-smi -pm 1
nvidia-smi -i 0 --lock-gpu-clocks=tdp,tdp

# 用 CPU smoke payload 验证 harness 结构 (不用 CUDA)
python -m pytest tests/_archive/test_exp08b_harness.py -v
```

## 测试模板 (unit, 无 CUDA)

1. Barrier per-iter: MagicMock `threading.Barrier`, 断言 `wait_count >= n_iter`
2. 恒定 workload: monkeypatch 一个 fake model 返回固定长度 KV, 10 次 call 后 shape 不变
3. 并发不 collapse: 两个 CPU phase (10ms vs 40ms), per-iter barrier 下 fast median 应该 ~10ms 而不是 被 collapse 成 40ms
4. Abort on load fail: monkeypatch `sys.modules["real_module"]=None`, `require_real=True` 必须 raise

## Notes

- Date: 2026-04-28
- Source: Codex adversarial review of `exp08b_interference_matrix.py` 2026-04-27
- Related files (archived): `scripts/_archive/exp08b_interference_matrix.py`, `tests/_archive/test_exp08b_harness.py`
- Related knowhow: `docs/knowhow/toolchain/cuda-profiling-patterns.md` (persistence mode + warmup=15 基础)
- 这些 pitfalls 在任何 "两 stream 并发测 contention" 的未来实验都会再出现 — 包括如果某天复活 exp08 contention 方向, 或者做 multi-request VLA serving 的 latency breakdown

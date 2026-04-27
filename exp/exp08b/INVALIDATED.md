# INVALIDATED — do not cite 2026-04-27

All JSON results in this directory (`results_*.json`, `interference_matrix.json`) are
produced by a broken harness. Do NOT use these numbers in slides, commits, PRs, weekly
reports, or `exp/summary.md` downstream analysis.

## What is broken

1. **A phase used dummy TransformerEncoder, not NitroGen.** `build_nitrogen_action_payload`
   silently fell back to a random-weights TransformerEncoder (`scripts/exp08b_interference_matrix.py:203-223`).
2. **No per-iteration cross-stream barrier.** Concurrent loops sync once at start, then
   drift. Per-iter medians are schedule artifacts (`results_EA.json` shows E 150ms→68ms
   within one run).
3. **Decode KV cache grows across iterations.** D's workload drifts monotonically.
4. **Launcher missed `nvidia-smi -pm 1`** — exp07a canonical requires persistence mode.

## Rescue plan

See `docs/superpowers/plans/2026-04-27-exp08bc-rescue.md`.

## What to cite instead

Nothing from this directory until the rescue plan's harness self-consistency gate
(Task 9) passes.

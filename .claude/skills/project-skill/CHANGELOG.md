# Project Skill Changelog

- v7 (2026-04-27): exp08a pilot done, vLLM-Omni scope audit, exp08 downgraded to mechanism study + VLA SLO

## 2026-04-21 — v4 WAM Profiling Complete (Fast-WAM + LingBot-VA)

Incremental update: exp04a Fast-WAM done, exp04b LingBot-VA full WAM done, LingBotVAController added.

- **Updated:** Section 1 — current_exp → exp04b (done), core data table +2 rows (Fast-WAM, LingBot-VA)
- **Updated:** Section 2 — LingBotVAController added, WAM E/V/A phase model documented, profile_fastwam.py + profile_lingbot_va.py scripts
- **Updated:** Section 3 — 2 new confirmed hypotheses (action phase dominance, full WAM 5x slower), 1 new open hypothesis (MoT vs shared-DiT routing)
- **Updated:** Section 4 — WAM benchmark baselines table (per-step cost reference), Fast-WAM + LingBot-VA in Pareto table
- **Updated:** Section 5 — exp04a + exp04b rows with prediction calibration
- **Updated:** Section 6 — 3 new systematic biases (#7-9: WAM per-step cost, action step count, video imagination overhead)
- **Added:** Engineering Lessons #30-#37 (WAM shared DiT profiling, sys.path injection, VAE z_dim, init_latent shape, timestep batch dim, text encoder VRAM, FlowMatchScheduler shift, streaming VAE cache)
- **Added:** Quick Reference: 2 new scripts, LingBotVAController registry entry

## 2026-04-20 — v3 LingBot-VLA-4B + Codex Review + Presentation Viewer

Incremental update: exp03a done, Codex adversarial review fixes, research viewer.

- **Updated:** Section 1 — current_exp → exp03a (done), core data table +1 row, version v0.4.3
- **Updated:** Section 2 — LingBotVLAController + PiZeroController in tree, _register_capture_hook note on both base classes, 3 environment strategy, new scripts/viewer/docs files
- **Updated:** Section 3 — 4 new verified hypotheses (exp03a), 3 new active hypotheses (LingBot attention, scaling law, OpenVLA)
- **Updated:** Section 4 — LingBot-VLA in Pareto table, 3 rejected alternatives (uv, eager attn, separate capture hook)
- **Updated:** Section 5 — exp03a row + prediction calibration 5.4 (worst: 1/4 accurate)
- **Updated:** Section 6 — 2 new systematic biases (backbone scaling non-linear, context underestimated), calibration trend
- **Added:** Engineering Lessons #22-#29 (inheritance chain, empty weights, shell injection, .forward() hooks, uv, PI0Config, patchified scaling, Flask catch-all)
- **Added:** 7 new commands, 5 knowhow entries, 2 new registry entries in Quick Reference

## 2026-04-15 — v2 Attention Overlay + exp01b/exp02a + VLA Controller

Incremental update: attention overlay visualization, VLA controller hierarchy, 3 experiments done.

- **Updated:** Section 1 — current_exp → exp02a, 3 experiments completed, core data summary table
- **Updated:** Section 2 — new modules (interpretability/, viz/, validation_task, attention_overlay_task), VLA controller branch (BaseVLAController → ACTController), 9 knowhow files, 2 new specs/plans
- **Updated:** Section 3 — 3 hypotheses verified (sink, sparsity, VA latency), 4 new active hypotheses, exp01b/exp02a findings
- **Updated:** Section 4 — ACT first-party baseline in Pareto table, Rejected Alternatives table
- **Updated:** Section 5 — exp01b + exp02a rows, prediction calibration 5.2/5.3, meta-learning trends
- **Added:** Engineering Lessons #12-#21 (GQA, head_dim check, ACT action queue, multi-image key collision, OmegaConf, vision token ID, timing cross-validation, detach before numpy)
- **Added:** Section 8 Knowhow Index table, 3 new launch commands, expanded registries

## 2026-04-15 — v1 Framework + exp01a

Major update: profiling framework built, first experiment completed.

- **New:** `src/` profiling & analysis framework (BaseController → BaseVLMController → QwenVLController)
- **New:** `model-probe-core` shared submodule extracted from rope2sink
- **New:** xdlab23 server deployment (scripts, runbook, git bundle sync)
- **New:** exp01a done — Qwen2.5-VL-7B E/P/D timing (per-input: text/image/multi)
- **Updated:** Section 1 (stage: dev → experiment), Section 2 (framework architecture), Section 3 (5 verified hypotheses from exp01a)
- **Added:** Section 5.1 Prediction Calibration, Engineering Lessons #4-#11
- **Added:** Server info, registry reference to Quick Reference

## 2026-04-11 — v0 Bootstrap

Initial bootstrap — auto-generated from existing codebase.

Scanned: CLAUDE.md, 4 survey documents (2856 lines total), notes/survey-plan.md, git history (3 commits), exp/ (empty).

Generated 7 sections: Project Overview, Architecture, System Cognition (3 strategic judgments + 8 research entries), Technical Archive (4 paradigms + maturity matrix + migration matrix), Experiment History (empty + 5 planned), Engineering Lessons, Quick Reference.

## v9 — 2026-05-14

- exp09a Cosmos Policy profiling (659ms/1.5Hz, per-step 76.8ms)
- exp11a OpenVLA-OFT (109ms/9.2Hz) + exp11b StarVLA-OFT (63ms/15.8Hz)
- Cosmos LIBERO eval 97.4% + Fast-WAM 94.5% baselines
- exp04d running, exp03b/07c shelved status
- OFT bottleneck flip discovery (Action → Backbone)
- 8 new engineering lessons (#59-71): cuDNN fix, LossKwargs shim, lerobot_stub, LIBERO env API, 4-GPU parallel eval
- LIBERO Quality Baselines table (Section 4.3)
- Prediction calibration for exp09a, exp11a/b, LIBERO

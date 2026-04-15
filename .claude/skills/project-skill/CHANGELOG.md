# Project Skill Changelog

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

# exp09 — Full Reproducibility + LIBERO Evaluation

**Goal:** Align all 7 profiled models to official weights + configs, then produce a unified latency × LIBERO success rate matrix.

**Spec of record:** `docs/specs/2026-04-28-reproducibility-spec.md`

## Sub-experiments

| ID | Model | Type | GPU |
|----|-------|------|-----|
| exp09_latency_rerun | All 6 VLA/VA models | Latency (official configs) | 0-5 |
| exp09a_fastwam_libero | Fast-WAM | LIBERO-4 success rate | 0 |
| exp09b_pizero_libero | Pi-Zero (pi0-base) | LIBERO-4 success rate | 1 |
| exp09c_lingbotvla_libero | LingBot-VLA-4B | LIBERO-4 success rate | 2 |
| exp09d_lingbotva_libero | LingBot-VA (conditional) | LIBERO-4 success rate | 3 |

## Not in scope (this sprint)

- ACT ALOHA sim eval → deferred to exp10
- NitroGen task-success benchmark (no public harness)

## Deliverables

- `exp/reproducibility_matrix.json` — consolidated latency + success rate
- `viewer/static/reproducibility.html` — 2-panel dashboard
- Updated `slides/hao-meeting-2026-04-28.html` with slide 11

## Status

- [ ] Phase 0: Prep (spec, LIBERO install)
- [ ] Phase 1: Latency realignment
- [ ] Phase 2: LIBERO eval
- [ ] Phase 3: Dashboard + slides

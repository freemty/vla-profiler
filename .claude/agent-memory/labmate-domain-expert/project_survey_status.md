---
name: Survey Project Status
description: Status of VLM/VLA/VA/WAM inference efficiency landscape survey, updated 2026-04-11 with VA+World Model deep dive; OpenPI profiling feasibility assessed 2026-04-13
type: project
---

Completed comprehensive landscape survey at `survey/landscape.md` on 2026-04-11.
Added VA + World Action Model deep dive at `survey/papers/va-world-models.md` on 2026-04-11.
Added OpenPI PyTorch profiling feasibility report at `docs/papers/openpi-pytorch-profiling-feasibility.md` on 2026-04-13.

**Coverage:**
1. VLM Inference (architectures, visual token explosion, KV-cache, vision encoder bottleneck)
2. VLM efficiency techniques (token pruning, serving systems, speculative decoding, quantization)
3. VLA Inference (autoregressive vs diffusion/flow vs hybrid, real-time challenges)
4. VLA efficiency (token pruning, speculative decoding, denoising step compression, edge deployment)
5. VA (non-language) models -- **deep dive added**: Diffusion Policy family, DP3, ACT, BeT, 1-step flow
6. World Action Models -- **deep dive added**: Dreamer series, DreamZero, Cosmos Policy, DDP-WM, Sparse Imagination
7. World Model for VLA fusion routes (data augmentation, visual planner, unified WAM)
8. VA/VLA/WAM inference efficiency comparison matrix
9. Systems perspective (serving needs, speculative rollout, ERA disaggregation)
10. FastVideo -> WAM acceleration transfer analysis
11. Gap Analysis from vLLM/FastVideo perspective
12. Efficiency Techniques Taxonomy (kernel/scheduling/model/architecture/system)
13. 5+ recommended research entry points

**Key findings (updated):**
- VLM serving is in "pre-vLLM" stage -- EPD disaggregation is the natural extension of DistServe
- No dedicated VLA/WAM serving system exists (major gap)
- VA inference has reached near-physical limits: 1-step flow = 0.56ms (Action-to-Action Flow)
- VA's new bottleneck is now vision encoding (~10ms), not action generation
- DreamZero (2026) proved WAM can be zero-shot policy, but only at 7Hz
- FastVideo distillation -> 1-step WAM is a high-value transfer opportunity (7Hz -> 40Hz+)
- DDP-WM achieves 9x speedup via dynamics disentanglement
- Sparse Imagination (ICLR 2026) enables token-sparse rollout for real-time WM planning
- ERA (Encode-Rollout-Action) disaggregation is a natural extension of EPD for WAM serving
- Speculative rollout for world models is an unexplored but promising systems concept
- Algorithm-System co-design is the biggest white space

**OpenPI profiling assessment (2026-04-13):**
- PyTorch path available since Sept 2025 (torch==2.7.1, transformers==4.53.2 required, pinned)
- Architecture: SigLIP ViT-So400m/14 (E) + PaliGemma Gemma 2B LM (C) + Gemma 300M Action Expert (A)
- Dual-stream Transformer: PaliGemma + Action Expert run in parallel per layer, sharing KV attention
- Default: num_steps=10 Euler denoising, KV cache built once per prefix (C), reused across A steps
- Key module paths: paligemma_with_expert.paligemma (E+C) / paligemma_with_expert.gemma_expert (A)
- BLOCKER: requires isolated conda env (conflicts with PyTorch 2.9 and transformers 5.0)
- BLOCKER: checkpoint only available on GCS in JAX format, needs conversion script

**Why:** User needs comprehensive field overview before starting PhD research.
**How to apply:** Use this survey as foundation for experiment design and paper archival. Update landscape.md and va-world-models.md as new papers are discovered.

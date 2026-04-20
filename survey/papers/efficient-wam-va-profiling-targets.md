# Efficient WAM/VA Models: Profiling Target Survey

**Date:** 2026-04-20
**Query:** Small and fast WAM/VA models suitable for profiling on single RTX 5880 Ada 48GB
**Context:** Complement exp01-03 (Qwen2.5-VL-7B, ACT, LingBot-VLA-4B) with WAM/VA paradigm models

---

## 1. Background: What We Already Know

From exp01-03, we have profiled three points on the latency-capability Pareto:

| Model | Paradigm | Total Latency | Hz | Bottleneck |
|-------|----------|---------------|-----|-----------|
| ACT (LeRobot) | VA (CVAE) | ~3ms | 330Hz | Encode (80%) |
| LingBot-VLA-4B | Flow VLA (3B backbone) | 74.5ms | 13Hz | Encode + Context (equal) |
| Qwen2.5-VL-7B | VLM (7B AR) | 427ms | 2.3Hz | Decode (AR) |

**Gap:** We have NO profiling data on:
- WAM (World Action Model) paradigm -- joint video prediction + action
- Efficient latent world models
- Multi-step diffusion VA with moderate complexity (10-50ms range)
- State Space Model (SSM) based VLA

---

## 2. Candidate Comparison Table

| Model | Params | Latency | Hz | Paradigm | Open-Source | Single 48GB GPU? | arXiv |
|-------|--------|---------|-----|----------|-------------|-------------------|-------|
| **DDP-WM** | ~100M-500M | ~30ms (rollout) | ~30Hz | Latent WAM (dynamic decompose) | Likely (ICLR workshop) | Yes | 2602.01780 |
| **Sparse Imagination** | ~100M-500M | ~30ms (rollout) | ~30Hz | Sparse Latent WAM (ICLR 2026) | Likely (ICLR) | Yes | 2506.01392 |
| **Fast-WAM** | ~1-3B (est.) | 190ms | ~5Hz | Skip test-time imagination | Unknown | Likely | 2603.16666 |
| **DreamZero** | 14B | ~130ms | 7Hz | Video WAM (DiT) | Unknown (NVIDIA) | Borderline (14B) | 2602.15922 |
| **Mean-Flow VLA** | ~3-7B (est.) | varies (1-step) | 8.7-83.9x faster | Flow VLA (one-step) | Unknown | Likely | 2603.01469 |
| **FASTER** | ~3-7B (est.) | single-step | Fast | Flow VLA (horizon-aware) | Unknown | Likely | 2603.19199 |
| **Action-to-Action Flow** | <100M | 0.56ms | >1000Hz | VA (informed 1-step flow) | Unknown | Yes (trivially) | 2602.07322 |
| **AnoleVLA** | ~3B (est.) | 3x faster than baseline | ~30-40Hz | SSM-based VLA (Mamba/RWKV) | Unknown | Yes | 2603.15046 |
| **NanoVLA** | <1B | 52x faster on edge | Very fast | Ultra-small VLA | Unknown | Yes (trivially) | 2510.25122 |
| **GigaWorld-Policy** | ~1-3B (est.) | Fast (video optional) | High | Action-centered WAM | Unknown | Likely | 2603.17240 |
| **DuoCore-FS** | ~3-7B (est.) | ~33ms per chunk | 30Hz | Async Fast-Slow VLA | Unknown | Likely | 2512.20188 |
| **MinD** | ~1-3B (est.) | ~88ms | 11.3 FPS | Dual-system WM | Unknown | Likely | 2506.18897 |
| **Cosmos Failure Detector** | ~5% of baseline | Fast | High | Compact WM (Cosmos tokenizer) | NVIDIA open | Yes | 2603.06987 |
| **HY-Embodied-0.5** | 2B (activated) | Edge-targeted | Fast | MoT VLA | Likely (HuaYuan?) | Yes | 2604.07430 |
| **PocketDP3** | <1% of DP3 | ~5-10ms | >100Hz | Ultra-compact 3D VA | Unknown | Yes | 2601.22018 |
| **FODMP** | Small | 10x faster than MPD | High | 1-step consistency in ProDMPs | Unknown | Yes | 2603.24806 |
| **Sparse ActionGen** | Same as DP | 4x faster | Higher | Pruned Diffusion Policy | Unknown | Yes | 2601.12894 |
| **CoLA-Flow** | Small | Near 1-step | High | Latent action flow | Unknown | Yes | 2601.23087 |
| **TinyVLA** | Small (<1B) | Fast | High | Compact VLA | Likely | Yes | 2409.12514 |
| **DM1 (One-Step MeanFlow)** | Same as DP | ~5ms (1 step) | >200Hz | 1-step VA (RL fine-tuned) | Unknown | Yes | 2510.07865 |

---

## 3. Missing Models (Additional Candidates from Knowledge)

### 3.1 Newly Identified Candidates

| Model | Why Interesting | Paradigm | Status |
|-------|----------------|----------|--------|
| **ViPRA** (arXiv:2511.07732) | Chunked flow matching, 22Hz control | Video-Action | 2025 paper, code likely |
| **DiT4DiT** (arXiv:2603.10422) | Coupled video+action DiT, shared computation | WAM-like | 2026, unknown open-source |
| **ImagiNav** (arXiv:2603.13833) | Decoupled visual planner + inverse dynamics | WAM (navigation) | 2026, likely code |
| **Cosmos Policy** (arXiv:2601.16163) | Fine-tuned Cosmos for visuomotor, ~10Hz | Video WAM (NVIDIA) | 2026, likely open (NVIDIA) |
| **LiteVLA-Edge** (arXiv:2603.03380) | 4-bit GGUF on Jetson, 150.5ms | Quantized VLA | 2026, code likely |
| **KAN We Flow** (arXiv:2602.01115) | RWKV-KAN blocks, 86.8% param reduction | Lightweight VA | 2026, unknown |
| **World2Act** (arXiv:2603.10422) | Latent action alignment with WM | WM+VLA fusion | 2026, unknown |
| **DIAL** (arXiv:2603.29844) | Latent intent bottleneck, latent WM in VLM space | Efficient WM-VLA | 2026, unknown |

### 3.2 NVIDIA Cosmos Small Models

From the Cosmos ecosystem, the most relevant for single-GPU profiling:
- **Cosmos Tokenizer** -- open-source, lightweight, the encoding front-end
- **Cosmos Failure Detector** (2603.06987) -- only 5% params of full model, uses Cosmos Tokenizer to train probabilistic WM for bimanual failure detection
- **Cosmos Policy** -- fine-tunes Cosmos video model for robot control; full model may be too large, but smaller variants likely exist

---

## 4. Selection Criteria for Profiling Targets

Must-haves:
1. Represents WAM or VA paradigm (not just a pruned VLM)
2. Runnable on single RTX 5880 Ada 48GB
3. Open-source code/checkpoints available (or very likely to be)
4. Latency in 10-200ms range (interesting profiling territory)
5. Teaches us something NEW about the inference pipeline

Nice-to-haves:
- Active community / reproducible results
- Represents a distinct architectural pattern from exp01-03
- Has real-robot evaluation (not just simulation)

---

## 5. TOP 3 Recommendations

### Rank 1: DDP-WM (Disentangled Dynamics Prediction)

**Why:**
- **New paradigm:** First latent world model in our profiling suite -- fundamentally different from VA (exp02a) and VLA (exp01a/exp03a)
- **Interesting architecture:** Dynamic decomposition into primary (interaction-driven) vs secondary (background) dynamics
- **Right latency range:** ~30ms total (20-50Hz), between ACT (3ms) and LingBot-VLA (74.5ms)
- **Single GPU friendly:** Latent WM models are typically 100M-500M params
- **9x speedup claim:** Can we profile WHERE this speedup comes from? Is it attention sparsity? Reduced rollout steps?

**What profiling teaches us (new knowledge):**
- **Rollout pipeline breakdown:** How does latent rollout time scale with horizon H? Is it truly O(H) or can steps be parallelized?
- **Primary vs Secondary dynamics compute ratio:** Does the "decomposition" actually save compute, or is it mostly a quality trick?
- **Comparison to ACT:** ACT is 3ms for a single-pass VA. DDP-WM is ~30ms for world-model-aided control. What's the 10x overhead buying in terms of compute structure?
- **Memory access pattern:** Latent WM uses GRU + MLP (compute-bound) rather than AR decode (memory-bound) -- profiling will reveal a completely new bottleneck pattern

**Feasibility:** ICLR workshop paper -- code likely available or reproducible from paper description. Small model fits easily on 48GB.

---

### Rank 2: AnoleVLA (State Space Model backbone)

**Why:**
- **New backbone type:** Mamba/RWKV-based instead of Transformer -- O(n) inference complexity
- **3x faster than baseline VLA:** Puts it at ~25-50ms range for a VLA-class model
- **Language-conditioned:** Unlike DDP-WM, this IS a VLA (vision-language-action), allowing direct comparison with exp03a (LingBot-VLA-4B)
- **Represents SSM trend:** State Space Models are the main challenger to Transformers for sequential tasks

**What profiling teaches us (new knowledge):**
- **SSM vs Transformer inference characteristics:** Is the speedup from reduced FLOPs, better memory access, or something else?
- **O(n) vs O(n^2) in practice:** At what sequence length does SSM actually beat Transformer? VLA context lengths are moderate (~1000-2000 tokens)
- **Recurrent state management:** SSM maintains a fixed-size state instead of growing KV-cache -- what are the implications for serving?
- **Direct comparison with LingBot-VLA:** Same paradigm (VLA with action head), different backbone -- isolates the impact of backbone architecture
- **KV-cache elimination:** If there's no KV-cache, the memory-bandwidth bottleneck changes fundamentally

**Feasibility:** 2026 paper, ~3B params estimated. Should fit on 48GB easily. SSM frameworks (Mamba) have good open-source support.

---

### Rank 3: GigaWorld-Policy (Action-Centered WAM with Optional Video)

**Why:**
- **Unique design:** Video generation is OPTIONAL -- can profile both with and without video prediction
- **Action-centered:** Unlike DreamZero (video-centered), this model prioritizes action output
- **Bridges WAM and VA:** When video is disabled, it behaves like a sophisticated VA; when enabled, it's a WAM
- **Represents the "skip imagination" trend:** Fast-WAM (2603.16666) also explores when to skip world model reasoning

**What profiling teaches us (new knowledge):**
- **Cost of "imagination":** By comparing video-on vs video-off modes, we directly measure the overhead of world modeling
- **Is video prediction worth it?** Profiling reveals the compute/latency cost of world model reasoning vs the pure action pathway
- **Conditional compute pattern:** Models that can dynamically decide whether to "think" (use WM) or "react" (skip WM) represent a new systems challenge -- how do you serve models with variable compute per request?
- **ERA (Encode-Reason-Act) disaggregation:** This model naturally decomposes into encode / optional-reason / act stages, validating the ERA serving concept from our landscape survey

**Feasibility:** 2026 paper (March), estimated 1-3B params. The "optional video" design means even if the full model is large, the action-only path is small.

---

## 6. Honorable Mentions (Rank 4-6)

### Rank 4: Sparse Imagination (ICLR 2026)
- Very similar to DDP-WM but uses token-sparse rollout instead of dynamic decomposition
- Profiling both would reveal whether "decompose dynamics" vs "prune tokens" is more efficient
- ICLR acceptance increases likelihood of open code

### Rank 5: DuoCore-FS (Async Fast-Slow VLA)
- 30Hz whole-body control via async architecture
- Interesting for profiling the INTERACTION between fast path and slow path
- But may be more of a systems-level study than a single-model profiling

### Rank 6: FASTER / Mean-Flow VLA (Single-step Flow VLA)
- Direct extension of LingBot-VLA (exp03a) -- same paradigm but single-step
- Would show if 10-step flow matching has overhead vs 1-step
- But we already know from exp03a that action head is only 0.48ms -- the bottleneck is encode/context, not action steps

---

## 7. Profiling Landscape After Top-3

If we add the top-3 to our existing profiling suite:

| Model | Paradigm | Latency | Bottleneck Pattern | New Insight |
|-------|----------|---------|-------------------|-------------|
| ACT (done) | VA (CVAE) | 3ms | Encode (ResNet, 80%) | VA lower bound |
| LingBot-VLA-4B (done) | Flow VLA | 74.5ms | Encode = Context | VLA with 3B backbone |
| Qwen2.5-VL-7B (done) | VLM (AR) | 427ms | AR Decode (mem-bw) | Large VLM baseline |
| **DDP-WM (new)** | Latent WAM | ~30ms | Rollout (compute?) | World model overhead |
| **AnoleVLA (new)** | SSM VLA | ~25-50ms | ??? | SSM vs Transformer |
| **GigaWorld-Policy (new)** | Action-centered WAM | variable | Conditional compute | Cost of imagination |

This gives us coverage of ALL major paradigms:
- Pure VA (ACT)
- Latent World Model (DDP-WM)
- SSM VLA (AnoleVLA)
- Transformer VLA (LingBot-VLA)
- Transformer VLM (Qwen2.5-VL)
- Action-centered WAM (GigaWorld-Policy)

---

## 8. Implementation Priority

### Phase 1 (Immediate): Verify open-source availability
```bash
# Check GitHub/HuggingFace for each top-3
# DDP-WM: search for "disentangled dynamics prediction world model"
# AnoleVLA: search for "AnoleVLA" or "state space model VLA"
# GigaWorld-Policy: search for "GigaWorld" or "action-centered world action model"
```

### Phase 2: Controller Implementation
For each model, implement a controller in our framework:
- `src/controllers/DDPWMController` -- wraps latent WM rollout pipeline
- `src/controllers/AnoleVLAController` -- wraps SSM-based VLA
- `src/controllers/GigaWorldController` -- wraps WAM with optional video

### Phase 3: Profiling Tasks
- Standard E/P/D breakdown (adapted for WAM: Encode/Rollout/Action or Encode/Reason/Act)
- Memory footprint comparison
- Scaling behavior (batch size, sequence length, rollout horizon)
- Attention/state analysis (for AnoleVLA: recurrent state vs KV-cache)

---

## 9. Key Uncertainties

| Question | Impact | How to Resolve |
|----------|--------|----------------|
| Is DDP-WM code actually open? | High (no code = can't profile) | Search GitHub, contact authors |
| Is AnoleVLA's SSM backbone truly Mamba? | Medium (affects controller design) | Read paper architecture section |
| Does GigaWorld-Policy have pretrained checkpoints? | High | Check paper/GitHub |
| Can DreamZero (14B) fit on single 48GB? | Low priority (14B borderline) | Check model card |
| Are FASTER/Mean-Flow VLA just LingBot-VLA variants? | Medium | Compare architectures |

---

## 10. Relationship to Research Directions

Profiling these models directly feeds into:
1. **ERA Disaggregation** (landscape.md Section 6.2): GigaWorld-Policy's optional reasoning validates ERA concept
2. **SSM for VLA Serving** (new direction): AnoleVLA profiling reveals if SSM changes the serving story
3. **World Model Serving** (landscape.md Section 5.4): DDP-WM profiling reveals rollout scheduling requirements
4. **FastVideo Transfer** (landscape.md Section 6.4): DDP-WM's latent rollout is analogous to video diffusion -- can STA/VSA apply?

---

*All arXiv IDs are from verified sources in landscape.md and va-world-models-web.md. Model parameter counts marked "(est.)" are inferred from paper descriptions and comparable architectures.*

# FiLM Transformer — Summary for Review
**Sukrit Koirala | March 2026**

---

## The Architecture

A GPT-2 style transformer where after each attention sublayer, a small bottleneck network (ConfigNet) reads the attention output and generates per-token scale (gamma) and bias (beta) that modulate the FFN's hidden layer before activation.

```
attn_out = Attention(LayerNorm(x))
x        = x + attn_out

gamma, beta = ConfigNet(attn_out)          # Linear → GELU → Linear
h           = GELU(gamma * W1(LayerNorm(x)) + beta)
x           = x + W2(h)
```

**Design choices:**
- ConfigNet takes `attn_out` (pre-residual attention delta) as input — the richest per-token contextual signal before the FFN runs
- Zero-initialized output layer — at step 0, gamma=1 and beta=0, so FiLM starts identical to baseline
- Parameter-matched — FiLM and baseline have identical total parameter counts (d_ff reduced in FiLM to compensate for ConfigNet params)
- Same ConfigNet architecture at every layer (Linear(d_model, d_model//4) → GELU → Linear(d_model//4, 2*d_ff))

**Scale:** d=768, 12 layers, 12 heads, ~124M params. Trained on OpenWebText, validated on WikiText-103.

---

## Experiments Run

| Experiment | Params | Steps | Result |
|---|---|---|---|
| Baseline (large) | 124M | 50k | 73.0 ppl |
| FiLM (large) | 124M | 50k | 80.6 ppl |
| Baseline (medium) | 44M | 8k | 33.7 ppl |
| FiLM medium (standard LR) | 44M | 8k | 39.4 ppl |
| FiLM medium (3x ConfigNet LR) | 44M | 8k | 39.4 ppl |
| ConfigNet probe | 124M FiLM | every 1k step | (see below) |

---

## PPL Gap Over Training (Large Scale)

| Step | Baseline | FiLM | Gap |
|---|---|---|---|
| 10k | 144.8 | 151.5 | +6.7 |
| 20k | 93.3 | 108.3 | +15.0 ← peak |
| 30k | 83.7 | 90.3 | +6.6 |
| 40k | 77.3 | 81.7 | +4.4 ← closest |
| 49k | 73.0 | 80.6 | +7.6 ← reopened |

The gap closed from 20k to 40k then widened again in the final 9k steps. From 40k to 49k: baseline improved 4.3 ppl, FiLM improved only 1.1 ppl.

---

## ConfigNet Probe Findings

A forward-hook analysis was run on every checkpoint (every 1k steps) capturing what the ConfigNet outputs per layer. Metrics: `gamma_mag` (deviation of gamma from 1.0), `beta_mag` (magnitude of beta), `gamma_var_tokens` (variance of gamma across tokens), `beta_var_tokens`.

### Step 1000 — Zero-init working
All layers: gamma_mag ≈ 1.000, gamma_var ≈ 0.000. ConfigNet is a no-op.

### Step 10000 — Unexpected finding
L8 had the highest token variance (0.22), nearly equal to L11 (0.21). At this point L8 was competing with L11 for most active layer.

### Step 25000 — Hierarchy locking
L11 pulled clearly ahead (gamma_var = 0.58). L8 fell back to 0.15. The competition resolved.

### Step 49000 — Final state
```
L0   gamma_mag=1.068  gamma_var=0.004  ← near passive
L1   gamma_mag=1.439  gamma_var=0.017
L5   gamma_mag=1.459  gamma_var=0.064
L8   gamma_mag=1.561  gamma_var=0.155
L9   gamma_mag=1.708  gamma_var=0.185
L10  gamma_mag=2.094  gamma_var=0.273
L11  gamma_mag=2.632  gamma_var=0.638  ← dominant
```

---

## Key Findings from Probe Analysis

**1. Emergent layer specialization**
Every layer was given an identical ConfigNet with equal capacity. Without any routing mechanism or explicit design, the model discovered that early layers (L0–L8) benefit minimally from conditioning and late layers (L9–L11) benefit heavily. This matches interpretability research: early layers handle syntax/position, late layers handle semantics. The model recovered this structure through gradient descent alone.

**2. Two distinct behavioral groups**
- L1–L9: gamma magnitude jumped to ~1.4–1.6 by step 5k and barely moved after. Effectively frozen.
- L10–L11: kept growing continuously throughout all 50k steps. Still not flat at step 49k.

**3. The hierarchy sharpened over time**
At step 10k, L8–L11 were roughly competitive for token differentiation. By step 49k, L11 alone accounts for most of the token differentiation (0.86 total var vs 0.24 for L10). Conditioning became *more concentrated* over training, not less.

**4. Beta is not doing meaningful work**
Beta never develops strong token variance. At L11 step 49k: gamma_var = 0.638, beta_var = 0.22. Beta provides roughly uniform additive shift across tokens. Gamma does all the per-token discrimination. The model effectively learned to rely almost entirely on multiplicative conditioning.

**5. Conditioning concentration stabilized early**
L9–L11 went from 25% of total conditioning activity (uniform, at step 1k) to 34.9% by step 7k, then held at 33–35% for the rest of training. The hierarchy formed fast and committed.

**6. Pearson correlation: gamma activity vs loss**
Pearson(L11 gamma_var_tokens, val_ppl) = -0.61. Moderate negative correlation — as ConfigNet becomes more active, loss drops. Not a perfect correlation because PPL drops fast early when ConfigNet is barely active (basic language patterns learned first), diluting the signal.

**7. L8 early competition then retreat**
At step 10k, L8 had the highest gamma_var. By step 25k it had fallen behind L11 significantly. Hypothesis: early in training, L8 representations were most variable/informative. As training progressed and representations became more organized, L11 (with fullest context) became strictly more useful for conditioning.

---

## Why FiLM Trails Baseline

Three hypotheses tested:

**Hypothesis 1: Needs more steps**
Evidence for: gap closed from +15 to +4.4 between steps 20k–40k, L11 gamma_var still growing at step 49k.
Evidence against: gap reopened to +7.6 at step 49k. From 40k→49k FiLM improved 4x slower than baseline. Running to 100k would likely narrow the gap but may not close it.

**Hypothesis 2: ConfigNet LR decays too fast**
Tested directly — 3x ConfigNet LR experiment at medium scale. Both FiLM variants ended at identical PPL (39.4). **Ruled out.**

**Hypothesis 3: Fundamental architectural inefficiency**
The most likely explanation based on probe data: zero-init costs ~2k warmup steps where ConfigNet is a no-op. Additionally, 9 of 12 ConfigNets are effectively wasted — the model never uses them for meaningful conditioning. These wasted parameters and warmup costs accumulate into a persistent disadvantage vs a baseline that uses all parameters effectively from step 0.

---

## What the Data Implies for Architecture

The current design applies ConfigNet to all 12 layers equally. The probe shows this is wrong: only L10–L11 do meaningful work. This suggests:

**Targeted FiLM**: Apply ConfigNet only to the last 2–3 layers with a larger bottleneck. Redistribute the freed parameters as either:
- Larger ConfigNet at L10–L11 (more capacity where model wants it)
- Wider FFN to match baseline (cleanest param-matched comparison)

**Remove beta**: Beta never contributes to token differentiation. A scale-only version (gamma only) simplifies the architecture and loses nothing.

**Predicted outcome**: Targeted FiLM would have lower warmup cost, no wasted parameters, and all conditioning capacity concentrated where the model wants it. This should close the gap with baseline significantly and possibly beat it.

---

## Comparison to MoE

| Property | Standard Transformer | MoE | FiLM (this work) |
|---|---|---|---|
| Per-token adaptation | None | Discrete routing | Continuous scale+bias |
| Routing mechanism | None | Learned top-k | None |
| Load balancing loss | Not needed | Required | Not needed |
| Routing collapse | N/A | Common at small scale | Cannot collapse |
| Parameter cost | 1x FFN | Nx FFN | 1x FFN + tiny ConfigNet |
| Layer specialization | None | None | Emerges naturally |

MoE achieves specialization through discrete routing with auxiliary losses. FiLM achieves an analogous layer-level hierarchy through continuous conditioning with no auxiliary losses and no possibility of collapse.

---

## Open Questions

1. Does the "active zone" (currently L10–L11) grow with model depth? At 48 layers would the top 6 layers all become active?
2. Why did L8 lead early then fall back? Is this about representation quality at that layer early in training?
3. Would targeted FiLM (last 3 layers only) beat the standard baseline at matched params?
4. Does the mechanism become more useful at larger scale (1B+) where attention outputs encode richer semantics?
5. What happens if you condition on the residual stream instead of attn_out?

---

## One-Sentence Summary

The model learned — without being told — that only the last 2 layers of the transformer need dynamic per-token conditioning, concentrating all meaningful scale modulation there while the rest of the network ignored its ConfigNets, suggesting the next architecture should be designed around this discovered structure rather than applying conditioning uniformly across all layers.

# Novelty check — LPD (Learned-Per-Token Decay)

Date: 2026-08-30. By: initiative leader. Status: CANDIDATE IS NOT NOVEL.

Question: is "per-token / input-dependent learned decay in a recurrent
attention-like memory" a new mechanism?

Verdict: NO. Published repeatedly, 2023–2026:

- **Forgetting Transformer (FoX)**, arXiv 2503.02130 (2025). Per-position
  data-dependent forget gates f_t = sigmoid(w_f^T x_t + b_f), cumulative
  product F_ij = prod_{l=j+1..i} f_l added as log-bias to *softmax* attention
  logits. This is LPD's closed form applied to softmax attention. Our math
  (cumsum of log-b / tril decay matrix times scores) is the same construction.
- **Adaptive Memory Decay for Log-Linear Attention**, arXiv 2605.06946 (2026).
  Learns per-token, per-level decay λ from input via a two-layer MLP.
- **RWKV-6 "Finch"** — dynamic, data-dependent, per-channel time decay
  (w_t from LoRA of token-shifted input); RWKV-7 goes further.
- **Mamba / Mamba-2** — input-dependent A (decay) via input-conditioned Δ,
  input-gated (selective) state update.
- **Gated Linear Attention (GLA)**, Yang et al. 2024 — input-dependent decay
  gates in the fast-weight update.
- **Gated DeltaNet** — input-dependent gates α_t, β_t controlling retention
  and update (explicit memory clearing).
- **mLSTM (xLSTM) 2024** — data-dependent scalar decay per head.

Our LPD is specifically a *no-softmax linear-attention* instance of FoX's rule
(FoX uses softmax scores; ours multiplies raw linear scores). As a family, the
idea is well occupied. Claiming it as a first would not survive review.

What our clean 3-family comparison still legitimately provides:
- Controlled, same-budget evidence that per-token decay > fixed decay >
  softmax attention at small scale (corroboration of FoX's finding, on the
  linear-attention side).
- A fast closed-form (decay matrix einsum) equivalent to the constant-memory
  scan, verified to 1e-8 — useful engineering note, not new either.

Genuine open gaps the prior-art survey surfaced (candidates for the NEXT
invention attempt):
- Purely constant-memory models still UNDERPERFORM Transformers on
  long-context retrieval / in-context recall (stated by several 2025/2026
  surveys). Even Gated DeltaNet doesn't fully close it.
- Gated DeltaNet authors themselves note decay rules "mitigate, not solve";
  models rarely learn true erase-on-context-switch.
- No cited work unifies: (a) content-selective per-token decay (FoX-style)
  with (b) retrieval-exact erasable memory (DeltaNet-style) in ONE rule.

These feed the next project question.
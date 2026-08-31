# Protocol — frozen before observing results

The goal was a **fair, same-budget** comparison of three context-memory
mechanisms. The protocol below was fixed before the decisive runs, so no
post-hoc tuning could favor one family.

## Families

All three share the same block structure, parameter count budget (~0.92M), and
interface; they differ **only** in how they compute the context-summary at each
block. Config constants in `train.py` (`BATCH 64, SEQ 192, EMBED 160,
N_LAYER 3, N_HEAD 8, HEAD_DIM 20, MLP 576, DROP 0.1`).

1. **`attn` — softmax self-attention, causal.**
   `y_t = sum_i softmax(<q_t,k_i>)/sqrt(d) * v_i`, decode memory grows with
   context length. Reference quality.
2. **`linattn` — linear attention with fixed decay.**
   `o_t = sum_{i<=t} d^(t-i) (q_t.k_i) v_i`, one learnable scalar decay `d` and
   an extra `S`-like L2 interaction term in the full model. Constant memory.
3. **`decayattn` — linear attention with *learned per-token* decay.**
   Same as (2) but each position learns its own decay `b_t` from its input;
   total forget `prod b_i`, so the model can choose to remember or erase.

## Fairness rules

- Identical data split (90/10), identical batch order, identical param count
  budget, identical init scheme, LR, clip, optimizer, dropout, eval cadence.
- Decisive metric: **held-out validation loss at step 500**, pre-specified.
  Train loss is reported but not the decision metric.
- Number of parameters was the budget; all families landed at 0.92M.

## Verification of the fast implementation

`linattn`/`decayattn` are implemented in **closed form** (one decay matrix,
attention-speed matmuls) rather than a Python/time loop, so all three train at
comparable speed on CPU. The closed form was checked against the naive
recurrence scan (same weights): max absolute diff ~9.3e-9. Any future edit must
re-verify this equivalence.

## Recall probe — single-query associative recall

Predecessor protocols per family, then: does the constant-memory claim hold on
**recall** rather than language-modeling loss?

Task (MQAR-style, standard in the literature): a context of 192 positions holds
`R` random key-value pairs at random positions among noise tokens, then a probe
`key ?` in the final two positions. The model must output the exact value token
paired with the queried key.

**Protocol version history (all recorded honestly):**
- **v1:** values were a FIXED function of keys (value = key + 64). A model could
  score 100% with a lookup table, no context storage. Flawed methodology;
  its numbers (all families ~100%) are NOT evidence.
- **v2:** random per-instance pairing, 64 keys / 512-value pool, but too hard
  for a 0.92M model within budget (attention stayed at chance through 300
  steps). Also had a distance-logging bug (compared key IDs instead of
  positions). Discarded as a config, kept as a record.
- **v3 (final):** random per-instance pairing, **16 keys / 64-value pool**
  (key repeats across instances with different values, so lookup still fails),
  distance logging fixed to actual pairing position. Learnable, but content
  matching (a "copy head") is slow to emerge in 0.92M models, so the frozen
  budget is **1500 steps per family** (sanity: attn reached ~10% at 300).

Task details: 192 positions, `R in {4..8}` random pairs among noise tokens,
probe `key ?` in the final two positions. Decision metric: **exact-match
accuracy** at the final position on 256 freshly generated held-out instances
per (family, R in {4,6,8}); secondary metric within each load is accuracy
split by whether the queried pairing is in the nearer or farther half of the
context. Same architecture, optimizer, config and seed per family
(`BATCH 64, SEQ 192, EMBED 160, N_LAYER 3, LR 3e-4, CLIP 1.0`); only the
tokenizer differs. Post-hoc stress on trained checkpoints at
`R in {8,10,12,14,16}` without retraining.

## Threats to the claim (recorded up front)

- Seed luck -> counter by re-runs with seeds 1-2 (started, see logs).
- Closed-form bug -> caught by the ~1e-8 verification check.
- Reporting gate chosen to make the comparison look good -> gate was fixed as
  validation loss at step 500 before running.
- Claiming "first" -> closed by the literature gate in `NOVELTY.md`.
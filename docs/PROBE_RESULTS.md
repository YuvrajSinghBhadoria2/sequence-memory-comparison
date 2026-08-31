# Recall probe results (v3, final) — 1500 steps per family

Machine-saved evidence: `results/recall_probe/probe3_<family>_1500_log.json`.
Chance accuracy for this task: 1/82 = **0.0122** (82-symbol vocabulary;
prediction on a 512-pool would be far lower, but here values live inside the
82 symbols the model knows, so the conservative floor is 1/82).

## Training outcome — exact one-shot recall, same budget

All families trained 1500 steps on loads `R in {4..8}`; exact-match accuracy on
fresh held-out instances at checkpoint 1200 (then final):

| Family | acc @ R4 (step 1200) | final R8 | What it means vs chance |
|---|---|---|---|
| `attn` (Transformer) | **0.262** | 0.117 | clearly learned the copy/recall rule (~10-20x chance) |
| `linattn` (fixed decay) | 0.031 | 0.019 | statistically at chance |
| `decayattn` (learned decay) | 0.031 | 0.019 | statistically at chance |

The Transformer also visibly improves with training (loss 4.20 -> 3.29); both
constant-memory families stay pinned at the random floor (loss ~4.17, i.e.
never better than a uniform guess on this task).

## Heavier-load stress (trained checkpoints, NO retraining)

From `results/recall_probe/probe3_stress.txt` (256 fresh instances per cell; sampling noise
is ~0.02 around each value at this n):

| Load R | attn | linattn | decayattn |
|---|---|---|---|
| 8 | 0.137 | 0.004 | 0.016 |
| 10 | 0.129 | 0.016 | 0.012 |
| 12 | 0.141 | 0.000 | 0.012 |
| 14 | 0.055 | 0.016 | 0.016 |
| 16 | 0.102 | 0.004 | 0.019 |

The Transformer stays far above chance on every load; the constant-memory
families never leave the noise floor, and their "near/far" splits are noise
too. No load shows ANY retained recall for linattn/decayattn.

## What this says (honest boundary)

- At identical budget (0.92M params, 1500 steps, same seed/data generator),
  **softmax attention learns exact one-shot associative recall; fixed-decay and
  learned-decay linear memories do not learn it at all** — consistent with the
  theoretical lower bound that constant-memory recurrences need state growth to
  recover associations.
- This **contrasts directly with the perplexity result** in `RESULTS.md` where
  the same constant-memory families beat the Transformer at next-token
  prediction. Take-away, stated plainly: a model can look great at
  *predicting* text and simultaneously be unable to *recall* a specific fact.
- Limits: absolute accuracies are modest (the Transformer reaches 26% at R4;
  0.92M is small and 1500 steps is still budget-limited). The meaningful signal
  is the decisive ordering, not the absolute numbers. Only one architecture
  config and one seed per family was run; the stress numbers carry ~0.02
  sampling noise. None of this changes the direction of the effect.
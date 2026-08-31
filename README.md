# Which memory mechanism wins? A controlled, honest, laptop-scale comparison

A small, reproducible experiment comparing **three ways a sequence model can
remember context**, trained under identical conditions on one CPU-only laptop:

| Memory | Idea | Params |
|---|---|---|
| `attn` | Softmax self-attention (Transformer) | 0.92M |
| `linattn` | Linear attention with a **fixed** decay per step | 0.92M |
| `decayattn` | Linear attention with a **learned, per-token** decay (learned forget) | 0.92M |

**What happened:** per-token learned decay beat both fixed decay and softmax
attention at the same parameter budget on a held-out validation split
(char-level Shakespeare, 600 training steps):

- `decayattn` (learned forget): validation loss **1.792**
- `linattn` (fixed decay): validation loss **2.003**
- `attn` (Transformer): validation loss **2.201**

**…but flip the question to exact one-shot recall and the winner inverts.**
The same budgets were re-tested on associative recall (find the value paired
with a key in the context). The Transformer learns the task (~26% exact, ~20x
chance) while both constant-memory families stay at the random floor (~3%,
chance is 1.2%) — see `results/PROBE_RESULTS.md`. A model can look great at
*predicting* text and still be unable to *recall* a specific fact.

**Why it matters:** this reproduces, at small scale and on the linear-attention
side, a finding published by the Forgetting Transformer (FoX, arXiv:2503.02130):
letting each token *learn how much to forget* is better than a fixed decay or no
decay. The repository also ships a **closed-form linear-time implementation of a
"constant-memory recurrent scan,"** verified numerically equal to the recurrence
(max abs diff ~1e-8), which is the trick that makes such models trainable at
attention speed.

## The honest headline

The winning mechanism is **not claimed as new**. A proper literature check
before any novelty claim found per-token learned decay already published
several times (FoX 2025; RWKV-6; Mamba; Gated Linear Attention; Gated DeltaNet;
mLSTM; see `NOVELTY.md` for the full list and how this experiment maps onto it).
This repository's real value is the **controlled, reproducible comparison
itself** — exact same data, budget, init, and eval — plus the verified
closed-form, plus a worked example of checking novelty *before* claiming it.

## Quick start

```bash
python -m venv .venv311 && source .venv311/bin/activate
pip install torch==2.2.2 numpy==1.26.4

# train one family (tiny smoke test first)
python train.py attn 30          # ~15 s on a laptop CPU
python train.py attn 600
python train.py linattn 600
python train.py decayattn 600

# sample text from a trained checkpoint
python generate.py decayattn

# recall probe: measure exact recall, not just language-model loss
# (1500 steps/family: content matching is slow to emerge in tiny models)
python probe_recall.py attn 1500
python probe_recall.py linattn 1500
python probe_recall.py decayattn 1500

# stress the trained models with heavier memory loads (no retraining)
python probe_stress.py attn
python probe_stress.py linattn
python probe_stress.py decayattn
```

First run auto-downloads Tiny Shakespeare (~1.1 MB, MIT). ~600 steps take
15-40 min per family on a laptop CPU. Plot `results/*_600_log.json` to see the
curves. The probe (`probe_recall.py`) is fully synthetic — no downloads —
and answers a different, still-open question: whether constant-memory models
can recall specific facts, not just predict text.

## Read next

- `RESULTS.md` — exact numbers, curves, and what they mean.
- `results/PROBE_RESULTS.md` — the recall probe: who can actually remember.
- `PROTOCOL.md` — the frozen protocol and fairness rules.
- `REPRODUCE.md` — full environment and reproduction steps.
- `NOVELTY.md` — the literature check and why no novelty claim is made.
- `PROVENANCE.md` — where everything came from and who did what.
- `AGENTS.md` — suggested next experiments.
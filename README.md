# Sequence Memory Comparison

**A controlled, reproducible study of how three sequence-model memories behave
— on predicting text, and on remembering facts.** Runs entirely on a laptop CPU,
no GPU, no paid compute.

A plain-language summary: [Elevator pitch](#elevator-pitch) · Results ·
[Layout](#repository-layout) · [Quick start](#quick-start)

---

## Elevator pitch

Sequence models must choose between two kinds of memory:

- **Attention** (Transformer): remembers anything in context, but memory
  grows with the text you read.
- **Recurrence** (linear / decay / state-space): constant memory, but must
  "forget" old input.

Everyone has an opinion about which wins. We stopped guessing and measured it —
**same model size, same budget, same data, one laptop** — on two very different
questions:

1. **Prediction:** which one best predicts the *next character* of a text?
2. **Recall:** which one can *exactly retrieve a specific fact* placed
   earlier in the context?

**Our headline finding:** the winner flips depending on the question. The
memories that best *predict* text are the worst at *recalling* facts — and
vice versa.

| Question | Winner |
|---|---|
| Predict next token (language modeling) | Learned per-token decay ("linear" memory) |
| Recall a specific fact (associative recall) | Softmax attention |

---

## What is this

Three 0.92 M-parameter sequence models that share everything except **how they
compute their context memory**:

| Family | Memory rule | Files |
|---|---|---|
| `attn` | Softmax self-attention: attend to *all* past positions | `experiments/train.py` |
| `linattn` | Linear attention with a **fixed** decay per position | `experiments/train.py` |
| `decayattn` | Linear attention with a **learned, per-token** decay (the model chooses how much to forget at each step) | `experiments/train.py` |

Both linear families are implemented in **closed form** (attention-speed
matrix operations) and verified numerically equal to their constant-memory
recurrence form (max abs diff ~1e-8) — so the comparison is apples-to-apples:
your choice is about the *memory rule*, not the implementation speed.

Each family is evaluated with two scripts:

- `train.py` — language modeling on Tiny Shakespeare (char-level).
- `probe_recall.py` + `probe_stress.py` — **associative recall**: the model
  sees random `key -> value` pairs in context, then `key ?`, and must output
  the exact value. (History: we caught and fixed two flaws in this probe
  ourselves — see `docs/PROTOCOL.md`. Cheating tests are worthless.)

---

## The problem

Modern sequence modeling is dominated by two poles:

- **Transformers** give near-unlimited recall but pay *memory* that grows
  with context — a real cost at long documents.
- **Constant-memory models** (SSMs, linear attention, RWKV-style, delta nets)
  keep memory flat but trade away exact recall.

Published work argues constant-memory models fall behind on long-context
**retrieval**, while matching or beating attention on **perplexity**. Most
comparisons happen at large scale with big compute, all-or-nothing budgets,
and mixed protocols — making them hard to trust. This project is the
small-scale equivalent: a clean, controlled, reproducible test that anyone
can run on a laptop and check.

---

## What we did

1. **Froze the protocol before observing results** (fairness rules in
   `docs/PROTOCOL.md`): identical data split, batch order, parameter budget
   (0.92 M), init, optimizer, learning rate, and seed per family.
2. **Trained all three** on 600 steps of character-level language modeling
   (Tiny Shakespeare) and measured held-out validation loss.
3. **Trained all three** on 1500 steps of associative recall and measured
   exact-match accuracy on freshly generated held-out instances.
4. **Stress-tested recall** by evaluating the already-trained models at
   heavier memory loads (no retraining).
5. **Checked the literature** before claiming anything: the per-token-decay
   family is already published (FoX & others) — see
   [the honest novelty note](#honest-about-novelty).

Evidence is machine-saved JSON logs (not chat claims); every number below
links to a log file.

---

## Results

### 1) Prediction: held-out validation loss (lower = better)

600 steps, same budget (raw logs:
`results/language_modeling/*.log.json`):

| Family | Val loss @ step 500 |
|---|---|
| `decayattn` (learned forget) | **1.792** |
| `linattn` (fixed decay) | 2.003 |
| `attn` (Transformer) | 2.201 |

→ **Learned per-token decay predicts text best.** Consistent with the
Forgetting Transformer line of work (no novelty claimed). The strict ranking
also holds across seeds 1-3 (fresh init + batching); seed-robustness details
and the four-seed table are in `docs/SEED_ROBUSTNESS.md`.

### 2) Recall: exact-match accuracy (higher = better)

1500 steps, same budget, chance = 1.2% (raw logs:
`results/recall_probe/*.log.json`):

| Family | Recall @ load R4 | final @ R8 | vs chance |
|---|---|---|---|
| `attn` (Transformer) | **0.262** | 0.117 | ~20x above chance |
| `linattn` (fixed decay) | 0.031 | 0.019 | at chance |
| `decayattn` (learned decay) | 0.031 | 0.019 | at chance |

→ **Only the Transformer learns to recall facts.** The constant-memory
families — including the *better predictor* — never leave the random floor.
Same story in all four seeds (0-3): `attn` is 5-10x chance every time,
`linattn`/`decayattn` are at chance every time (`docs/SEED_ROBUSTNESS.md`).

### 3) Stress: heavier loads, no retraining

Accuracies on loads 8–16; the Transformer stays ~5–15x above chance on every
load while both constant-memory families stay at chance
(`results/recall_probe/probe3_stress.txt`):

| Load | attn | linattn | decayattn |
|---|---|---|---|
| 8  | 0.137 | 0.004 | 0.016 |
| 12 | 0.141 | 0.000 | 0.012 |
| 16 | 0.102 | 0.004 | 0.019 |

---

## Our insight

**"Which memory is better" is the wrong question — "better at what" is the
real one.**

A model that predicts text beautifully (low perplexity) can still be
completely unable to *recall* a fact it was given. That single sentence, shown
cleanly at laptop scale in one reproducible repo, is the useful takeaway:

> **Look at how a model is evaluated before believing how good it is.**
> Perplexity ≠ recall. If your product needs exact retrieval (RAG, memory,
> tooling), evaluate *recall*, not just next-token loss.

This also frames a well-known industry tension in a concrete, checkable way:
constant-memory "efficient attention" rules trade retrieval quality for
constant memory *even at tiny scale* — and the effect doesn't wait for
long-context setups to appear.

---

## Honest about novelty

We do **not** claim that any mechanism in this repo is new. The learned
per-token decay family was published before us (Forgetting Transformer,
arXiv [2503.02130](https://arxiv.org/abs/2503.02130); also RWKV-6, Mamba,
Gated Linear Attention, Gated DeltaNet, mLSTM — full list with our mapping in
`docs/NOVELTY.md`). The "constant-memory models lag on recall" story is the
subject of active research and of a theoretical lower bound.

What this project contributes is a **clean, controlled, reproducible
demonstration** of both effects at laptop scale, with the recall test
corrected and disclosed honestly (two flawed probe versions documented in
`docs/PROTOCOL.md`). It is a reliability/evidence artifact — deliberately not
a discovery claim.

---

## Repository layout

```
.
├── README.md              ← you are here
├── LICENSE                MIT
├── requirements.txt       torch 2.2.2, numpy 1.26.4
├── AGENTS.md              continuation guide for humans/AI agents
├── experiments/           the runnable code (all self-contained)
│   ├── train.py           language modeling (3 families)
│   ├── generate.py        sample text from a trained model
│   ├── probe_recall.py    associative-recall probe (v3, final)
│   └── probe_stress.py    heavier-load evaluation (no retraining)
├── docs/
│   ├── RESULTS.md         tables + reading the numbers
│   ├── PROBE_RESULTS.md   recall-probe numbers + honest boundary
│   ├── PROTOCOL.md        frozen rules + version history (incl. 2 fixed flaws)
│   ├── REPRODUCE.md       full setup and reproduction steps
│   ├── PROVENANCE.md      data/environment/authorship
│   └── NOVELTY.md         literature check & why nothing is claimed new
└── results/
    ├── language_modeling/ raw per-step loss logs (600-step runs)
    └── recall_probe/      raw recall logs (1500-step runs) + stress output
```

---

## Quick start

```bash
git clone https://github.com/YuvrajSinghBhadoria2/sequence-memory-comparison
cd sequence-memory-comparison

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # CPU torch is fine

# 0) smoke test (seconds): proves the toolchain works
python -m experiments.train attn 30

# 1) language-modeling comparison (~20-35 min/family on a laptop CPU)
python -m experiments.train attn 600
python -m experiments.train linattn 600
python -m experiments.train decayattn 600

# 2) see learning by eye (text samples)
python -m experiments.generate decayattn

# 3) recall probe (~30 min/family; 1500 steps for the paper numbers)
python -m experiments.probe_recall attn 1500
python -m experiments.probe_recall linattn 1500
python -m experiments.probe_recall decayattn 1500

# 4) heavier-load stress on trained checkpoints (~1 min)
python -m experiments.probe_stress attn
python -m experiments.probe_stress linattn
python -m experiments.probe_stress decayattn
```

`train.py` auto-downloads Tiny Shakespeare (~1.1 MB) on first run. Total cost:
zero GPU-hours, roughly 3 CPU-hours for a full comparison, or ~1 free
Google-Colab style session.

---

## Evidence & reproducibility

- Every headline number lives in a machine-saved log under `results/`
  (per-step loss + validation/recall checkpoints) — never a chat summary.
- The closed-form linear families were verified equal to their recurrence form
  (max abs diff ~1e-8) before being used (`docs/PROTOCOL.md`).
- The recall probe's two early versions were flawed; both flaws are disclosed
  with the fixed version (`docs/PROTOCOL.md`).
- Model checkpoints are intentionally excluded (~3.6 MB each, trivially
  regenerated). Reproduce from scratch with the commands above.
- Limitations are stated explicitly in `docs/PROBE_RESULTS.md`.

---

## Status & roadmap

This is the first tile of an ongoing, open portfolio. Planned next studies
(details in `AGENTS.md`):

1. Seed-robustness runs (kill the "was it luck?" objection).
2. A third memory family (LRU / state-space) added to both tests.
3. Recall probe at longer contexts than 192 tokens.
4. Independent audits of public claims, run the same careful way.

---

## License

MIT — see `LICENSE`. The Tiny Shakespeare dataset is MIT-licensed and
downloaded automatically by `train.py`.

---

*Directed by [Yuvraj Singh Bhadoria](https://github.com/YuvrajSinghBhadoria2);
executed with the assistance of an AI research agent. Authorship, environment,
and provenance are documented in `docs/PROVENANCE.md`.*
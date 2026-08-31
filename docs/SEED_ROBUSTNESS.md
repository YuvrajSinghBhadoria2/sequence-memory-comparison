# Tile 2: seed robustness (is the result luck?)

The two headline claims in `RESULTS.md` and `PROBE_RESULTS.md` come from a
single random seed (seed 0). This tile reruns both experiments with three
independent seeds (1, 2, 3) using the **identical frozen protocol** (same
model sizes, budgets, data, learning rate, eval split) — only the master
random seed changes, changing weight init, batching, probe instance
generation, and training order.

Nothing else changes. No hyperparameter is tuned after seeing results.

## Decision rules (frozen BEFORE the runs, 2026-08-31)

Success / failure is judged only by the criteria below, fixed in advance.
The evidence files named here are produced directly by the scripts.

### Language-modeling ranking (seed 0 baseline: decayattn 1.7925 < linattn 2.0035 < attn 2.2007)

- **Success:** in every seed s in {1, 2, 3}, the step-500 (600-step run)
  val-loss ranking is the same strict order
  `decayattn < linattn < attn`, with the decayattn gap over linattn and the
  linattn gap over attn both positive.
- **Failure:** any seed inverts or ties the ordering, or any gap is <= 0.0.

### Recall-probe result (seed 0: attn ~10-20x chance, constant-memory ~chance)

- **Success:** in every seed s in {1, 2, 3}, at the frozen 1500-step check:
  - `attn` final R8 accuracy is at least **3x the chance rate (1/82)** AND at
    least **3x** each of `linattn` and `decayattn` final R8;
  - `linattn` and `decayattn` final R8 are each `<= 0.05` (near chance).
- **Failure:** any seed has attn below 3x chance, or a constant-memory family
  reaches or exceeds attn.

## Runs and evidence

Commands (each run writes its own file; seed 0 files from the original study
are reused as the seed-0 column):

```
python -m experiments.train attn 600 1      # -> results/language_modeling/attn_600_seed1_log.json
python -m experiments.train attn 600 2
python -m experiments.train attn 600 3
# ... same for linattn, decayattn (9 LM runs total)

python -m experiments.probe_recall attn 1500 1  # -> results/recall_probe/probe3_attn_1500_seed1_log.json
python -m experiments.probe_recall attn 1500 2
python -m experiments.probe_recall attn 1500 3
# ... same for linattn, decayattn (9 probe runs total)
```

Seed 0 is `sys.argv`/seed 0 and keeps the original untagged filenames.

## Verdict table (filled in after the runs)

| Check | seed 0 | seed 1 | seed 2 | seed 3 | Rule met? |
|---|---|---|---|---|---|
| LM ranking decayattn < linattn < attn | yes | | | | |
| attn probe R8 >= 3x chance | yes | | | | |
| linattn/decayattn probe R8 <= 0.05 | yes | | | | |

## Honesty notes

- These are **repeatability-in-practice** checks, not mathematical proofs.
- If any rule fails, this document is kept as-is (the frozen rule) and the
  failure is reported plainly in `RESULTS.md`/`PROBE_RESULTS.md` — a real
  seed failure is a real finding, not something to hide or retune away.
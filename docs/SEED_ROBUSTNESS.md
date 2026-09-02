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

Note on paths: the scripts print seed-tagged files to `results/`; after the
runs they are collected into `results/language_modeling/` and
`results/recall_probe/` alongside the seed-0 files listed above. The content
is the machine-saved run logs, unmodified.

## Verdict table (filled in after the runs)

Verdict: **PASS — all frozen rules met on every seed (0-3).** Data below is
read from the machine-saved logs in `results/language_modeling/` and
`results/recall_probe/`.

| Check | seed 0 | seed 1 | seed 2 | seed 3 | Rule met? |
|---|---|---|---|---|---|
| LM ranking decayattn < linattn < attn (step-500 val loss) | 1.7925 < 2.0035 < 2.2007 | 1.8051 < 2.0022 < 2.2131 | 1.8081 < 1.9931 < 2.1698 | 1.7978 < 2.0147 < 2.1992 | yes, all seeds |
| attn probe R8 >= 3x chance (0.0366) | 0.117 | 0.121 | 0.129 | 0.098 | yes, all seeds |
| linattn probe R8 <= 0.05 | 0.019 | 0.016 | 0.016 | 0.004 | yes, all seeds |
| decayattn probe R8 <= 0.05 | 0.019 | 0.016 | 0.016 | 0.004 | yes, all seeds |
| attn R8 >= 3x each const-memory family | yes | yes | yes | yes | yes, all seeds |

## Honesty notes

- These are **repeatability-in-practice** checks, not mathematical proofs.
- If any rule fails, this document is kept as-is (the frozen rule) and the
  failure is reported plainly in `RESULTS.md`/`PROBE_RESULTS.md` — a real
  seed failure is a real finding, not something to hide or retune away.
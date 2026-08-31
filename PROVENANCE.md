# Provenance — where everything came from

## Data

- Tiny Shakespeare corpus, downloaded automatically from
  <https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt>
  (MIT-licensed, ~1.1 MB). Char-level tokenizer: 65 unique characters, split
  90/10 automatically inside `train.py`.

## Environment

- 2019 Intel MacBook Pro 16", i7-9750H, 16 GB RAM, no CUDA, no Metal
  acceleration in use.
- local venv: Python 3.11 + torch 2.2.2 + numpy 1.26.4 (CPU build).
- Everything run from the `projects/` workspace; results saved by `train.py`
  to `results/`.

## Who did what (authorship, stated honestly)

- **Directed by:** Yuvraj Singh Bhadoria (GitHub: `YuvrajSinghBhadoria2`) —
  question selection, decisions, claims, publishing authority.
- **Executed with the assistance of an AI research agent** (autonomous harness
  running under the human's direction): protocol design, implementation,
  runs, math verification, literature check, and this write-up were performed
  by the agent; the human reviewed and owns every claim.

## How results were preserved

- Machine-saved evidence (not chat claims): raw per-step logs in `results/`
  `*_600_log.json` are included here under the same `results/` path, and the
  recall-probe logs as `results/probe3_*_1500_log.json` with the stress
  transcript `results/probe3_stress.txt`.
- The recall probe runs (v3) executed 2026-08-31 07:21-10:17 IST on this
  machine (the overnight attempt was interrupted when the laptop slept; the
  runs were restarted in the morning and completed without further pauses).
- The closed-form-vs-recurrence verification number (max abs diff ~9.3e-9)
  was produced by a scripted numerical comparison on this machine.
- Checkpoints (`.pt`, ~3.6 MB each) are deliberately excluded; they are
  regenerated exactly by the commands in `REPRODUCE.md`.
- Robustness re-runs with seeds 1-2 exist in the originating initiative's `logs/`
  for independent confirmation of the family ordering.

## Known limitations of this record

- Character-level, 0.92M params, 600 steps, single dataset, single config.
- Validation runs used a fixed 40-batch sample of the held-out 10%.
- This is evidence that the *comparison* is clean and the *math* is verified;
  it is explicitly not evidence of any new mechanism (see `NOVELTY.md`).
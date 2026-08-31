# Reproduce — get the exact same numbers

## Environment

Tested on a 2019 Intel MacBook Pro (i7-9750H, 16 GB, no GPU).

- Python 3.11
- `torch==2.2.2`
- `numpy==1.26.4` (torch 2.2 pins numpy < 2)
- No other dependencies. `train.py` is self-contained; no CUDA needed.

```bash
python -m venv .venv311
source .venv311/bin/activate
pip install torch==2.2.2 numpy==1.26.4
```

`torch.set_num_threads(6)` is already set inside `train.py` for a 6-core CPU.

## Commands

```bash
# 0) tiny smoke test (a few seconds) - proves the environment works
python train.py attn 30

# 1) decisive runs (600 steps each, ~20-35 min per family on a laptop CPU)
python train.py attn 600
python train.py linattn 600
python train.py decayattn 600

# 2) sample text from the best family to see learning by eye
python generate.py decayattn

# 3) recall probe (fully synthetic, no downloads; per family, ~30 min on CPU)
python probe_recall.py attn 600
python probe_recall.py linattn 600
python probe_recall.py decayattn 600
```

First run downloads Tiny Shakespeare (~1.1 MB) from the raw char-rnn repo into
`data/` automatically.

## What you should see

- Three `results/<family>_600_log.json` files.
- Validation loss at step 500: `decayattn` ~1.79 < `linattn` ~2.00 < `attn`
  ~2.20 (see `RESULTS.md`).
- `generate.py decayattn` prints visibly Shakespeare-shaped text; an untouched
  random-init checkpoint prints noise.

## Costs

Zero GPU, zero paid compute; runs on a normal laptop CPU. Roughly 1.5 CPU-hours
total for a full 3×600 comparison, or ~30-50 free colab-style runtime minutes
on any free CPU.
"""Heavier-load stress test on already-trained recall checkpoints.

Models trained with load R in {4..8}; this evaluates them with NO retraining at
heavier loads R in {8..16}. Tests whether constant-memory families keep exact
recall when the one-shot memory load exceeds what they saw in training.

Usage: python probe_stress.py <family>
"""
import sys

import torch

from probe_recall import BATCH, EVAL_N, VOCAB, eval_acc, make_batch
from train import Model


def main():
    fam = sys.argv[1]
    ckpt = torch.load(f"results/probe3_{fam}_1500.pt", map_location="cpu")
    model = Model(fam, VOCAB)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[{fam}] params {ckpt['params']/1e6:.2f}M", flush=True)
    with torch.no_grad():
        for R in (8, 10, 12, 14, 16):
            acc, near, far = eval_acc(model, R, EVAL_N, slots=True)
            print(f"[{fam}] R{R:>2} acc {acc:.4f} near {near:.4f} far {far:.4f}",
                  flush=True)


if __name__ == "__main__":
    main()
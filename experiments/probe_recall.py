"""Recall probe v3: single-query associative recall, laptop-learnable.

Version history (honest):
- v1: values were a FIXED function of keys (value = key + 64). Models could
  score 100% with a lookup table, no context storage. Flawed.
- v2: random per-instance pairing with 64 keys / 512-value pool. No lookup
  shortcut, but too hard for a 0.94M model in 600 steps (attn stayed at
  chance). Also had a near/far logging bug (compared key IDs, not positions).
- v3 (this file): same true-memory requirement (random per-instance pairing,
  keys repeat across instances) but small enough pools (16 keys / 64 values)
  that the task is learnable at laptop scale. Distance logging fixed to use
  actual position of the queried pairing.

Usage: python probe_recall.py <family> <steps> [seed]
Protocol frozen in repository/PROTOCOL.md (recall section).
"""
import json
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

from experiments.train import Model

KEY_N = 16                  # key tokens 0..15
VAL_N = 64                  # value pool 16..79
NOISE = KEY_N + VAL_N       # 80 filler token
MARKER = NOISE + 1          # 81 "?" probe separator
VOCAB = MARKER + 1          # 82
SEQ = 192
BATCH = 64
LR = 3e-4
EVAL_N = 256


def make_instance(R):
    seq = [NOISE] * SEQ
    keys = random.sample(range(KEY_N), R)
    vals = random.sample(range(KEY_N, KEY_N + VAL_N), R)
    pairs = list(zip(keys, vals))
    random.shuffle(pairs)
    slots = random.sample(range(SEQ - 2), len(pairs))
    for s, (k, v) in zip(slots, pairs):
        seq[s] = k
        seq[s + 1] = v
    qk, qv = random.choice(pairs)
    qi = pairs.index((qk, qv))
    q_slot = slots[qi]
    seq[-2] = qk
    seq[-1] = MARKER
    return torch.tensor(seq), torch.tensor(qv), q_slot


def make_batch(R, bs):
    xs, ys, slots = [], [], []
    for _ in range(bs):
        x, y, s = make_instance(R)
        xs.append(x)
        ys.append(y)
        slots.append(s)
    return torch.stack(xs), torch.tensor(ys), torch.tensor(slots)


def eval_acc(model, R, n, slots=False):
    model.eval()
    correct, near_c, far_c = 0, 0, 0
    near_n, far_n = 0, 0
    with torch.no_grad():
        for _ in range(n // BATCH):
            xs, ys, ss = make_batch(R, BATCH)
            pred = model(xs)[:, -1].argmax(-1)
            ok = pred == ys
            correct += ok.sum().item()
            if slots:
                near = ss >= SEQ // 2
                far = ~near
                near_c += (ok & near).sum().item()
                far_c += (ok & far).sum().item()
                near_n += near.sum().item()
                far_n += far.sum().item()
    acc = correct / n
    near = near_c / max(near_n, 1) if slots else 0.0
    far = far_c / max(far_n, 1) if slots else 0.0
    return acc, near, far


def main():
    torch.set_num_threads(6)
    fam = sys.argv[1]
    steps = int(sys.argv[2])
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    torch.manual_seed(seed)
    random.seed(seed)
    model = Model(fam, VOCAB)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    log = []
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        R = random.randint(4, 8)
        xs, ys, _ = make_batch(R, BATCH)
        logits = model(xs)
        loss = F.cross_entropy(logits[:, -1], ys)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        row = {"step": step, "loss": loss.item(),
               "elapsed_s": round(time.perf_counter() - t0, 1)}
        if step % 100 == 0:
            with torch.no_grad():
                for Rv in (4, 6, 8):
                    acc, near, far = eval_acc(model, Rv, EVAL_N, slots=True)
                    row[f"acc_R{Rv}"] = round(acc, 4)
                    row[f"near_R{Rv}"] = round(near, 4)
                    row[f"far_R{Rv}"] = round(far, 4)
            rate = step * BATCH * SEQ / (time.perf_counter() - t0) / 1e6
            print(f"[{fam}] step {step}/{steps} loss {loss.item():.4f} "
                  f"acc R4 {row['acc_R4']:.3f} R6 {row['acc_R6']:.3f} "
                  f"R8 {row['acc_R8']:.3f} rate {rate:.2f}M "
                  f"{time.perf_counter()-t0:.0f}s", flush=True)
        log.append(row)
    os.makedirs("results", exist_ok=True)
    torch.save({"model": model.state_dict(), "family": fam, "params": sum(
        p.numel() for p in model.parameters())}, f"results/probe3_{fam}_{steps}.pt")
    with open(f"results/probe3_{fam}_{steps}_log.json", "w") as f:
        json.dump(log, f)
    print(f"[{fam}] probe v3 done: {sum(p.numel() for p in model.parameters())/1e6:.2f}M "
          f"final R8 acc {row.get('acc_R8', 'n/a')}", flush=True)


if __name__ == "__main__":
    main()
"""Controlled family comparison, char-level, CPU-only.

Families: transformer (softmax attention), linear-attention (cache-friendly),
LRU (linear recurrent unit, constant-memory state). Frozen protocol in PLAN.md.
Usage: python train.py <family> <steps> [seed]
"""
import json
import math
import os
import random
import sys
import time
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "data/tinyshakespeare.txt"
LR = 3e-4
BATCH = 64
SEQ = 192
EMBED = 160
N_LAYER = 3
N_HEAD = 8
HEAD_DIM = EMBED // N_HEAD  # 20
MLP = 576
DROP = 0.1
CLIP = 1.0
EVAL_EVERY = 500
EVAL_STEPS = 40


def load_data():
    if not os.path.exists(DATA_PATH):
        os.makedirs("data", exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    text = open(DATA_PATH, encoding="utf-8").read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(ids))
    return ids[:n], ids[n:], len(chars)


def batches(data, bs, seq):
    n = len(data) // (bs * seq) * (bs * seq)
    t = data[:n].view(bs, -1)
    for i in range(0, t.size(1) - seq, seq):
        yield t[:, i:i + seq], t[:, i + 1:i + seq + 1]


class CausalScan(nn.Module):
    """Sequential scan over time, vectorized over batch. Correct, not fast."""

    def forward(self, step_closure, init_state, seq_len):
        pass  # concrete in each family


class SelfAttention(nn.Module):
    def __init__(self, emb, n_head, head_dim):
        super().__init__()
        self.n_head, self.head_dim = n_head, head_dim
        self.qkv = nn.Linear(emb, 3 * emb, bias=False)
        self.proj = nn.Linear(emb, emb, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q = q * (self.head_dim ** -0.5)
        att = torch.matmul(q, k.transpose(-1, -2))
        att = att.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1), float("-inf"))
        att = F.softmax(att, dim=-1)
        y = torch.matmul(att, v).transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class LinearAttention(nn.Module):
    """Fixed-decay linear attention, closed form: o_t = sum_{i<=t} d^(t-i) (q_t.k_i) v_i."""

    def __init__(self, emb, head_dim, decay=0.9):
        super().__init__()
        self.head_dim = head_dim
        self.n_head = emb // head_dim
        self.log_d = math.log(decay)  # fixed global decay
        self.qkv = nn.Linear(emb, 3 * emb, bias=False)
        self.proj = nn.Linear(emb, emb, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        nh, hd = self.n_head, self.head_dim
        q, k, v = self.qkv(x).reshape(B, T, 3, nh, hd).permute(2, 0, 3, 1, 4)
        q = q * (hd ** -0.5)
        t = torch.arange(T, device=x.device).unsqueeze(1)
        i = torch.arange(T, device=x.device).unsqueeze(0)
        D = torch.tril(torch.exp(self.log_d * torch.clamp(t - i, min=0)))  # T,T
        scores = torch.matmul(q, k.transpose(-1, -2)) * D.unsqueeze(0).unsqueeze(0)
        y = torch.matmul(scores, v).transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class LearnedDecayAttention(nn.Module):
    """LPD: per-token learned decay. Closed form:
    o_t = sum_{i<=t} (prod_{j=i+1..t} b_j) (q_t.k_i) v_i,  b_j = sigmoid(learned(x_j)).
    Subsumes attention (b->1) and fixed-decay linear attention (b constant)."""

    def __init__(self, emb, head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.n_head = emb // head_dim
        self.qkv = nn.Linear(emb, 3 * emb, bias=False)
        self.decay_head = nn.Linear(emb, 1)
        self.decay_head.weight.data.mul_(0.01)
        self.decay_head.bias.data.fill_(1.5)  # start near attention-like (b ~ .82)
        self.proj = nn.Linear(emb, emb, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        nh, hd = self.n_head, self.head_dim
        q, k, v = self.qkv(x).reshape(B, T, 3, nh, hd).permute(2, 0, 3, 1, 4)
        q = q * (hd ** -0.5)
        b = torch.sigmoid(self.decay_head(x)).squeeze(-1)          # (B,T)
        log_b = torch.log(b + 1e-8)
        cum = torch.cumsum(log_b, dim=1).unsqueeze(2)            # (B,T,1)
        logD = cum - cum.transpose(1, 2)                           # (B,Tt,Ti): C[t]-C[i]
        logD = logD.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool,
                                                      device=x.device), diagonal=1), -1e9)
        D = torch.exp(logD)                                        # diag = 1
        scores = torch.matmul(q, k.transpose(-1, -2)) * D.unsqueeze(1)
        y = torch.matmul(scores, v).transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class LRU(nn.Module):
    """Linear recurrent unit (SSM stand-in): constant-memory diagonal recurrence."""

    def __init__(self, emb, dim=96):
        super().__init__()
        self.dim = dim
        self.log_nu = nn.Parameter(torch.log(0.9 * torch.rand(dim) + 0.05))
        self.register_buffer("theta", torch.linspace(0.1, 1.5, dim))
        self.B = nn.Linear(emb, 2 * dim, bias=False)
        self.C = nn.Linear(2 * dim, emb, bias=False)
        self.nonlin = nn.GELU()

    def forward(self, x):
        B, T, C = x.shape
        lamb = torch.exp(-torch.exp(self.log_nu)) * torch.exp(1j * self.theta)
        bu = self.B(x)  # B,T,2d
        bu = (bu[:, :, :self.dim] + 1j * bu[:, :, self.dim:])
        state = torch.zeros(B, self.dim, dtype=torch.complex64, device=x.device)
        ys = []
        for t in range(T):
            state = lamb * state + bu[:, t]
            ys.append(state)
        y = torch.stack(ys, dim=1).contiguous()
        cu = torch.view_as_real(y).reshape(B, T, 2 * self.dim)
        return self.C(cu)


class Block(nn.Module):
    def __init__(self, fam, emb, n_head, head_dim, mlp):
        super().__init__()
        self.n1 = nn.LayerNorm(emb)
        if fam == "attn":
            self.mix = SelfAttention(emb, n_head, head_dim)
        elif fam == "linattn":
            self.mix = LinearAttention(emb, head_dim)
        elif fam == "decayattn":
            self.mix = LearnedDecayAttention(emb, head_dim)
        elif fam == "lru":
            self.mix = LRU(emb)
        self.n2 = nn.LayerNorm(emb)
        self.mlp = nn.Sequential(nn.Linear(emb, mlp), nn.GELU(), nn.Linear(mlp, emb))

    def forward(self, x):
        x = x + self.mix(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x


class Model(nn.Module):
    def __init__(self, fam, vocab, emb=EMBED, n_layer=N_LAYER, n_head=N_HEAD,
                 head_dim=HEAD_DIM, mlp=MLP):
        super().__init__()
        self.tok = nn.Embedding(vocab, emb)
        self.pos = nn.Parameter(torch.zeros(1, SEQ, emb))
        self.blocks = nn.ModuleList([Block(fam, emb, n_head, head_dim, mlp) for _ in range(n_layer)])
        self.nf = nn.LayerNorm(emb)
        self.out = nn.Linear(emb, vocab, bias=False)
        self.apply(self._init)
        self.out.weight.data.zero_()

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, x):
        e = self.tok(x) + self.pos
        for b in self.blocks:
            e = b(e)
        return self.out(self.nf(e))


def main():
    torch.set_num_threads(6)
    fam = sys.argv[1]
    steps = int(sys.argv[2])
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    torch.manual_seed(seed)
    random.seed(seed)
    train, val, vocab = load_data()
    model = Model(fam, vocab)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    it = batches(train, BATCH, SEQ)
    t0 = time.perf_counter()
    last_log = 0.0
    log = []
    for step in range(1, steps + 1):
        xb, yb = next(it, (None, None))
        if xb is None:
            it = batches(train, BATCH, SEQ)
            xb, yb = next(it)
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, vocab), yb.reshape(-1))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        log.append({"step": step, "loss": loss.item(),
                    "elapsed_s": round(time.perf_counter() - t0, 1)})
        if step % 50 == 0:
            rate = (step * BATCH * SEQ) / (time.perf_counter() - t0) / 1e6
            print(f"[{fam}] step {step}/{steps} loss {loss.item():.4f} "
                  f"rate {rate:.2f}M tok/s {time.perf_counter()-t0:.0f}s", flush=True)
        if step % EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                vloss = 0.0
                n = 0
                for xb, yb in batches(val, BATCH, SEQ):
                    if n >= EVAL_STEPS:
                        break
                    vloss += F.cross_entropy(model(xb).reshape(-1, vocab), yb.reshape(-1)).item()
                    n += 1
                vloss /= max(n, 1)
            model.train()
            log[-1]["val_loss"] = vloss
            rate = (step * BATCH * SEQ) / (time.perf_counter() - t0) / 1e6
            print(f"[{fam}] step {step}/{steps} loss {loss.item():.4f} val {vloss:.4f} "
                  f"rate {rate:.2f}M tok/s mem {n_params/1e6:.1f}M", flush=True)
    os.makedirs("results", exist_ok=True)
    torch.save({"model": model.state_dict(), "families": fam, "params": n_params},
               f"results/{fam}_{steps}.pt")
    os.makedirs("results", exist_ok=True)
    with open(f"results/{fam}_{steps}_log.json", "w") as f:
        json.dump(log, f)
    print(f"[{fam}] done: params {n_params/1e6:.2f}M final loss {log[-1]['loss']:.4f} saved", flush=True)


if __name__ == "__main__":
    main()
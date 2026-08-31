"""Plot two publication-style figures from the raw result logs.

Figures written to results/figures/:
- inversion_bars.png : the two-metric inversion, side by side
- recall_curves.png  : recall accuracy vs training step, all three families

Usage: python -m experiments.plots
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LG = "results/language_modeling"
RP = "results/recall_probe"
OUT = "results/figures"
CHANCE = 1 / 82

FAMS = [("attn", "softmax attention", "#d1495b"),
        ("linattn", "fixed-decay linear", "#8a8a8a"),
        ("decayattn", "learned decay", "#8a8a8a")]

os.makedirs(OUT, exist_ok=True)

vals = {f: next(r for r in json.load(open(f"{LG}/{f}_600_log.json"))
                if r["step"] == 500)["val_loss"] for f, _, _ in FAMS}

probe = {}
curves = {}
for f, _, _ in FAMS:
    rows = json.load(open(f"{RP}/probe3_{f}_1500_log.json"))
    probe[f] = (rows[-1]["acc_R4"], rows[-1]["acc_R8"])
    curves[f] = [(r["step"], r.get("acc_R8")) for r in rows if "acc_R8" in r]

plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13})

# --- Panel 1: the inversion, two bars -------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), dpi=170)
xs = range(3)
for ax, key, lab, flip in [
    (axes[0], "vals", "held-out val loss @500  (lower = better)", False),
    (axes[1], "recall", "one-shot recall acc @1500  (higher = better)", True),
]:
    data = [(-vals[f] if flip else float(vals[f])) for f, _, _ in FAMS] \
        if key == "vals" else [probe[f][1] for f, _, _ in FAMS]
    if flip:
        data = [float(vals[f]) for f, _, _ in FAMS]
    color = ["#d1495b" if f == "attn" else "#8a8a8a" for f, _, _ in FAMS]
    bars = ax.bar(xs, data, width=0.62, color=color)
    for x, d in zip(xs, data):
        ax.text(x, d, f"{d:.4f}" if key == "vals" else f"{d:.0%}",
                ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([l for _, l, _ in FAMS], rotation=12, fontsize=11)
    ax.set_ylabel(lab)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_title("Prediction")
axes[1].set_title("Recall")
axes[1].axhline(CHANCE, color="#555555", ls="--", lw=1.2)
axes[1].text(2.35, CHANCE, f"chance {CHANCE:.1%}", color="#555555",
             fontsize=10, ha="right", va="bottom")
fig.suptitle("Same models, same budget: the ranking inverts between tasks",
             y=1.02, fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/inversion_bars.png", bbox_inches="tight")
plt.close(fig)

# --- Panel 2: recall learning curves ---------------------------------
fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=170)
for f, lab, c in FAMS:
    steps = [s for s, _ in curves[f]]
    accs = [a if a is not None else 0.0 for _, a in curves[f]]
    ax.plot(steps, accs, lw=2.4, color=c, label=lab, marker="o", ms=3)
ax.axhline(CHANCE, color="#555555", ls="--", lw=1.2, label="chance 1.2%")
ax.set_xlabel("training step")
ax.set_ylabel("recall accuracy (8 keys)")
ax.set_title("Attention learns to recall; constant-memory families do not")
ax.legend(frameon=False, fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(f"{OUT}/recall_curves.png", bbox_inches="tight")
plt.close(fig)

print("wrote:", os.path.abspath(f"{OUT}/inversion_bars.png"))
print("wrote:", os.path.abspath(f"{OUT}/recall_curves.png"))
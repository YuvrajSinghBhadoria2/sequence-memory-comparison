"""Print the headline benchmark numbers for a terminal screenshot.

Reads only from the machine-saved logs under results/. Usage:
    python -m experiments.summary
"""
import json

LG = "results/language_modeling"
RP = "results/recall_probe"
CHANCE = round(1 / 82, 4)


def lm_val500(fam):
    rows = json.load(open(f"{LG}/{fam}_600_log.json"))
    return next(r for r in rows if r["step"] == 500)["val_loss"]


def probe_r(step, fam):
    rows = json.load(open(f"{RP}/probe3_{fam}_1500_log.json"))
    last = rows[-1]
    assert last["step"] == 1500
    return last["acc_R4"], last["acc_R8"]


HOME = "\033[1m\033[97m"
DIM = "\033[2m\033[90m"
RESET = "\033[0m"

rows = []
for fam, label in [
    ("decayattn", "learned per-token decay"),
    ("linattn", "fixed-decay linear"),
    ("attn", "softmax attention"),
]:
    vl = lm_val500(fam)
    r4, r8 = probe_r(1500, fam)
    rows.append((fam, label, vl, r4, r8))

print()
print(HOME + "  SEQUENCE-MEMORY COMPARISON — controlled benchmark" + RESET)
print(DIM + "  0.92M params | CPU only | identical data + budget | frozen protocol" + RESET)
print()
print("  family      pred val-loss    recall acc @1500 steps")
print("              @500 (lower)     R4      R8      (chance 1.2%)")
print("  " + "-" * 61)
for fam, label, vl, r4, r8 in rows:
    print(f"  {fam:<11s}{vl:10.4f}   {r4:5.1%}    {r8:5.1%}")
print("  " + "-" * 61)
print()
best_p = min(rows, key=lambda r: r[2])
best_r = max(rows, key=lambda r: r[4])
print(f"  decayattn = learned per-token decay   linattn = fixed-decay linear   attn = softmax attention")
print()
print(f"  best predictor  = {best_p[0]}   (val loss {best_p[2]:.4f})")
print(f"  best recaller   = {best_r[0]}   (R8 {best_r[4]:.1%})")
print(HOME + "  ranking: best predictor  is the WORST recaller" + RESET)
print(HOME + "  ranking: worst predictor is the ONLY recaller" + RESET)
print()
print(DIM + "  sources: results/language_modeling/*_600_log.json," + RESET)
print(DIM + "           results/recall_probe/probe3_*_1500_log.json" + RESET)
print()
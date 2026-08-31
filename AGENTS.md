# AGENTS.md — continuation guidance

## What this repo is

A clean, controlled, honest comparison of three context-memory mechanisms at
0.92M params on CPU. The per-token-decay winner is **not novel** (see
`NOVELTY.md`); do not ship any claim that it is. The portable value is the
fair comparison protocol, the verified closed-form implementation, and the
worked example of checking novelty before claiming.

## Reproduce first, then change

1. Set up per `REPRODUCE.md`, run at least the 600-step comparison once, and
   confirm you can reproduce the ordering to within ~0.01 val-loss.
2. Re-verify the closed-form equivalence (closed form vs naive scan, expect
   ~1e-9 max abs diff) after any edit to `linattn`/`decayattn`.
3. Nothing about the headline claim changes without an explicit note in
   `RESULTS.md` and `PROVENANCE.md`.

## Highest-value next experiments (cheap, decisive)

1. **Same-family ablation:** does learned per-token decay help because of
   content-dependence, or just because it adds a second learnable gate? Test a
   single-scalar-learned-decay version (one shared learned decay per layer) vs
   `decayattn`. This isolates *what* is helping.
2. **Memory/quality trade-off:** extend the protocol to measure decode-time
   memory (state size) per family and plot the Pareto frontier with 2-3 more
   budget points.
3. **Recall probe:** the open gap in the literature (per `NOVELTY.md`) is
   constant-memory models failing long-context recall. Status: **in flight
   (v3, 1500 steps/family)** — see `PROTOCOL.md` for the three-version history
   (v1 lookup flaw, v2 too-hard + distance-logging bug, v3 final). Update
   `RESULTS.md` with accuracy per family/R/distance after the runs, then run
   `probe_stress.py` on the trained checkpoints for heavier loads.

## Rules

- Preserve raw logs and seeds; freeze decision metrics before running.
- Never weaken a claim into a "discovery." If a result contradicts published
  work, record and explain it honestly.
- Reuse this protocol for any new family added to the comparison.
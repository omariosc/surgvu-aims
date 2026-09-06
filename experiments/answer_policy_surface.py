"""
SurgVU 2026 Cat-2 — ANSWER-POLICY PAYOFF SURFACE.

Decides the shipping decoding policy, and tells us how much vision accuracy is
actually worth under the 2026 ranked metric (BERTScore-F1, max-over-5-refs).

The metric's reference set is a verbosity ladder whose refs[0] is ALWAYS the bare
answer token, so:
  * a CORRECT bare token is an exact string match -> 1.0, the global maximum;
  * but a WRONG bare token has no scaffolding to fall back on, whereas a wrong
    VERBOSE answer still matches the reference carrier words.
=> terse is high-mean/high-variance, verbose is low-mean/low-variance, and there is
   a CROSSOVER in p(correct). Which policy to ship depends on where that crossover
   sits relative to our achievable accuracy. That is what this measures.

For each question we build a 3 (length register) x 2 (correct / wrong) grid and
then compute expected score  E[policy](p) = p*correct + (1-p)*wrong  over p in [0,1].

NOTE ON SCALE (from bert_score source): rescale_with_baseline is a monotone AFFINE
map, raw -> (raw - 0.83122575)/(1 - 0.83122575) at roberta-large layer 17. It cannot
change any ranking, but it multiplies every delta by ~5.93x. Rescaled deltas therefore
look ~6x larger than raw-BERTScore intuition suggests -- for both real gains and noise.
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import surgvu26_scorer as S

OUT = os.environ.get(
    "OUT", "/scratch/sc20osc/miccai-2026/SurgVU/models/answer_policy_surface.json")

# Per-case ground-truth short answer and a plausible WRONG alternative of the same
# lexical type. Wrong answers are deliberately *plausible* (the kind of error a real
# model makes), not random text.
CASES = {
    # case: (bare_correct, bare_wrong, sent_correct, sent_wrong)
    "case122": ("No", "Yes", "No forceps are being used.", "Yes, forceps are being used."),
    "case123": ("No", "Yes", "No, a large needle driver is not listed.",
                "Yes, a large needle driver is listed."),
    "case124": ("Cadiere Forceps", "Bipolar Forceps",
                "The forceps type is Cadiere Forceps.", "The forceps type is Bipolar Forceps."),
    "case125": ("Yes", "No", "Yes, sutures are required.", "No, sutures are not required."),
    "case126": ("Yes", "No", "Yes, a large needle driver was utilized.",
                "No, a large needle driver was not utilized."),
    "case127": ("Uterine horn", "Sigmoid colon",
                "The organ being manipulated is the uterine horn.",
                "The organ being manipulated is the sigmoid colon."),
    "case128": ("Yes", "No", "Yes, a needle driver is involved.",
                "No, a needle driver is not involved."),
    "case129": ("Endoscopic surgery", "Open surgery",
                "The summary is describing endoscopic or laparoscopic surgery.",
                "The summary is describing open abdominal surgery."),
    "case130": ("Grasping tissue", "Cutting tissue",
                "The forceps are used for grasping and holding tissues or objects.",
                "The forceps are used for cutting and dividing tissues or vessels."),
    "case131": ("Yes", "No", "Yes, tissue is being cut.", "No, tissue is not being cut."),
    "case132": ("No", "Yes", "No, a large needle driver was not used.",
                "Yes, a large needle driver was used."),
}

VERBOSE_TAIL = (" This is clearly observable in the surgical video clip provided, "
                "where the operating surgeon performs the described manoeuvre.")


def main():
    samples = S.load_samples()
    samples = [s for s in samples if s["case"] in CASES]
    refs_list = [s["refs"] for s in samples]
    cases = [s["case"] for s in samples]

    policies = {}
    for corr in ("correct", "wrong"):
        i_bare = 0 if corr == "correct" else 1
        i_sent = 2 if corr == "correct" else 3
        policies[f"bare_{corr}"] = [CASES[c][i_bare] for c in cases]
        policies[f"sentence_{corr}"] = [CASES[c][i_sent] for c in cases]
        policies[f"verbose_{corr}"] = [CASES[c][i_sent].rstrip(".") + "." + VERBOSE_TAIL
                                       for c in cases]

    scored = {}
    for name, cands in policies.items():
        per_case, agg = S.score_pairs(cands, refs_list, with_nli=True)
        scored[name] = {
            "bertscore_f1": agg["bertscore_f1"],
            "nli_entailment": agg.get("nli_entailment"),
            "bleu_score": agg["bleu_score"],
            "mean_words": float(np.mean([len(c.split()) for c in cands])),
            "per_case": {c: p["bertscore_f1"] for c, p in zip(cases, per_case)},
        }
        print(f"{name:20s} BERTScore-F1={agg['bertscore_f1']:.4f}  "
              f"NLI={agg.get('nli_entailment', float('nan')):.4f}  "
              f"words={scored[name]['mean_words']:.1f}", flush=True)

    # ---- expected score vs p(correct), and pairwise crossovers ----------
    registers = ["bare", "sentence", "verbose"]

    def E(reg, p):
        return p * scored[f"{reg}_correct"]["bertscore_f1"] + \
               (1 - p) * scored[f"{reg}_wrong"]["bertscore_f1"]

    grid = {}
    for p in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        grid[f"{p:.1f}"] = {r: E(r, p) for r in registers}

    print("\n=== EXPECTED BERTScore-F1 vs p(correct) ===")
    print(f"{'p':>5} " + "".join(f"{r:>12}" for r in registers) + "   best")
    for p, row in grid.items():
        best = max(row, key=row.get)
        print(f"{p:>5} " + "".join(f"{row[r]:12.4f}" for r in registers) + f"   {best}")

    crossovers = {}
    for a in registers:
        for b in registers:
            if a >= b:
                continue
            ca, wa = scored[f"{a}_correct"]["bertscore_f1"], scored[f"{a}_wrong"]["bertscore_f1"]
            cb, wb = scored[f"{b}_correct"]["bertscore_f1"], scored[f"{b}_wrong"]["bertscore_f1"]
            denom = (ca - wa) - (cb - wb)
            crossovers[f"{a}_vs_{b}"] = None if abs(denom) < 1e-9 else float((wb - wa) / denom)

    # value of being right, per register = the slope of E in p
    value_of_accuracy = {
        r: scored[f"{r}_correct"]["bertscore_f1"] - scored[f"{r}_wrong"]["bertscore_f1"]
        for r in registers
    }

    print("\n=== VALUE OF BEING CORRECT (slope of E wrt p) ===")
    for r, v in value_of_accuracy.items():
        print(f"  {r:10s} {v:+.4f}   (raw-BERTScore equivalent {v * (1 - 0.83122575):+.4f})")
    print("\n=== CROSSOVER p(correct) ===")
    for k, v in crossovers.items():
        print(f"  {k:22s} {v}")

    res = {"policies": scored, "expected_grid": grid,
           "crossovers": crossovers, "value_of_accuracy": value_of_accuracy}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

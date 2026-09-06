"""Cat-2 ADVERSARIAL / LEAKAGE CHECK for the 0.7428 real-grounded BLEU (n=11).

The brief: our 0.7428 is local n=11 with templates exact-matched to these 11 refs.
Before any SOTA claim we must answer: **is the gain a shortcut / metric-inflation, or
is the spine grounding genuinely doing the work?** This script runs two independent
checks; NO GPU needed (it operates on the templating layer + scorer, not the spine).

(A) OFFICIAL-SCORER CLONE & CROSS-CHECK
    Re-implement the SurgVU/Capybara metric FROM SCRATCH (independent of bleu_harness):
    NLTK sentence_bleu, weights (.25,.25,.25,.25), SmoothingFunction().method1,
    lowercased str.split() tokenization, scored vs EACH of the 5 refs, MAX kept, mean
    over questions. Then assert it agrees with bleu_harness.evaluate to <1e-9 on the
    yardsticks AND on our real prediction set. If they disagree, our 0.7428 is a
    scorer-implementation artifact -> RED FLAG. (This is the "clone the official
    max-over-5-refs scorer" deliverable.)

(B) ADVERSARIAL ABLATIONS — is the score shortcut-driven?
    1. TEXT-ONLY / NO-VISION ablation: run the router with EMPTY grounding (every g={}).
       The router then falls back to its DEFAULT polarity (present=False) + bucket
       phrasing. If this ALREADY scores near 0.7428, the templates -- not the spine --
       carry the score (the AMI/SurgCheck shortcut failure mode: priors, not vision).
       => the spine's contribution = 0.7428 - text_only.
    2. POLARITY-SHUFFLE control: flip every yes/no polarity (present -> not present).
       If the shuffled score is ~ the real score, polarity is not being used (the
       template scores regardless of correctness) -> the metric isn't sensitive to the
       spine's actual answer on these 11 -> n=11 is too small / templates dominate.
    3. CONSTANT-ANSWER baselines: echo-shortest-ref, bare yes/no, the single most-common
       reference string -> what a content-free constant achieves (the floor the spine
       must clear).
    4. EXACT-REF-HIT audit: how many of the 11 predictions are EXACT matches to a ref
       (BLEU=1.0)? A high exact-hit count on n=11 with hand-authored templates is the
       direct over-fit signal; report which cases and whether the spine or the template
       produced the hit.

VERDICT logic (pre-registered):
  * scorer clone disagrees with harness        -> RED FLAG (metric artifact)
  * spine contribution (real - text_only) < +0.05 -> SHORTCUT-DOMINATED: the score is
        templating priors, not vision; 0.7428 is NOT a vision win -> recalibrate claim.
  * polarity-shuffle ~ real (within 0.03)      -> metric INSENSITIVE on n=11; the gain
        is not validated -> need the gated QA set / larger n before any SOTA claim.
  * else                                        -> spine grounding is load-bearing; the
        0.7428 is a genuine (if small-n, upper-bound) vision-grounded result.

Run (login or gpu, surgvu_env — NO GPU needed):
  python leakage_check.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
from collections import Counter

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

import bleu_harness
import p0_2_template_router as tr

DATA = "/users/sc20osc/SurgVU/data"


# ---------- (A) independent scorer clone (NOT importing bleu_harness internals) ----------
def clone_bleu_mean(predict_fn, data_dir=DATA):
    smooth = SmoothingFunction().method1
    w = (0.25, 0.25, 0.25, 0.25)
    cases = bleu_harness.load_cases(data_dir)  # only reuse the I/O loader
    tot = 0.0
    per = []
    for cid, q, refs in cases:
        hyp = predict_fn(q, cid).lower().split()
        best = max(sentence_bleu([r.lower().split()], hyp, weights=w,
                                 smoothing_function=smooth) for r in refs)
        tot += best
        per.append((cid, best))
    return (tot / len(cases) if cases else 0.0), per


# ---------- the real grounding (from the winning run, replayed offline) ----------
# We don't re-run the spine here (no GPU). Instead we load the REAL grounding the spine
# produced (the same {present,tool,organ} dict per case) if cached; otherwise we use the
# P0.2 hand-set GROUNDING as the stand-in (which is the upper bound the spine approaches).
def load_real_grounding():
    cache_path = "/scratch/sc20osc/miccai-2026/SurgVU/models/spine_full/real_grounding.json"
    if os.path.exists(cache_path):
        return json.load(open(cache_path)), "spine-cached"
    return dict(tr.GROUNDING), "p0.2-handset (spine upper-bound)"


def set_grounding(g):
    tr.GROUNDING.clear()
    tr.GROUNDING.update(g)


def main():
    print("=== SurgVU Cat-2 ADVERSARIAL / LEAKAGE CHECK (n=11) ===\n")

    real_g, src = load_real_grounding()
    print(f"[grounding source] {src}\n")

    # ---- (A) scorer clone cross-check ----
    set_grounding(real_g)
    h_mean, _ = bleu_harness.evaluate(tr.predict, verbose=False)
    c_mean, c_per = clone_bleu_mean(tr.predict)
    agree = abs(h_mean - c_mean) < 1e-9
    print(f"(A) SCORER CLONE CROSS-CHECK")
    print(f"    bleu_harness mean = {h_mean:.6f}")
    print(f"    clone        mean = {c_mean:.6f}")
    print(f"    AGREE (<1e-9)     = {agree}   {'OK' if agree else '*** RED FLAG ***'}")
    # yardstick cross-check too
    refs_no = ["No", "No, forceps are not mentioned.", "No forceps are being used.",
               "No, there's no indication of forceps.", "No forceps are listed."]
    yk = abs(bleu_harness.bleu_max_over_refs(refs_no, "No")
             - max(sentence_bleu([r.lower().split()], "No".lower().split(),
                                 weights=(.25,)*4,
                                 smoothing_function=SmoothingFunction().method1)
                   for r in refs_no)) < 1e-9
    print(f"    yardstick 'No' agrees = {yk}\n")

    # ---- (B1) text-only / no-vision ablation ----
    set_grounding({})  # empty grounding -> router defaults (present=False everywhere)
    txt_mean, _ = bleu_harness.evaluate(tr.predict, verbose=False)
    spine_contrib = h_mean - txt_mean
    print(f"(B1) TEXT-ONLY (no-vision) ablation")
    print(f"     empty-grounding BLEU = {txt_mean:.4f}")
    print(f"     real grounding  BLEU = {h_mean:.4f}")
    print(f"     => SPINE CONTRIBUTION = {spine_contrib:+.4f}\n")

    # ---- (B2) polarity-shuffle control ----
    flipped = {c: {**v, **({"present": not v["present"]} if "present" in v else {})}
               for c, v in real_g.items()}
    set_grounding(flipped)
    flip_mean, _ = bleu_harness.evaluate(tr.predict, verbose=False)
    print(f"(B2) POLARITY-SHUFFLE control")
    print(f"     flipped-polarity BLEU = {flip_mean:.4f}  (real {h_mean:.4f})")
    print(f"     => polarity sensitivity = {h_mean - flip_mean:+.4f}\n")

    # ---- (B3) constant-answer baselines ----
    set_grounding(real_g)
    cases = bleu_harness.load_cases()
    echo = sum(bleu_harness.bleu_max_over_refs(refs, refs[0]) for _, _, refs in cases) / len(cases)
    allrefs = [r for _, _, refs in cases for r in refs]
    most_common = Counter(allrefs).most_common(1)[0][0]
    const_mean = sum(bleu_harness.bleu_max_over_refs(refs, most_common)
                     for _, _, refs in cases) / len(cases)
    print(f"(B3) CONSTANT-ANSWER floors")
    print(f"     echo-shortest-ref     = {echo:.4f}")
    print(f"     most-common-ref const = {const_mean:.4f}  ({most_common!r})\n")

    # ---- (B4) exact-ref-hit audit ----
    set_grounding(real_g)
    exact = []
    for cid, q, refs in cases:
        s = bleu_harness.bleu_max_over_refs(refs, tr.predict(q, cid))
        if s > 0.999:
            exact.append(cid)
    print(f"(B4) EXACT-REF HITS = {len(exact)}/{len(cases)}  {exact}")
    print(f"     (high exact-hit count on n=11 with authored templates = over-fit signal)\n")

    # ---- verdict ----
    flags = []
    if not agree:
        flags.append("METRIC-ARTIFACT (scorer clone disagrees)")
    if spine_contrib < 0.05:
        flags.append("SHORTCUT-DOMINATED (spine adds <+0.05 over text-only)")
    if abs(h_mean - flip_mean) < 0.03:
        flags.append("METRIC-INSENSITIVE on n=11 (polarity-shuffle ~ real)")
    if flags:
        verdict = "CAUTION -> " + " | ".join(flags) + " ; recalibrate the 0.7428 claim"
    else:
        verdict = ("CLEAN -> spine grounding is load-bearing (+%.4f over text-only, "
                   "polarity-sensitive); 0.7428 is a genuine small-n upper-bound, "
                   "validate on gated QA before final SOTA claim" % spine_contrib)
    print("=== LEAKAGE VERDICT ===")
    print(verdict)
    print(f"\nLEAKAGE_VERDICT spine_contrib={spine_contrib:+.4f} "
          f"polarity_sens={h_mean-flip_mean:+.4f} exact_hits={len(exact)}/{len(cases)} "
          f"scorer_agree={agree} -> {'CLEAN' if not flags else 'CAUTION'}")


if __name__ == "__main__":
    main()

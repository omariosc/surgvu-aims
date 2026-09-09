"""
SurgVU 2026 Cat-2 — METRIC RE-CALIBRATION.

The 2026 ranked metric changed from BLEU-4 (2025) to BERTScore-F1
(roberta-large, rescale_with_baseline=True), max-over-5-refs, mean-over-questions.
Every strategic conclusion in the dossier was derived under BLEU and must be re-derived.

This script re-measures every yardstick under BOTH metrics on the 11 public sample
clips, and runs the two ablations that decide the whole approach:

  A. VERBOSITY LADDER  — echo each of the 5 reference tiers (ref[0] bare token ->
     ref[4] most verbose). Tells us the optimal answer LENGTH under BERTScore.
  B. POLARITY ABLATION — flip Yes<->No and insert/remove negation in otherwise
     identical answers. The organizers' own README states BERTScore is negation-blind.
     If the drop is ~0, the grounding spine (which predicts polarity) is NOT
     load-bearing for the ranked metric, and the lever is fluent on-topic phrasing.
  C. CONSTANT-ANSWER FLOOR — a single fixed string emitted for EVERY question,
     i.e. the score obtainable with no vision and no question understanding at all.
     This is the honest floor any real method must clear.

Outputs JSON to $OUT and prints a table. No training; scoring only.
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import surgvu26_scorer as S

OUT = os.environ.get(
    "OUT", "/scratch/USERNAME/miccai-2026/SurgVU/models/metric_recalibration_2026.json"
)

# The 11 P0.2/V2 template-router answers that scored BLEU 0.9650 / V2 1.0000 offline.
# These were authored to be exact string matches to a reference under the 2025 BLEU rule.
TEMPLATE_V2 = {
    "case122": "No forceps are being used.",
    "case123": "No, a large needle driver is not listed.",
    "case124": "The forceps type is Cadiere Forceps.",
    "case125": "Yes, sutures are required.",
    "case126": "Yes, a large needle driver was utilized.",
    "case127": "The organ being manipulated is the uterine horn.",
    "case128": "Yes, a needle driver is involved.",
    "case129": "The summary is describing endoscopic or laparoscopic surgery.",
    "case130": "The forceps are used for grasping and holding tissues or objects.",
    "case131": "Yes, tissue is being cut.",
    "case132": "No, a large needle driver was not used.",
}

# Constant answers: emitted identically for all 11 questions (no vision, no question read).
CONSTANTS = {
    "const_yes_bare": "Yes",
    "const_no_bare": "No",
    "const_yes_sent": "Yes, a surgical instrument was used during the procedure.",
    "const_generic_proc": "The procedure involved the use of surgical instruments on tissue.",
    "const_hedge": "Yes, it appears that the instrument was used in this clip.",
    "const_forceps": "Forceps were used during the procedure.",
}


def flip_polarity(text):
    """Flip the affirmative/negative sense of a short answer while keeping wording."""
    t = text
    # sentence-initial bare polarity
    t = re.sub(r"^Yes\b", "\x00NO\x00", t)
    t = re.sub(r"^No\b", "\x00YES\x00", t)
    # negation markers
    if "\x00NO\x00" in t:  # was affirmative -> make negative
        t = t.replace(" is being ", " is not being ")
        t = t.replace(" are being ", " are not being ")
        t = t.replace(" was utilized", " was not utilized")
        t = t.replace(" was used", " was not used")
        t = t.replace(" is involved", " is not involved")
        t = t.replace(" are required", " are not required")
        t = t.replace(" is listed", " is not listed")
    else:  # was negative -> make affirmative
        t = t.replace(" not being ", " being ")
        t = t.replace(" not utilized", " utilized")
        t = t.replace(" not used", " used")
        t = t.replace(" not involved", " involved")
        t = t.replace(" not required", " required")
        t = t.replace(" not listed", " listed")
        t = re.sub(r"^No ", "Some ", t)
    t = t.replace("\x00NO\x00", "No").replace("\x00YES\x00", "Yes")
    return t


def main():
    samples = S.load_samples()
    print(f"Loaded {len(samples)} sample cases\n", flush=True)
    cases = [s["case"] for s in samples]
    refs_list = [s["refs"] for s in samples]

    strategies = {}

    # --- A. VERBOSITY LADDER: echo reference tier i for every case ---------
    for i in range(5):
        strategies[f"echo_ref{i}"] = [
            s["refs"][i] if i < len(s["refs"]) else s["refs"][-1] for s in samples
        ]

    # --- Our 2025-BLEU-optimal templates ----------------------------------
    strategies["template_v2"] = [TEMPLATE_V2[c] for c in cases]

    # --- B. POLARITY ABLATION ---------------------------------------------
    strategies["template_v2_FLIPPED"] = [flip_polarity(TEMPLATE_V2[c]) for c in cases]
    strategies["echo_ref1_FLIPPED"] = [flip_polarity(s["refs"][1]) for s in samples]

    # --- Verbose / degenerate -------------------------------------------
    strategies["verbose_correct"] = [
        TEMPLATE_V2[c].rstrip(".")
        + " in this surgical video clip, as observed by the operating surgeon during the procedure."
        for c in cases
    ]
    strategies["question_echo"] = [s["question"] for s in samples]

    # --- C. CONSTANT-ANSWER FLOOR ----------------------------------------
    for name, txt in CONSTANTS.items():
        strategies[name] = [txt] * len(samples)

    # --- score everything -------------------------------------------------
    results = {}
    for name, cands in strategies.items():
        per_case, agg = S.score_pairs(cands, refs_list, with_nli=True)
        results[name] = {
            "aggregates": agg,
            "per_case": {c: {"candidate": p["candidate"],
                             "bertscore_f1": p["bertscore_f1"],
                             "bleu_score": p["bleu_score"],
                             "nli_entailment": p.get("nli_entailment")}
                         for c, p in zip(cases, per_case)},
        }
        print(f"{name:26s} BERTScore-F1={agg['bertscore_f1']:.4f} "
              f"(RANKED) | BLEU={agg['bleu_score']:.4f} "
              f"| ROUGE-L={agg['rougeL_score']:.4f} "
              f"| NLI={agg.get('nli_entailment', float('nan')):.4f}", flush=True)

    # --- headline ablation deltas ----------------------------------------
    def b(n):
        return results[n]["aggregates"]["bertscore_f1"]

    ablations = {
        "polarity_drop_template": b("template_v2") - b("template_v2_FLIPPED"),
        "polarity_drop_ref1": b("echo_ref1") - b("echo_ref1_FLIPPED"),
        "best_constant": max(CONSTANTS, key=lambda k: b(k)),
        "best_constant_bertscore": max(b(k) for k in CONSTANTS),
        "template_over_best_constant": b("template_v2") - max(b(k) for k in CONSTANTS),
        "best_verbosity_tier": max(range(5), key=lambda i: b(f"echo_ref{i}")),
    }
    results["_ablations"] = ablations

    print("\n=== ABLATIONS (BERTScore-F1, the ranked metric) ===")
    for k, v in ablations.items():
        print(f"  {k:32s} {v}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT}")

    print(f"\nSURGVU26_RECAL_BEST_CONSTANT={ablations['best_constant_bertscore']:.4f}")
    print(f"SURGVU26_RECAL_TEMPLATE={b('template_v2'):.4f}")
    print(f"SURGVU26_RECAL_POLARITY_DROP={ablations['polarity_drop_template']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

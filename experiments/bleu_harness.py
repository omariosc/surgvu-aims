"""Faithful SurgVU Cat-2 BLEU harness (P0.1 hardened for P0.2).

The official SurgVU-2025 Cat-2 metric is sentence-level BLEU:
  - NLTK ``sentence_bleu`` with uniform 4-gram weights (0.25, 0.25, 0.25, 0.25)
  - ``SmoothingFunction().method1``
  - lowercased whitespace tokenization
  - scored against EACH of the 5 references INDIVIDUALLY, then the MAX is kept
  - the per-question max-BLEU is averaged (mean) over all questions.

This module wraps the *real* NLTK implementation (nltk 3.9.x is installed in the
vqa_env) rather than a hand-rolled copy, so the harness matches the official
metric exactly. Punkt is NOT required: tokenization is plain ``str.lower().split()``.

Validate before trusting: run ``python bleu_harness.py`` to print the yardstick
checks (bare "no" ~0.18, echo-shortest-ref ~0.35, exact-match = 1.0).
"""
import glob
import json
import os
from typing import List

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

_SMOOTH = SmoothingFunction().method1
_WEIGHTS = (0.25, 0.25, 0.25, 0.25)


def tokenize(text: str) -> List[str]:
    """Lowercased whitespace tokenization (the official metric does not strip
    punctuation, but it lowercases; punctuation stays glued to its token)."""
    return text.lower().split()


def bleu_one(reference: str, hypothesis: str) -> float:
    """BLEU of one hypothesis against a single reference (method1-smoothed)."""
    return sentence_bleu(
        [tokenize(reference)],
        tokenize(hypothesis),
        weights=_WEIGHTS,
        smoothing_function=_SMOOTH,
    )


def bleu_max_over_refs(references: List[str], hypothesis: str) -> float:
    """Score the hypothesis against each reference individually, keep the MAX."""
    return max(bleu_one(ref, hypothesis) for ref in references)


def load_cases(data_dir: str = "/users/USERNAME/SurgVU/data"):
    """Yield (case_id, question_str, [refs]) for each caseXXX dir."""
    cases = []
    for d in sorted(glob.glob(os.path.join(data_dir, "case*"))):
        c = os.path.basename(d)
        qf = os.path.join(d, f"{c}_question.json")
        af = os.path.join(d, f"{c}.json")
        if not (os.path.exists(qf) and os.path.exists(af)):
            continue
        with open(qf) as fh:
            question = json.load(fh)
        with open(af) as fh:
            refs = json.load(fh)
        cases.append((c, question, refs))
    return cases


def evaluate(predict_fn, data_dir: str = "/users/USERNAME/SurgVU/data", verbose: bool = True):
    """Run ``predict_fn(question, case_id) -> answer`` over every case and report
    per-question max-BLEU + the mean. Returns (mean, per_case list)."""
    cases = load_cases(data_dir)
    per_case = []
    total = 0.0
    for cid, q, refs in cases:
        pred = predict_fn(q, cid)
        score = bleu_max_over_refs(refs, pred)
        total += score
        per_case.append((cid, q, pred, score))
        if verbose:
            print(f"  {cid}: BLEU={score:.4f}  pred={pred!r}")
    mean = total / len(cases) if cases else 0.0
    if verbose:
        print(f"\n  MEAN max-BLEU over {len(cases)} cases = {mean:.4f}")
    return mean, per_case


def _validate():
    """Yardstick checks the spec demands before trusting the harness."""
    print("=== Harness validation (yardsticks) ===")
    refs_no = [
        "No",
        "No, forceps are not mentioned.",
        "No forceps are being used.",
        "No, there's no indication of forceps.",
        "No forceps are listed.",
    ]
    checks = {
        "bare 'no' (~0.18)": ("No", refs_no),
        "exact-match to a ref (=1.0)": ("No forceps are being used.", refs_no),
    }
    for label, (cand, refs) in checks.items():
        print(f"  {label:32s}: {bleu_max_over_refs(refs, cand):.4f}  | {cand!r}")

    # echo-shortest-ref over the 11 samples (~0.35)
    cases = load_cases()
    if cases:
        total = sum(bleu_max_over_refs(refs, refs[0]) for _, _, refs in cases)
        echo = total / len(cases)
        print(f"  echo-shortest-ref over {len(cases)} samples (~0.35): {echo:.4f}")
        bare_total = 0.0
        for _, _, refs in cases:
            yn = "Yes" if refs[0].lower().startswith("y") else "No"
            bare_total += bleu_max_over_refs(refs, yn)
        print(f"  bare yes/no over {len(cases)} samples (~0.12):        {bare_total/len(cases):.4f}")
        oracle = sum(max(bleu_max_over_refs(refs, r) for r in refs) for _, _, refs in cases) / len(cases)
        print(f"  oracle (best ref as pred, =1.0):                {oracle:.4f}")


if __name__ == "__main__":
    _validate()

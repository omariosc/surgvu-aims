"""
SurgVU 2026 Category-2 OFFICIAL SCORER CLONE.

Mirrors, line-for-line in behaviour, the organizers' `evaluation/evaluate.py` from
https://github.com/isi-challenges/surgvu26-category-2-eval-public

AUTHORITATIVE FACTS (verified from that repo, 2026-07-30):
  * PRIMARY / ONLY RANKED METRIC = BERTScore-F1
        BERTScorer(model_type="roberta-large", lang="en", rescale_with_baseline=True)
    README: "Only metric used for ranking and valid for challenge".
  * Per question: score = MAX over the 5 references. Final = MEAN over all questions.
  * ** BERTScore is computed on the RAW, UN-NORMALIZED candidate + references. **
    `normalize()` (lowercase + strip punctuation) is applied ONLY to the inputs of
    BLEU and ROUGE. This is a real asymmetry in the official code: in
    `process_interf0` the returned dict carries the RAW `candidate`/`references`,
    and `compute_heavy_metrics` consumes those raw strings.
    => case and punctuation DO matter for the ranked metric, and do NOT matter for BLEU.
  * Secondary (analysis only, NOT ranked): NLI entailment
    (cross-encoder/nli-deberta-v3-base), NLI*BERT, BLEU-4, ROUGE-1/2/L.

This module exposes `score_pairs()` so every SurgVU experiment scores against the
real 2026 objective instead of the retired 2025 BLEU objective.
"""

import os
import string

os.environ.setdefault("HF_HOME", "/scratch/sc20osc/hf_cache")

import numpy as np


# ---------------------------------------------------------------------------
# Text handling — identical to the official evaluate.py
# ---------------------------------------------------------------------------

def normalize(text):
    """Lowercase, strip punctuation. Official: feeds BLEU/ROUGE ONLY (not BERTScore)."""
    return text.translate(str.maketrans("", "", string.punctuation)).lower().strip()


def compute_bleu(candidate, references):
    """Max multi-reference BLEU-4, NLTK smoothing method1. Expects NORMALIZED text."""
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    cand_tokens = candidate.split()
    smoothing = SmoothingFunction().method1
    weights = (0.25, 0.25, 0.25, 0.25)
    scores = []
    for ref in references:
        scores.append(
            sentence_bleu([ref.split()], cand_tokens,
                          weights=weights, smoothing_function=smoothing)
        )
    return max(scores) if scores else 0.0


def compute_rouge(candidate, references):
    """Max multi-reference ROUGE-1/2/L F-measure. Expects NORMALIZED text."""
    from rouge_score import rouge_scorer as _rs

    scorer = _rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rL = [], [], []
    for ref in references:
        sc = scorer.score(ref, candidate)
        r1.append(sc["rouge1"].fmeasure)
        r2.append(sc["rouge2"].fmeasure)
        rL.append(sc["rougeL"].fmeasure)
    return {
        "rouge1": max(r1) if r1 else 0.0,
        "rouge2": max(r2) if r2 else 0.0,
        "rougeL": max(rL) if rL else 0.0,
    }


# ---------------------------------------------------------------------------
# Heavy models (loaded once, cached at module level)
# ---------------------------------------------------------------------------

_BERT = None
_NLI = None


def get_bert_scorer():
    global _BERT
    if _BERT is None:
        from bert_score import BERTScorer
        _BERT = BERTScorer(model_type="roberta-large", lang="en",
                           rescale_with_baseline=True)
    return _BERT


def get_nli():
    global _NLI
    if _NLI is None:
        from sentence_transformers import CrossEncoder
        _NLI = CrossEncoder("cross-encoder/nli-deberta-v3-base")
    return _NLI


def compute_bertscore(candidate, references, scorer=None):
    """PRIMARY METRIC. RAW (un-normalized) text, max over references."""
    scorer = scorer or get_bert_scorer()
    _P, _R, F1 = scorer.score([candidate] * len(references), list(references))
    return F1.max().item()


def compute_nli_entailment(candidate, references, model=None):
    model = model or get_nli()
    logits = model.predict([(candidate, r) for r in references])
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return float(probs[:, 1].max())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_one(candidate, references, with_nli=False):
    """Score one (candidate, 5-references) pair exactly as the official container does."""
    refs_clean = [normalize(r) for r in references]
    cand_clean = normalize(candidate)

    out = {
        "bertscore_f1": compute_bertscore(candidate, references),   # RAW  <- RANKED
        "bleu_score": compute_bleu(cand_clean, refs_clean),         # normalized
    }
    rouge = compute_rouge(cand_clean, refs_clean)
    out["rouge1_score"] = rouge["rouge1"]
    out["rouge2_score"] = rouge["rouge2"]
    out["rougeL_score"] = rouge["rougeL"]
    if with_nli:
        nli = compute_nli_entailment(candidate, references)
        out["nli_entailment"] = nli
        out["nli_bertscore_f1"] = nli * out["bertscore_f1"]
    return out


def score_pairs(candidates, references_list, with_nli=False):
    """Batched. Returns (per_case_list, aggregates_dict). Aggregate = MEAN, as official."""
    scorer = get_bert_scorer()
    nli_model = get_nli() if with_nli else None

    per_case = []
    for cand, refs in zip(candidates, references_list):
        refs_clean = [normalize(r) for r in refs]
        cand_clean = normalize(cand)
        rouge = compute_rouge(cand_clean, refs_clean)
        rec = {
            "candidate": cand,
            "bertscore_f1": compute_bertscore(cand, refs, scorer),
            "bleu_score": compute_bleu(cand_clean, refs_clean),
            "rouge1_score": rouge["rouge1"],
            "rouge2_score": rouge["rouge2"],
            "rougeL_score": rouge["rougeL"],
        }
        if with_nli:
            nli = compute_nli_entailment(cand, refs, nli_model)
            rec["nli_entailment"] = nli
            rec["nli_bertscore_f1"] = nli * rec["bertscore_f1"]
        per_case.append(rec)

    keys = [k for k in per_case[0] if k != "candidate"] if per_case else []
    agg = {k: float(np.mean([r[k] for r in per_case])) for k in keys}
    return per_case, agg


# ---------------------------------------------------------------------------
# Sample-set loader (the 11 public Cat-2 clips)
# ---------------------------------------------------------------------------

SAMPLE_DIR = "/scratch/sc20osc/miccai-2026/SurgVU/data"
SAMPLE_CASES = [f"case{n}" for n in range(122, 133)]


def load_samples(sample_dir=SAMPLE_DIR):
    """-> list of dicts {case, question, refs, video}."""
    import json
    from pathlib import Path

    out = []
    for c in SAMPLE_CASES:
        d = Path(sample_dir) / c
        if not d.is_dir():
            continue
        with open(d / f"{c}.json") as f:
            refs = json.load(f)
        with open(d / f"{c}_question.json") as f:
            q = json.load(f)
        out.append({"case": c, "question": q, "refs": refs,
                    "video": str(d / f"{c}.mp4")})
    return out

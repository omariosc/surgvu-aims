"""
SurgVU 2026 Cat-2 — VLM-AS-ENTITY-EXTRACTOR -> SYNTACTIC TEMPLATE.

WHAT THE MEASUREMENTS SO FAR ACTUALLY SAY (jobs 6933448 / 6933607 / 6980942)
---------------------------------------------------------------------------
Ranked metric = BERTScore-F1 (roberta-large, rescale_with_baseline, max over the 5
refs, mean over questions).  Established floors, all on the same n=11:

    constant "No", no vision, question unread ................ 0.6383
    bare token, Qwen3-VL-8B, 5 frames (best zero-shot, 6980942)  0.7236
    ~7-word declarative that is WRONG ........................ 0.7932
    ~7-word declarative that is RIGHT ........................ 1.0000*

So the BEST free-text zero-shot VLM configuration scores 0.070 BELOW a template
bank that gets every single answer wrong.  The VLM's problem is not knowledge, it
is REGISTER: prompted for reference style it emitted 13-15 words (D_refstyle), and
raw it emitted 45 (A_plain -> 0.21).  Length control cannot be bought by prompting.

=> This script tests the architecture that follows from that: the VLM never writes
   the answer.  It emits only a SHORT SPAN (a Yes/No polarity, or a <=4-word entity
   noun phrase), and a deterministic template writes the sentence.

THE TEMPLATE IS A PURELY SYNTACTIC FUNCTION OF THE QUESTION STRING.
------------------------------------------------------------------
This matters for rigor.  The 1.0000* and 0.7932 figures above came from templates
we AUTHORED against these same 11 references, so their register match is partly
leakage.  The templates here are produced by auxiliary-inversion / wh-fronting of
the question alone -- they never saw a reference.  Any register match they achieve
is therefore an honest, transferable property, not tuning.

THE CONTROL THAT ACTUALLY ANSWERS THE QUESTION
----------------------------------------------
`N0_template_novision` runs the identical template path with a FIXED polarity and
generic entities and never opens the video.  H_hybrid - N0_template_novision is the
value of vision inside the shipping register, which is the quantity the earlier
runs never isolated (they confounded register with correctness).

n=11.  Bootstrap CIs are reported and they are enormous.  Nothing here settles
anything on its own -- the preliminary leaderboard is the only real channel.
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", "/scratch/sc20osc/hf_cache")

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("cat2hybrid")

# ---------------------------------------------------------------------------
# 1. QUESTION ROUTER + SYNTACTIC TEMPLATES  (no reference knowledge whatsoever)
# ---------------------------------------------------------------------------

YESNO_AUX = {"is", "are", "was", "were", "does", "do", "did", "has", "have", "can", "could"}
# tokens that terminate the leading subject noun phrase
NP_END = {
    "being", "used", "using", "involved", "required", "cut", "mentioned", "listed",
    "among", "in", "on", "during", "performed", "present", "visible", "manipulated",
    "describing", "applied", "shown", "seen", "held", "grasped", "here", "this",
    "within", "at", "with", "by", "for", "to",
}
NEG = {"is": "is not", "are": "are not", "was": "was not", "were": "were not",
       "has": "has not", "have": "have not", "can": "cannot", "could": "could not",
       "does": "does not", "do": "do not", "did": "did not"}


def _toks(q):
    return q.strip().rstrip("?").split()


def classify(question):
    """-> 'yesno' | 'purpose' | 'entity'  (deterministic, question-string only)."""
    t = _toks(question)
    if not t:
        return "entity"
    first = t[0].lower()
    if first in YESNO_AUX:
        return "yesno"
    if "purpose" in question.lower() or question.lower().startswith("why"):
        return "purpose"
    return "entity"


def yesno_template(question, polarity):
    """'Is a suture required in this surgical step?' + Yes
         -> 'Yes, a suture is required in this surgical step.'
       Auxiliary inversion only. Never sees a reference."""
    t = _toks(question)
    aux = t[0].lower()
    rest_tokens = t[1:]
    existential = bool(rest_tokens) and rest_tokens[0].lower() == "there"
    if existential:
        rest_tokens = rest_tokens[1:]

    i = 0
    while i < len(rest_tokens) and rest_tokens[i].lower().strip(",") not in NP_END:
        i += 1
    subj = " ".join(rest_tokens[:i]).strip()
    tail = " ".join(rest_tokens[i:]).strip()
    if not subj:                                   # degenerate parse -> safe fallback
        subj, tail = " ".join(rest_tokens).strip(), ""

    verb = aux if polarity else NEG.get(aux, aux + " not")
    head = "Yes" if polarity else "No"
    body = " ".join(x for x in (subj, verb, tail) if x)
    return f"{head}, {body}."


def entity_template(question, entity):
    """wh-fronting.  'What organ is being manipulated?' + 'the bowel'
         -> 'The organ being manipulated is the bowel.'"""
    q = question.strip().rstrip("?")
    ql = q.lower()
    ent = entity.strip().rstrip(".")

    m = re.match(r"^what\s+(is|are|was|were)\s+(.+)$", ql)
    if m:                                          # 'What is the purpose of ...'
        return f"{m.group(2)[0].upper()}{m.group(2)[1:]} {m.group(1)} {ent}."

    m = re.match(r"^(what|which)\s+(.+?)\s+(is|are|was|were)\s+(.+)$", ql)
    if m:
        head, aux, rest = m.group(2), m.group(3), m.group(4)
        if "summary" in rest or "describing" in rest or "describe" in rest:
            return f"This summary is describing {ent}."
        return f"The {head} {rest} {aux} {ent}."

    return f"The answer is {ent}."


def purpose_template(question, span):
    """'What is the purpose of using forceps in this procedure?' + 'grasp and hold tissue'
         -> 'The purpose of using forceps is to grasp and hold tissue.'"""
    s = span.strip().rstrip(".")
    s = re.sub(r"^(it is used |used |it is |is )", "", s, flags=re.I).strip()
    if not s.lower().startswith("to "):
        s = "to " + s
    ql = question.strip().rstrip("?").lower()
    m = re.search(r"purpose of (.+)$", ql)
    if m:
        head = m.group(1).strip()
        # drop a trailing prepositional phrase so the sentence stays near the
        # measured 7-word reference median instead of running to 13
        head = re.split(r"\s+(?:in this|in the|during|within)\s+", head)[0].strip()
        return f"The purpose of {head} is {s}."
    return f"It is used {s}."


def compose(question, slot):
    """slot: bool for yesno; str for entity/purpose."""
    kind = classify(question)
    if kind == "yesno":
        return yesno_template(question, bool(slot))
    if kind == "purpose":
        return purpose_template(question, str(slot))
    return entity_template(question, str(slot))


# ---------------------------------------------------------------------------
# 2. VLM SHORT-SPAN EXTRACTION
# ---------------------------------------------------------------------------

EXTRACT_PROMPT = {
    "yesno": "{q}\n\nReply with exactly one word: Yes or No. Nothing else.",
    "entity": "{q}\n\nReply with ONLY the name, as a noun phrase of at most four words. "
              "No sentence, no explanation, no punctuation.",
    "purpose": "{q}\n\nReply with at most eight words describing the purpose, starting with 'to'. "
               "No sentence, no explanation.",
}
MAX_NEW = {"yesno": 4, "entity": 12, "purpose": 20}


@torch.no_grad()
def generate(model, processor, frames, prompt_text, max_new_tokens, sample=False, temp=0.7):
    content = [{"type": "image", "image": f} for f in frames]
    content.append({"type": "text", "text": prompt_text})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=frames, return_tensors="pt").to(model.device)
    kw = dict(max_new_tokens=max_new_tokens)
    if sample:
        kw.update(do_sample=True, temperature=temp, top_p=0.9)
    else:
        kw.update(do_sample=False)
    out = model.generate(**inputs, **kw)
    trimmed = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def parse_polarity(raw):
    r = raw.strip().lower()
    if r.startswith("no") or " no" in r[:8]:
        return False
    return True                                     # default affirmative


def clean_entity(raw):
    e = raw.strip().split("\n")[0].strip().strip('"').strip("*").rstrip(".,;:")
    e = re.sub(r"^(the answer is|it is|this is|answer:)\s*", "", e, flags=re.I).strip()
    return " ".join(e.split()[:6]) or "unknown"


def extract(model, processor, frames, question, sample=False):
    kind = classify(question)
    raw = generate(model, processor, frames, EXTRACT_PROMPT[kind].format(q=question),
                   MAX_NEW[kind], sample=sample)
    return (parse_polarity(raw) if kind == "yesno" else clean_entity(raw)), raw


def mbr_select(cands, scorer_fn):
    """Minimum-Bayes-Risk over K samples using BERTScore-F1 as its OWN utility
    (dossier lever #6; training-free, label-free, references never touched)."""
    if len(set(map(str, cands))) == 1:
        return cands[0]
    if isinstance(cands[0], bool):                  # polarity -> majority vote
        return Counter(cands).most_common(1)[0][0]
    best, best_u = cands[0], -1e9
    for i, c in enumerate(cands):
        others = [str(x) for j, x in enumerate(cands) if j != i]
        u = scorer_fn(str(c), others)
        if u > best_u:
            best, best_u = c, u
    return best


# ---------------------------------------------------------------------------

def bootstrap_ci(values, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    boots = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


GENERIC_ENTITY = {"entity": "the tissue", "purpose": "to grasp and hold tissue"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--frames", type=int, nargs="+", default=[1, 5])
    ap.add_argument("--k_samples", type=int, default=5)
    ap.add_argument("--out", default="/scratch/sc20osc/miccai-2026/SurgVU/models/cat2_hybrid_2026.json")
    args = ap.parse_args()

    import surgvu26_scorer as S
    import zeroshot_cat2_2026 as Z

    samples = S.load_samples()
    logger.info("loaded %d sample clips", len(samples))
    refs_list = [s["refs"] for s in samples]
    cases = [s["case"] for s in samples]
    qs = [s["question"] for s in samples]

    for c, q in zip(cases, qs):
        logger.info("ROUTER %s -> %-8s | %s", c, classify(q), q)

    configs = {}

    # ---- no-vision controls (no model needed) -----------------------------
    configs["F0_const_no"] = ["No"] * len(samples)
    configs["N0_template_novision"] = [
        compose(q, True if classify(q) == "yesno" else GENERIC_ENTITY[classify(q)]) for q in qs
    ]
    configs["N1_template_novision_neg"] = [
        compose(q, False if classify(q) == "yesno" else GENERIC_ENTITY[classify(q)]) for q in qs
    ]

    logger.info("loading %s ...", args.model)
    model, processor = Z.load_model(args.model)

    frame_cache = {nf: {s["case"]: Z.sample_frames(s["video"], nf) for s in samples}
                   for nf in args.frames}
    for nf in args.frames:
        logger.info("decoded %d frames/clip", nf)

    scorer = S.get_bert_scorer()

    def utility(cand, others):
        return S.compute_bertscore(cand, others, scorer)

    for nf in args.frames:
        # anchor: replicate the zero-shot winner (bare token) at this frame count
        anchor = []
        for s in samples:
            a = Z.answer(model, processor, frame_cache[nf][s["case"]],
                         Z.PROMPTS["C_bare"].format(q=s["question"]), 64)
            anchor.append(a)
        configs[f"Z_bare_nf{nf}"] = anchor

        # hybrid: greedy short-span extraction -> syntactic template
        hyb = []
        for s in samples:
            slot, raw = extract(model, processor, frame_cache[nf][s["case"]], s["question"])
            ans = compose(s["question"], slot)
            hyb.append(ans)
            logger.info("[H_nf%d] %s | slot=%r raw=%r -> %s", nf, s["case"], slot, raw, ans)
        configs[f"H_hybrid_nf{nf}"] = hyb

    # hybrid + K-sample self-consistency / MBR at the largest frame count
    nf = max(args.frames)
    sc = []
    for s in samples:
        draws = [extract(model, processor, frame_cache[nf][s["case"]], s["question"], sample=True)[0]
                 for _ in range(args.k_samples)]
        slot = mbr_select(draws, utility)
        ans = compose(s["question"], slot)
        sc.append(ans)
        logger.info("[H_sc%d_nf%d] %s | draws=%r -> slot=%r -> %s",
                    args.k_samples, nf, s["case"], draws, slot, ans)
    configs[f"H_hybrid_mbr{args.k_samples}_nf{nf}"] = sc

    # ---- score everything on the official metric --------------------------
    results = {}
    for name, cands in configs.items():
        per_case, agg = S.score_pairs(cands, refs_list, with_nli=True)
        bs = [p["bertscore_f1"] for p in per_case]
        lo, hi = bootstrap_ci(bs)
        results[name] = {
            "aggregates": agg, "bertscore_ci95": [lo, hi],
            "mean_answer_words": float(np.mean([len(c.split()) for c in cands])),
            "per_case": {c: {"question": q, "candidate": p["candidate"],
                             "bertscore_f1": p["bertscore_f1"], "bleu_score": p["bleu_score"],
                             "nli_entailment": p.get("nli_entailment")}
                         for c, q, p in zip(cases, qs, per_case)},
        }
        logger.info("CONFIG %-28s BERTScore-F1=%.4f [%.4f,%.4f] | BLEU=%.4f | words=%.1f",
                    name, agg["bertscore_f1"], lo, hi, agg["bleu_score"],
                    results[name]["mean_answer_words"])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    FLOORS = {"constant-'No' (no vision)": 0.6383,
              "best zero-shot free text (6980942 C_bare_nf5)": 0.7236,
              "authored ~7-word declarative, WRONG (6933607)": 0.7932}
    print("\n=== SUMMARY (n=11, BERTScore-F1 = the 2026 RANKED metric) ===")
    for k, v in sorted(results.items(), key=lambda kv: -kv[1]["aggregates"]["bertscore_f1"]):
        print(f"  {k:30s} {v['aggregates']['bertscore_f1']:.4f}  "
              f"CI95[{v['bertscore_ci95'][0]:.4f},{v['bertscore_ci95'][1]:.4f}]  "
              f"words={v['mean_answer_words']:.1f}")
    print("\n--- pre-registered floors ---")
    for k, v in FLOORS.items():
        print(f"  {k:48s} {v:.4f}")
    best = max(results, key=lambda k: results[k]["aggregates"]["bertscore_f1"])
    h = [k for k in results if k.startswith("H_hybrid")]
    hb = max(h, key=lambda k: results[k]["aggregates"]["bertscore_f1"]) if h else None
    n0 = results["N0_template_novision"]["aggregates"]["bertscore_f1"]
    print(f"\nSURGVU26_HYBRID_BEST={best}:{results[best]['aggregates']['bertscore_f1']:.4f}")
    if hb:
        print(f"SURGVU26_VALUE_OF_VISION_IN_REGISTER="
              f"{results[hb]['aggregates']['bertscore_f1'] - n0:+.4f}  "
              f"({hb} minus N0_template_novision)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

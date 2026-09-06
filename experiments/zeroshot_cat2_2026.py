"""
SurgVU 2026 Cat-2 — ZERO-SHOT VLM BASELINE, scored on the REAL 2026 metric.

Rationale (field evidence): the 2025 podium was won by ZERO-SHOT VLMs.
  1st Capybara 0.4215  zero-shot LLaVA-OneVision-7B (+ classifier grounding), NO VLM fine-tune
  2nd UoM      0.3656  zero-shot InternVL3.5 -- beat its OWN PEFT fine-tune (0.202) by 0.16
So zero-shot is the evidence-backed starting point; fine-tuning is a later, pre-registered arm.

Metric: the 2026 ranked metric is BERTScore-F1 (roberta-large, rescale_with_baseline=True),
max over 5 refs, mean over questions -- NOT the 2025 BLEU. Scored via `surgvu26_scorer`,
a line-for-line clone of the organizers' evaluate.py.

PRE-REGISTERED SWEPT AXIS = ANSWER LENGTH / REGISTER.
Under BERTScore the reference set is a verbosity ladder (refs[0] is always the bare
answer token, refs[1..4] progressively more verbose) and the score is a MAX over that
ladder, so the optimal output length is an empirical question, not a guess. We sweep
four prompt registers x frame counts and let the official scorer decide.

Usage:
  python zeroshot_cat2_2026.py --model Qwen/Qwen3-VL-8B-Instruct --frames 5
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("HF_HOME", "/scratch/sc20osc/hf_cache")

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("zeroshot26")

# ---------------------------------------------------------------------------
# Prompt registers (the pre-registered swept axis)
# ---------------------------------------------------------------------------
PROMPTS = {
    # A: no length steering at all -- the honest "raw VLM" baseline.
    "A_plain":
        "{q}",

    # B: one short declarative sentence (the 2025 BLEU-era recipe).
    "B_short":
        "{q}\n\nAnswer in one short declarative sentence.",

    # C: minimal answer -- targets refs[0], which is always the bare answer token.
    "C_bare":
        "{q}\n\nAnswer with a single word or a very short phrase. Do not write a sentence.",

    # D: register priming with the PUBLIC mock examples from the official eval repo
    #    (public transparency data, not challenge ground truth).
    "D_refstyle":
        "You are answering questions about a short robotic surgery video clip.\n"
        "Answer in the same style as these examples:\n"
        "  Q: Was a surgical instrument used?  A: Yes, a surgical instrument was used.\n"
        "  Q: Was the specific tool used?      A: No, the specific tool was not used.\n"
        "  Q: What tool was used?              A: Forceps were used during the procedure.\n\n"
        "{q}\n\nAnswer:",
}


def sample_frames(video_path, n_frames):
    """Uniformly sample n_frames RGB PIL images from the clip."""
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = 1
    idxs = np.linspace(0, max(total - 1, 0), n_frames).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, img = cap.read()
        if not ok:
            continue
        frames.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {video_path}")
    return frames


def load_model(model_id):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_id)
    if model_id.startswith("Qwen/Qwen3-VL"):
        from transformers import Qwen3VLForConditionalGeneration as Cls
    elif model_id.startswith("Qwen/Qwen2.5-VL"):
        from transformers import Qwen2_5_VLForConditionalGeneration as Cls
    else:
        from transformers import AutoModelForImageTextToText as Cls
    model = Cls.from_pretrained(model_id, torch_dtype="auto", device_map="cuda").eval()
    return model, processor


@torch.no_grad()
def answer(model, processor, frames, prompt_text, max_new_tokens):
    content = [{"type": "image", "image": f} for f in frames]
    content.append({"type": "text", "text": prompt_text})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=frames, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def bootstrap_ci(values, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    boots = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--frames", type=int, nargs="+", default=[1, 5])
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--out", default="/scratch/sc20osc/miccai-2026/SurgVU/models/zeroshot_cat2_2026.json")
    args = ap.parse_args()

    import surgvu26_scorer as S

    samples = S.load_samples()
    logger.info("loaded %d sample clips", len(samples))
    refs_list = [s["refs"] for s in samples]
    cases = [s["case"] for s in samples]

    logger.info("loading %s ...", args.model)
    model, processor = load_model(args.model)

    # decode frames once per frame-count
    frame_cache = {}
    for nf in args.frames:
        frame_cache[nf] = {s["case"]: sample_frames(s["video"], nf) for s in samples}
        logger.info("decoded %d frames/clip for all %d clips", nf, len(samples))

    results = {}
    for nf in args.frames:
        for pname, ptmpl in PROMPTS.items():
            cfg = f"{pname}_nf{nf}"
            cands = []
            for s in samples:
                txt = ptmpl.format(q=s["question"])
                a = answer(model, processor, frame_cache[nf][s["case"]], txt,
                           args.max_new_tokens)
                cands.append(a)
                logger.info("[%s] %s | Q=%s | A=%s", cfg, s["case"], s["question"], a)

            per_case, agg = S.score_pairs(cands, refs_list, with_nli=True)
            bs = [p["bertscore_f1"] for p in per_case]
            lo, hi = bootstrap_ci(bs)
            results[cfg] = {
                "aggregates": agg,
                "bertscore_ci95": [lo, hi],
                "mean_answer_words": float(np.mean([len(c.split()) for c in cands])),
                "per_case": {c: {"candidate": p["candidate"],
                                 "bertscore_f1": p["bertscore_f1"],
                                 "bleu_score": p["bleu_score"],
                                 "nli_entailment": p.get("nli_entailment")}
                             for c, p in zip(cases, per_case)},
            }
            logger.info(
                "CONFIG %-16s BERTScore-F1=%.4f [%.4f,%.4f] (RANKED) | BLEU=%.4f "
                "| NLI=%.4f | mean_words=%.1f",
                cfg, agg["bertscore_f1"], lo, hi, agg["bleu_score"],
                agg.get("nli_entailment", float("nan")),
                results[cfg]["mean_answer_words"])

    best = max(results, key=lambda k: results[k]["aggregates"]["bertscore_f1"])
    results["_best"] = {"config": best,
                        "bertscore_f1": results[best]["aggregates"]["bertscore_f1"]}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== SUMMARY (n=11, BERTScore-F1 = the 2026 ranked metric) ===")
    for k, v in results.items():
        if k.startswith("_"):
            continue
        print(f"  {k:18s} {v['aggregates']['bertscore_f1']:.4f}  "
              f"CI95[{v['bertscore_ci95'][0]:.4f},{v['bertscore_ci95'][1]:.4f}]  "
              f"words={v['mean_answer_words']:.1f}")
    print(f"\nSURGVU26_ZEROSHOT_BEST={best}:{results['_best']['bertscore_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

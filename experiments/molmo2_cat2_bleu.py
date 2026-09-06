"""SurgVU-2026 Cat-2 SECONDARY zero-shot probe: allenai/Molmo2-8B grounded-BLEU.

Question: does a stronger native-grounding VLM (Molmo2-8B, Apache-2.0; SigLIP-2 vision
+ Qwen3-8B LM, native grounding/pointing) lift REAL grounded-BLEU on the 11 SurgVU Cat-2
sample clips vs our classifier-spine approach?

Bars (sanity only, NOT targets):
  - spine grounded-BLEU = 0.7428 (n=11, OPTIMISTIC upper bound — templates exact-matched to
    these 11 refs; sanity check only).
  - 2025 winner (Capybara, zero-shot LLaVA-OV-7B + classifiers) = 0.4215 (the real bar).

This is an HONEST raw zero-shot Molmo2 measurement: NO template router, NO post-processing
toward the references (no leakage). The only constraint is a prompt suffix asking for ONE
short declarative sentence — the BLEU metric punishes verbose answers (a 14-word correct
answer ~0.21; bare "No" ~0.18; a clean 4-7-word reference-register sentence scores well).

Frame sampling: ONE representative MIDDLE frame of the 30s clip (cleanest single-image
Molmo2 call; Capybara used only 5 sampled frames — the point here is a real-vision sanity
check, not over-engineering).

Molmo2 load/generate logic copied VERBATIM from the validated ORENA runner
(/users/sc20osc/ORENA/experiments/run_frame_zeroshot_molmo2.py) — trust_remote_code load,
single-user-turn chat template (no system role), greedy decode, decode only new tokens.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# bleu_harness lives in the SurgVU experiments dir
sys.path.insert(0, "/users/sc20osc/SurgVU/experiments")
import bleu_harness  # noqa: E402

import cv2  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

MODEL_ID = "allenai/Molmo2-8B"
DATA_DIR = "/users/sc20osc/SurgVU/data"
OUT_DIR = Path("/scratch/sc20osc/miccai-2026/SurgVU/models")
OUT_JSON = OUT_DIR / "molmo2_cat2_bleu.json"

# Length-constraint prompt suffix (constrain length via the prompt only — NO leakage toward
# the refs). BLEU rewards a concise canonical sentence in the reference register.
LEN_SUFFIX = " Answer in one short declarative sentence."

_MODEL = None
_PROC = None


def _load_model():
    """Lazy singleton: load Molmo2-8B ONCE so all 11 cases reuse it.

    Copied from the validated ORENA runner: Molmo2 ships remote code
    (Molmo2ForConditionalGeneration); AutoModelForImageTextToText dispatches to it via
    trust_remote_code. torch_dtype='auto', device_map='cuda'."""
    global _MODEL, _PROC
    if _MODEL is None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        _PROC = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        _MODEL = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, trust_remote_code=True, torch_dtype="auto", device_map="cuda"
        ).eval()
    return _MODEL, _PROC


def _middle_frame(mp4_path: str) -> Image.Image:
    """Decode the temporal MIDDLE frame of the clip as a PIL RGB image (cv2)."""
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open {mp4_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid = max(0, n // 2) if n > 0 else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ok, frame = cap.read()
    if not ok:
        # fall back to the first decodable frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"cv2 could not decode a frame from {mp4_path}")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


@torch.no_grad()
def _ask(question: str, pil_image: Image.Image) -> str:
    """Single user turn (Molmo2 chat template forbids a system role): fold the length
    instruction into the user text; image as dict(type=image, image=<PIL>). Greedy decode,
    return only the newly-generated tokens, stripped."""
    model, proc = _load_model()
    user_text = f"{question}{LEN_SUFFIX}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": user_text},
            ],
        },
    ]
    inputs = proc.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    in_len = inputs["input_ids"].shape[1]
    gen_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    new_tokens = gen_ids[0, in_len:]
    return proc.tokenizer.decode(
        new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    ).strip()


def predict_fn(question: str, case_id: str) -> str:
    """harness contract: predict_fn(question, case_id) -> answer string.

    Loads the case mp4, samples the middle frame, asks Molmo2 the question (length-capped
    via the prompt). On any failure return a safe non-empty fallback so the harness never
    crashes (an empty answer scores 0 BLEU anyway)."""
    mp4 = f"{DATA_DIR}/{case_id}/{case_id}.mp4"
    try:
        img = _middle_frame(mp4)
        ans = _ask(question, img)
        return ans if ans else "No"
    except Exception as exc:  # noqa: BLE001
        print(f"  [{case_id}] predict_fn error: {exc}", flush=True)
        return "No"


def main():
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # warm the model once up front (clearer logs / timing)
    _load_model()
    print(f"Molmo2-8B loaded in {time.perf_counter() - t0:.1f}s; running 11 Cat-2 clips...",
          flush=True)

    mean, per_case = bleu_harness.evaluate(predict_fn, data_dir=DATA_DIR, verbose=True)

    payload = {
        "mean_bleu": mean,
        "model": MODEL_ID,
        "frame_sampling": "single middle frame",
        "len_suffix": LEN_SUFFIX,
        "per_case": [
            {"case": cid, "q": q, "pred": pred, "bleu": score}
            for (cid, q, pred, score) in per_case
        ],
        "bars": {"spine_n11": 0.7428, "winner_2025": 0.4215},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nWrote {OUT_JSON}")
    print(f"MOLMO2_CAT2_BLEU={mean:.4f}")


if __name__ == "__main__":
    main()

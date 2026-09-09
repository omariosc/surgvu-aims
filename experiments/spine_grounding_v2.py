"""Cat-2 grounding V2 -- adds ACTION bridges on top of spine_grounding.ground_clip.

The v1 grounding (`spine_grounding.py`) only sets `g['present']` for questions whose
noun matches a tool key in QUESTION_TOOL_GROUPS. Two action questions in the in-hand
set carry NO tool noun, so present is never set -> defaults False in the router:

  * case131 "Is tissue being cut during this clip?"   (ref: "Yes, tissue is being cut.")
  * case125 "Is a suture required in this surgical step?" (ref: "Yes, sutures are required.")

This module COPIES the v1 ground_clip aggregation but ADDS two action->present bridges
that consume the SAME spine signals (tool-presence + dominant task) already computed:

  (a) CUT / CUTTING action -> present=True when the dominant task is a dissection/cutting
      task OR a cutting tool is present (monopolar_curved_scissor, permanent cautery
      hook/spatula, vessel_sealer).
  (b) SUTURE-REQUIRED action -> present=True when needle_driver is present OR the dominant
      task is "Suturing".

All existing tool/organ/forceps logic is preserved unchanged. The action bridges only
FIRE present=True (they never force False), so a confident negative is still possible by
the spine simply not detecting the cutting/suturing signal.

Importable without a checkpoint; the heavy decode+forward only runs in ground_clip_v2().
Mirrors the (n_frames, thr_override, pooling) knobs of grounding_variants_bleu so the
final pick can reuse the best grounding config (nf4_thr0.3_mean).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import spine_grounding as sg  # reuse TASK_TO_ORGAN, TOOL_DISPLAY, _question_tool_keys

# Spine tool keys that, when present, imply a CUTTING action is underway.
CUT_TOOL_KEYS = ("monopolar_curved_scissor", "permanent_cautery_hook_spatula",
                 "vessel_sealer")
# Dominant tasks that imply cutting/dissection (the SurgVU tasks are anatomy/skill-named;
# dissection of these structures is a cutting action). "Suspensory ligaments" and
# "Rectal artery/vein" dissection both involve cutting; keep it inclusive of the dissection
# tasks but NOT the generic "Other" / motion / retraction tasks.
CUT_TASKS = ("Suspensory ligaments", "Rectal artery/vein")
# Suture-required is implied by the needle driver OR the Suturing task.
SUTURE_TOOL_KEYS = ("needle_driver",)
SUTURE_TASKS = ("Suturing",)


def _is_cut_q(ql):
    return "cut" in ql or "cutting" in ql


def _is_suture_required_q(ql):
    return "sutur" in ql and "requir" in ql


def _action_present(question, present_keys, dom_task):
    """Return (handled, present_bool) for the cut / suture-required action questions.

    handled=False means this is not an action question -> caller falls back to v1 logic.
    """
    ql = question.lower()
    if _is_cut_q(ql):
        present = (any(k in present_keys for k in CUT_TOOL_KEYS)
                   or dom_task in CUT_TASKS)
        return True, present
    if _is_suture_required_q(ql):
        present = (any(k in present_keys for k in SUTURE_TOOL_KEYS)
                   or dom_task in SUTURE_TASKS)
        return True, present
    return False, False


def _aggregate(tp, kp, thr, pooling, TOOLS, TASKS):
    pooled = tp.max(axis=0) if pooling == "max" else tp.mean(axis=0)
    present_keys = {TOOLS[i] for i in range(len(TOOLS)) if pooled[i] > thr}
    dom_task = TASKS[int(kp.mean(axis=0).argmax())]
    return present_keys, dom_task


def _grounding_from(question, present_keys, dom_task):
    """Build the {present, tool, organ} dict the router consumes, with the v2 action
    bridges layered on the v1 tool/organ logic."""
    g = {}
    ql = question.lower()

    # (1) action bridge first (cut / suture-required carry no tool noun in v1)
    handled, present = _action_present(question, present_keys, dom_task)
    if handled:
        g["present"] = present

    # (2) v1 tool-presence yes/no (forceps / needle driver / scissors / ...)
    if "present" not in g:
        noun, keys = sg._question_tool_keys(question)
        if keys:
            g["present"] = any(k in present_keys for k in keys)

    # (3) v1 forceps-type open question
    if "what type" in ql and "forceps" in ql:
        for k in ("cadiere_forceps", "prograsp_forceps", "bipolar_forceps"):
            if k in present_keys:
                g["tool"] = sg.TOOL_DISPLAY[k]
                break

    # (4) v1 organ question (from dominant task)
    if "organ" in ql:
        g["organ"] = sg.TASK_TO_ORGAN.get(dom_task, dom_task.lower())
    return g


# ---- raw per-frame prob extraction (decode+forward once, cache by (clip,n_frames)) ----
def extract_probs(video_path, ckpt_path, n_frames):
    import torch
    import cv2
    from train_spine import Spine, TOOLS, TASKS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = Spine(ck["args"]["backbone"]).to(device).eval()
    model.load_state_dict(ck["model"])
    img = ck["args"]["img_size"]
    ck_thr = ck.get("tool_thr", 0.5)
    cap = cv2.VideoCapture(video_path)
    dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1)) or 30.0
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    tp, kp = [], []
    for t in np.linspace(0.5, max(dur - 0.5, 1.0), n_frames):
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
        ok, fr = cap.read()
        if not ok or fr is None:
            continue
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        h, w = fr.shape[:2]
        cy, cx = int(h * 0.06), int(w * 0.06)
        fr = cv2.resize(fr[cy:h - cy, cx:w - cx], (img, img), interpolation=cv2.INTER_AREA)
        x = ((fr.astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1)
        with torch.no_grad():
            lt, lk = model(torch.from_numpy(x[None]).to(device))
        tp.append(torch.sigmoid(lt)[0].cpu().numpy())
        kp.append(torch.softmax(lk, 1)[0].cpu().numpy())
    cap.release()
    return (np.stack(tp) if tp else None,
            np.stack(kp) if kp else None, ck_thr, TOOLS, TASKS)


def ground_clip_v2(video_path, question, ckpt_path, n_frames=4, thr_override=0.3,
                   pooling="mean", _cache=None):
    """v2 grounding for the best config (nf4_thr0.3_mean by default). Returns the
    {present, tool, organ} dict for the v2 template router."""
    _cache = _cache if _cache is not None else {}
    key = (video_path, n_frames)
    if key not in _cache:
        _cache[key] = extract_probs(video_path, ckpt_path, n_frames)
    tp, kp, ck_thr, TOOLS, TASKS = _cache[key]
    if tp is None:
        return {}
    thr = ck_thr if thr_override is None else thr_override
    present_keys, dom_task = _aggregate(tp, kp, thr, pooling, TOOLS, TASKS)
    return _grounding_from(question, present_keys, dom_task)


if __name__ == "__main__":
    import argparse, json, glob
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--samples_dir", default="/users/USERNAME/SurgVU/data")
    ap.add_argument("--n_frames", type=int, default=4)
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--pool", default="mean")
    args = ap.parse_args()
    cache = {}
    for d in sorted(glob.glob(os.path.join(args.samples_dir, "case*"))):
        cid = os.path.basename(d)
        qf = os.path.join(d, f"{cid}_question.json")
        vf = os.path.join(d, f"{cid}.mp4")
        if not (os.path.exists(qf) and os.path.exists(vf)):
            continue
        q = json.load(open(qf))
        g = ground_clip_v2(vf, q, args.ckpt, args.n_frames, args.thr, args.pool, cache)
        print(f"{cid}: grounding={g}  | Q={q!r}")

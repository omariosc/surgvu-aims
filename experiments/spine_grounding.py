"""P1.2 wiring -- replace P0.2's hand-set GROUNDING with REAL spine predictions.

Runs the trained grounding-spine (train_spine.py checkpoint) over frames sampled
from a 30 s Cat-2 clip, aggregates to:
  - tool-presence set (max-pooled sigmoid > tool_thr over sampled frames)
  - dominant task (argmax of mean task-softmax)
and derives the grounding facts the template router consumes:
  {present: bool (for the queried tool), tool: <name>, organ: <name>}.

The task->organ bridge (the SurgVU tasks ARE anatomy-named, so they directly
yield the manipulated organ for the 'organ' question bucket):
  Uterine horn -> uterine horn ; Suspensory ligaments -> suspensory ligaments ;
  Rectal artery/vein -> rectal artery/vein ; (others -> no specific organ).

This module is the bridge from the classifier spine (P1.1) to the Cat-2 BLEU
template bank (P0.2). It is import-safe without a checkpoint (the heavy bits run
only inside ground_clip()).
"""
import re

TASK_TO_ORGAN = {
    "Uterine horn": "uterine horn",
    "Suspensory ligaments": "suspensory ligaments",
    "Rectal artery/vein": "rectal artery/vein",
}

# spine tool key -> the surface noun used in the reference register
TOOL_DISPLAY = {
    "needle_driver": "needle driver",
    "monopolar_curved_scissor": "monopolar curved scissors",
    "force_bipolar": "force bipolar",
    "clip_applier": "clip applier",
    "tip_up_fenestrated_grasper": "tip-up fenestrated grasper",
    "cadiere_forceps": "Cadiere Forceps",
    "bipolar_forceps": "bipolar forceps",
    "vessel_sealer": "vessel sealer",
    "suction_irrigator": "suction irrigator",
    "bipolar_dissector": "bipolar dissector",
    "prograsp_forceps": "prograsp forceps",
    "stapler": "stapler",
    "permanent_cautery_hook_spatula": "permanent cautery hook/spatula",
    "grasping_retractor": "grasping retractor",
}

# question-noun -> the spine tool keys that satisfy it (a 'forceps' question is
# satisfied by ANY forceps-type tool present)
QUESTION_TOOL_GROUPS = {
    "large needle driver": ["needle_driver"],
    "needle driver": ["needle_driver"],
    "forceps": ["cadiere_forceps", "bipolar_forceps", "prograsp_forceps",
                "tip_up_fenestrated_grasper"],
    "scissors": ["monopolar_curved_scissor"],
    "stapler": ["stapler"],
    "clip applier": ["clip_applier"],
    "vessel sealer": ["vessel_sealer"],
}


def _question_tool_keys(question):
    ql = question.lower()
    for noun in sorted(QUESTION_TOOL_GROUPS, key=len, reverse=True):
        if noun in ql:
            return noun, QUESTION_TOOL_GROUPS[noun]
    return None, []


def aggregate_predictions(tool_probs, task_probs, tool_thr, tools, tasks):
    """tool_probs: [F,14] sigmoid; task_probs: [F,8] softmax (numpy)."""
    import numpy as np
    tool_present = (tool_probs.max(axis=0) > tool_thr)  # max-pool over frames
    present_keys = {tools[i] for i in range(len(tools)) if tool_present[i]}
    dom_task = tasks[int(task_probs.mean(axis=0).argmax())]
    return present_keys, dom_task


def ground_clip(video_path, question, ckpt_path, n_frames=8, device=None):
    """Sample n_frames uniformly across the clip, run the spine, return the
    grounding dict {present, tool, organ} for the template router."""
    import numpy as np
    import torch
    import cv2
    from train_spine import Spine, TOOLS, TASKS

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = Spine(ck["args"]["backbone"]).to(device).eval()
    model.load_state_dict(ck["model"])
    thr = ck.get("tool_thr", 0.5)
    img = ck["args"]["img_size"]

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
    if not tp:
        return {}
    present_keys, dom_task = aggregate_predictions(
        np.stack(tp), np.stack(kp), thr, TOOLS, TASKS)

    g = {}
    noun, keys = _question_tool_keys(question)
    if keys:  # yes/no presence question -> is the queried tool present?
        g["present"] = any(k in present_keys for k in keys)
    ql = question.lower()
    if "what type" in ql and "forceps" in ql:  # tool-type question
        for k in ("cadiere_forceps", "prograsp_forceps", "bipolar_forceps"):
            if k in present_keys:
                g["tool"] = TOOL_DISPLAY[k]
                break
    if "organ" in ql:  # organ question -> from dominant task
        g["organ"] = TASK_TO_ORGAN.get(dom_task, dom_task.lower())
    return g


if __name__ == "__main__":
    import argparse, json, glob, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--samples_dir", default="/users/sc20osc/SurgVU/data")
    args = ap.parse_args()
    for d in sorted(glob.glob(os.path.join(args.samples_dir, "case*"))):
        cid = os.path.basename(d)
        qf = os.path.join(d, f"{cid}_question.json")
        vf = os.path.join(d, f"{cid}.mp4")
        if not (os.path.exists(qf) and os.path.exists(vf)):
            continue
        q = json.load(open(qf))
        g = ground_clip(vf, q, args.ckpt)
        print(f"{cid}: grounding={g}  | Q={q!r}")

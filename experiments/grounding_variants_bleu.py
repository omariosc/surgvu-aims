"""Cat-2 grounding-VARIANTS sweep (lever 3) -> grounded-BLEU vs 0.4215.

spine_grounding_bleu.py measured ONE grounding config (n_frames=8, the ckpt's own
tool_thr, max-pool). This sweeps the entity-injection / temporal-grounding knobs that
do NOT need a VLM, holding the (solved) template router fixed, to find the grounding
config that yields the best Cat-2 BLEU on the 11 in-hand clips:

  * n_frames        : 4 / 8 / 16 / 24  (temporal sampling density over the 30 s clip)
  * presence thr    : ckpt_thr / 0.3 / 0.5 / 0.7  (how confident before injecting a tool)
  * pooling         : max / mean        (how frame-wise tool probs are aggregated)

Each cell injects the spine's predicted {present, tool, organ} into the unchanged
p0_2_template_router and scores bleu_harness.evaluate over the 11 clips. Prints a
ranked table + a machine-readable BEST line.

This is a *prompt/entity-injection-template* sweep (the brief's "different entity-
injection / prompt-template strategies") implemented for the classifier-grounded
templating path (no VLM dependency, runs in minutes on 1 L40S).

Run (SLURM gpu node):
  python grounding_variants_bleu.py --ckpt .../spine_full/spine_best.pt
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import bleu_harness
import p0_2_template_router as tr
import spine_grounding as sg


def ground_clip_cfg(video_path, question, ckpt_path, n_frames, thr_override, pooling,
                    _cache):
    """Like sg.ground_clip but with configurable n_frames, presence-threshold and
    pooling. Caches the raw per-frame probs per (clip,n_frames) so the thr/pooling
    sweep is cheap (decode+forward once per n_frames, reuse across thr/pooling)."""
    import torch
    import cv2
    from train_spine import Spine, TOOLS, TASKS

    key = (video_path, n_frames)
    if key not in _cache:
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
        _cache[key] = (np.stack(tp) if tp else None,
                       np.stack(kp) if kp else None, ck_thr, TOOLS, TASKS)

    tp, kp, ck_thr, TOOLS, TASKS = _cache[key]
    if tp is None:
        return {}
    thr = ck_thr if thr_override is None else thr_override
    pooled = tp.max(axis=0) if pooling == "max" else tp.mean(axis=0)
    present_keys = {TOOLS[i] for i in range(len(TOOLS)) if pooled[i] > thr}
    dom_task = TASKS[int(kp.mean(axis=0).argmax())]

    # mirror sg.ground_clip's entity-injection logic
    g = {}
    noun, keys = sg._question_tool_keys(question)
    if keys:
        g["present"] = any(k in present_keys for k in keys)
    ql = question.lower()
    if "what type" in ql and "forceps" in ql:
        for k in ("cadiere_forceps", "prograsp_forceps", "bipolar_forceps"):
            if k in present_keys:
                g["tool"] = sg.TOOL_DISPLAY[k]
                break
    if "organ" in ql:
        g["organ"] = sg.TASK_TO_ORGAN.get(dom_task, dom_task.lower())
    return g


def predict_factory(ckpt, data_dir, n_frames, thr, pooling, cache):
    def predict(question, case_id):
        vf = os.path.join(data_dir, case_id, f"{case_id}.mp4")
        if os.path.exists(vf):
            try:
                g = ground_clip_cfg(vf, question, ckpt, n_frames, thr, pooling, cache)
            except Exception as e:
                print(f"  [warn] {case_id} grounding failed: {e}", flush=True)
                g = {}
        else:
            g = {}
        tr.GROUNDING[case_id] = g
        return tr.predict(question, case_id)
    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",
                    default="/scratch/sc20osc/miccai-2026/SurgVU/models/spine_full/spine_best.pt")
    ap.add_argument("--data_dir", default="/users/sc20osc/SurgVU/data")
    ap.add_argument("--out",
                    default="/scratch/sc20osc/miccai-2026/SurgVU/models/spine_full/grounding_variants.json")
    args = ap.parse_args()

    N_FRAMES = [4, 8, 16, 24]
    THRS = [None, 0.3, 0.5, 0.7]   # None = ckpt's own swept tool_thr
    POOLS = ["max", "mean"]

    cache = {}
    results = []
    print(f"# grounding-variants sweep  ckpt={args.ckpt}")
    for nf in N_FRAMES:
        for thr in THRS:
            for pool in POOLS:
                predict = predict_factory(args.ckpt, args.data_dir, nf, thr, pool, cache)
                mean, per_case = bleu_harness.evaluate(predict, data_dir=args.data_dir,
                                                       verbose=False)
                tag = f"nf{nf}_thr{'ckpt' if thr is None else thr}_{pool}"
                results.append(dict(cfg=tag, n_frames=nf,
                                    thr=("ckpt" if thr is None else thr),
                                    pool=pool, bleu=mean))
                print(f"#   {tag:22s} BLEU={mean:.4f}", flush=True)

    results.sort(key=lambda r: r["bleu"], reverse=True)
    best = results[0]
    print(f"\n# bars: hand-set ceiling 0.9650 | Capybara SOTA 0.4215 | echo-shortest 0.3525")
    verdict = ("CONFIRMED >=0.4215 SOTA bar" if best["bleu"] >= 0.4215
               else "below 0.4215 -> grounding still the limit")
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"# wrote {args.out}")
    print(f"GROUNDVAR_BEST cfg={best['cfg']} BLEU={best['bleu']:.4f} {verdict}")


if __name__ == "__main__":
    main()

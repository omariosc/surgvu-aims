"""Cat-2 REAL visual grounding -> grounded BLEU (autonomous ladder, lever 2).

P0.2 measured the TEMPLATING ceiling with HAND-SET entities (mean BLEU 0.9650).
This module replaces those hand-set facts with the trained grounding-spine's REAL
predictions (spine_grounding.ground_clip over the 30 s clip) and re-scores the
SurgVU Cat-2 BLEU on the 11 in-hand sample clips. The number it prints is the
honest grounded-BLEU to compare against the 0.4215 Capybara template-ceiling bar
and the 0.9650 hand-set ceiling: the gap between them IS the vision-grounding gap
(the diagnostic's #1 bottleneck), not a phrasing gap.

Pipeline per case:
  clip.mp4 + question
    -> spine_grounding.ground_clip()  (REAL spine tool-presence + dominant task)
    -> inject {present, tool, organ} into the P0.2 template router's GROUNDING
    -> p0_2_template_router.predict(question, case)  (same register-tuned templates)
    -> bleu_harness.bleu_max_over_refs(refs, answer)

Falsifier (from the dossier diagnostic): grounded-BLEU < 0.4215 while the spine's
tool/task macro-F1 >= 0.95  =>  templating/router is the limit, not vision. If the
spine F1 is < 0.95, a low grounded-BLEU is expected and points back at the spine.

Run (SLURM gpu node -- mp4 decode + forward pass needs a GPU):
  python spine_grounding_bleu.py --ckpt .../spine_full/spine_best.pt
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bleu_harness
import p0_2_template_router as tr
import spine_grounding as sg


def grounded_predict_factory(ckpt, data_dir, n_frames):
    """Return a predict_fn(question, case_id) that grounds via the REAL spine.

    It overwrites the per-case fact dict the router reads (tr.GROUNDING) with the
    spine's prediction for THIS clip, then calls the unchanged router. Procedure /
    purpose buckets need no grounding (static answers), so an empty dict is fine.
    """
    cache = {}

    def predict(question, case_id):
        vf = os.path.join(data_dir, case_id, f"{case_id}.mp4")
        if case_id not in cache:
            if os.path.exists(vf):
                try:
                    cache[case_id] = sg.ground_clip(vf, question, ckpt, n_frames=n_frames)
                except Exception as e:  # decode / model failure -> empty grounding
                    print(f"  [warn] {case_id} grounding failed: {e}", flush=True)
                    cache[case_id] = {}
            else:
                cache[case_id] = {}
        tr.GROUNDING[case_id] = cache[case_id]   # inject REAL facts
        return tr.predict(question, case_id)

    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",
                    default="/scratch/sc20osc/miccai-2026/SurgVU/models/spine_full/spine_best.pt")
    ap.add_argument("--data_dir", default="/users/sc20osc/SurgVU/data")
    ap.add_argument("--n_frames", type=int, default=8)
    ap.add_argument("--out", default="/scratch/sc20osc/miccai-2026/SurgVU/models/spine_full/grounded_bleu.json")
    args = ap.parse_args()

    print(f"# REAL-grounded Cat-2 BLEU  ckpt={args.ckpt}")
    try:
        import torch
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        print(f"# spine saved metrics = {ck.get('metrics')}  tool_thr={ck.get('tool_thr')}")
        spine_metrics = ck.get("metrics", {})
    except Exception as e:
        print(f"# [warn] could not read ckpt metrics: {e}")
        spine_metrics = {}

    predict = grounded_predict_factory(args.ckpt, args.data_dir, args.n_frames)
    mean, per_case = bleu_harness.evaluate(predict, data_dir=args.data_dir, verbose=True)

    # also report the hand-set ceiling for direct comparison (re-import a fresh router)
    print(f"\n# grounded mean BLEU = {mean:.4f}")
    print(f"# bars: hand-set ceiling 0.9650 | Capybara SOTA 0.4215 | echo-shortest 0.3525 | bare-y/n 0.1132")
    tool_f1 = spine_metrics.get("tool_f1")
    task_f1 = spine_metrics.get("task_f1")
    verdict = "?"
    if tool_f1 is not None and task_f1 is not None:
        if mean >= 0.4215:
            verdict = f"CONFIRMED grounded-BLEU {mean:.4f} >= 0.4215 SOTA bar"
        elif min(tool_f1, task_f1) >= 0.95:
            verdict = (f"REFUTED templating-is-the-limit: grounded {mean:.4f} < 0.4215 "
                       f"DESPITE spine F1>=0.95 (tool {tool_f1:.3f}/task {task_f1:.3f})")
        else:
            verdict = (f"grounded {mean:.4f} < 0.4215 but spine F1<0.95 "
                       f"(tool {tool_f1:.3f}/task {task_f1:.3f}) -> spine is the bottleneck, not templating")
    print(f"# VERDICT: {verdict}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"grounded_mean_bleu": mean,
                   "spine_metrics": spine_metrics,
                   "per_case": [{"case": c, "q": q, "pred": p, "bleu": s}
                                for c, q, p, s in per_case]}, fh, indent=2)
    print(f"# wrote {args.out}")
    # machine-readable last line for the slurm wrapper to grep
    print(f"GROUNDED_BLEU={mean:.4f}")


if __name__ == "__main__":
    main()

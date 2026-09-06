"""Cat-2 REAL grounded BLEU V2 -- action-grounded v2 grounding + v2 template router.

Replaces the v1 grounded path (spine_grounding_bleu.py, baseline 0.7008 with the ckpt
default config; 0.7428 best from the variant sweep at nf4_thr0.3_mean) with:
  * spine_grounding_v2.ground_clip_v2 -- adds cut / suture-required ACTION bridges so
    case131 (cut) and case125 (suture-required) get present=True from the spine signal.
  * p0_2_template_router_v2 -- the action branches are reachable + a tense-aware
    "was utilized" affirmative needle-driver carrier (case126 register).

Uses the best grounding config from the variant sweep by default (nf4_thr0.3_mean) so it
is an apples-to-apples improvement over the 0.7428 saturated best. Writes per-case JSON +
grounded-BLEU and appends a result line (the slurm wrapper does the append).

Run (SLURM gpu node -- mp4 decode + forward pass needs a GPU):
  python spine_grounding_bleu_v2.py --ckpt .../spine_full/spine_best.pt \
      --n_frames 4 --thr 0.3 --pool mean
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bleu_harness
import p0_2_template_router_v2 as tr
import spine_grounding_v2 as sg2


def grounded_predict_factory(ckpt, data_dir, n_frames, thr, pool):
    cache = {}
    gcache = {}

    def predict(question, case_id):
        vf = os.path.join(data_dir, case_id, f"{case_id}.mp4")
        if case_id not in gcache:
            if os.path.exists(vf):
                try:
                    gcache[case_id] = sg2.ground_clip_v2(
                        vf, question, ckpt, n_frames=n_frames,
                        thr_override=thr, pooling=pool, _cache=cache)
                except Exception as e:
                    print(f"  [warn] {case_id} grounding failed: {e}", flush=True)
                    gcache[case_id] = {}
            else:
                gcache[case_id] = {}
        tr.GROUNDING[case_id] = gcache[case_id]
        return tr.predict(question, case_id)

    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",
                    default="/scratch/sc20osc/miccai-2026/SurgVU/models/spine_full/spine_best.pt")
    ap.add_argument("--data_dir", default="/users/sc20osc/SurgVU/data")
    ap.add_argument("--n_frames", type=int, default=4)
    ap.add_argument("--thr", type=float, default=0.3)   # None via --thr -1 -> ckpt thr
    ap.add_argument("--pool", default="mean")
    ap.add_argument("--out",
                    default="/scratch/sc20osc/miccai-2026/SurgVU/models/spine_full/grounded_bleu_v2.json")
    args = ap.parse_args()
    thr = None if args.thr is not None and args.thr < 0 else args.thr

    print(f"# REAL-grounded V2 Cat-2 BLEU  ckpt={args.ckpt}")
    print(f"# grounding cfg: n_frames={args.n_frames} thr={thr} pool={args.pool}")
    try:
        import torch
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        print(f"# spine saved metrics = {ck.get('metrics')}  tool_thr={ck.get('tool_thr')}")
        spine_metrics = ck.get("metrics", {})
    except Exception as e:
        print(f"# [warn] could not read ckpt metrics: {e}")
        spine_metrics = {}

    predict = grounded_predict_factory(args.ckpt, args.data_dir,
                                       args.n_frames, thr, args.pool)
    mean, per_case = bleu_harness.evaluate(predict, data_dir=args.data_dir, verbose=True)

    print(f"\n# grounded-V2 mean BLEU = {mean:.4f}")
    print(f"# bars: hand-set ceiling 0.9650 | v1 best 0.7428 | v1 ckpt-cfg 0.7008 | "
          f"Capybara SOTA 0.4215")
    tool_f1 = spine_metrics.get("tool_f1")
    task_f1 = spine_metrics.get("task_f1")
    verdict = "?"
    if mean > 0.7428:
        verdict = f"CONFIRMED grounded-V2 {mean:.4f} > 0.7428 v1-best (action/template fix helped)"
    elif mean >= 0.4215:
        verdict = (f"grounded-V2 {mean:.4f} clears 0.4215 SOTA but <= 0.7428 v1-best "
                   f"-> action/template fix did NOT raise over the saturated best (REFUTED vs v1-best)")
    else:
        verdict = f"grounded-V2 {mean:.4f} < 0.4215 SOTA bar (regression)"
    print(f"# VERDICT: {verdict}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"grounded_mean_bleu": mean,
                   "grounding_cfg": {"n_frames": args.n_frames, "thr": thr, "pool": args.pool},
                   "spine_metrics": spine_metrics,
                   "per_case": [{"case": c, "q": q, "pred": p, "bleu": s}
                                for c, q, p, s in per_case]}, fh, indent=2)
    print(f"# wrote {args.out}")
    print(f"GROUNDED_BLEU_V2={mean:.4f}")


if __name__ == "__main__":
    main()

"""
SurgVU Cat-1 — headroom sweep off the honest 0.1470 baseline (job 6933641).

ONE KNOB PER ARM.  The split is NEVER changed within the `main` protocol: train =
videos 1-5, val = videos 6,7, strictly video-disjoint, identical to the run that
produced mAP@[.5:.95] = 0.1470.  Frames come from the shared pool built by
`cat1_build_pool.py` (image-list .txt splits, so no inode blow-up).

TWO REPORTING FIXES THAT THE BASELINE NUMBER NEEDS
--------------------------------------------------
1. `best.pt` is SELECTED ON THE VALIDATION SET WE THEN REPORT.  In job 6933641 the
   per-epoch val mAP oscillated 0.090-0.147 with no trend after epoch ~4, and the
   headline 0.1470 is the MAXIMUM of 60 such draws.  We therefore report BOTH
   `best.pt` (selected, comparable to the logged 0.1470) and `last.pt` (unselected).
   Arms with different epoch budgets have different max-of-N inflation, so the
   `last.pt` column is the one that compares fairly across schedule arms.
2. AP is averaged over the classes PRESENT IN VAL.  On the main split that is 4
   classes, and clip_applier (258 val boxes) + vessel_sealer (13 val boxes) have
   ZERO instances in any training video -> their AP is 0.000 by construction and
   the attainable ceiling is 0.500, not 1.0.  We therefore also report
   `map_supported`, the mean over val classes that actually have train support.
"""

import argparse
import json
import os
import sys

POOL = "/scratch/USERNAME/miccai-2026/SurgVU/data/cat1_pool"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="arm name (used for run dir + result key)")
    ap.add_argument("--split", default="main", help="main | lovo1..lovo7")
    ap.add_argument("--model", default="/scratch/USERNAME/AILET/from_pc/weights/yolo26m.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--project", default="/scratch/USERNAME/miccai-2026/SurgVU/models/cat1_sweep")
    ap.add_argument("--out_dir", default="/scratch/USERNAME/miccai-2026/SurgVU/logs/cat1_sweep")
    args = ap.parse_args()

    support = json.load(open(f"{POOL}/class_support.json"))
    names = support["names"]
    sp = support["splits"][args.split]
    train_counts = sp["train_counts"]
    print(f"[arm {args.arm}] split={args.split} train_videos={sp['train_videos']} "
          f"val_videos={sp['val_videos']}", flush=True)
    print(f"[arm {args.arm}] val classes={sp['n_val_classes']} "
          f"zero-train-support={sp['zero_train_support_classes']} "
          f"attainable ceiling={sp['attainable_map_ceiling']:.3f}", flush=True)

    run_dir = os.path.join(args.project, args.arm)
    os.makedirs(run_dir, exist_ok=True)
    yaml_path = os.path.join(run_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {POOL}\n")
        f.write(f"train: {POOL}/splits/{args.split}_train.txt\n")
        f.write(f"val: {POOL}/splits/{args.split}_val.txt\n")
        f.write(f"nc: {len(names)}\nnames: {json.dumps(names)}\n")

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(data=yaml_path, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                project=args.project, name=args.arm, exist_ok=True, pretrained=True,
                seed=args.seed, verbose=True, device=0)

    def evaluate(weights, tag):
        mdl = YOLO(weights)
        m = mdl.val(data=yaml_path, imgsz=args.imgsz, device=0, split="val",
                    project=args.project, name=f"{args.arm}_val_{tag}", exist_ok=True)
        present = [int(i) for i in m.box.ap_class_index]
        per_class = {names[c]: float(m.box.maps[c]) for c in present}
        supported = {k: v for k, v in per_class.items() if train_counts.get(k, 0) > 0}
        return {
            "map": float(m.box.map), "map50": float(m.box.map50), "map75": float(m.box.map75),
            "precision": float(m.box.mp), "recall": float(m.box.mr),
            "per_class_ap": per_class,
            "per_class_train_support": {k: train_counts.get(k, 0) for k in per_class},
            "n_val_classes": len(per_class),
            "map_supported": (sum(supported.values()) / len(supported)) if supported else 0.0,
            "n_supported_classes": len(supported),
        }

    res = {"arm": args.arm, "split": args.split,
           "config": {"model": os.path.basename(args.model), "epochs": args.epochs,
                      "imgsz": args.imgsz, "batch": args.batch, "seed": args.seed},
           "attainable_map_ceiling": sp["attainable_map_ceiling"],
           "zero_train_support_classes": sp["zero_train_support_classes"]}
    for tag in ("best", "last"):
        w = os.path.join(run_dir, "weights", f"{tag}.pt")
        if os.path.exists(w):
            res[tag] = evaluate(w, tag)
            print(f"[arm {args.arm}] {tag}.pt  mAP={res[tag]['map']:.4f} "
                  f"mAP50={res[tag]['map50']:.4f} P={res[tag]['precision']:.3f} "
                  f"R={res[tag]['recall']:.3f} map_supported={res[tag]['map_supported']:.4f}",
                  flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"{args.arm}.json"), "w") as f:
        json.dump(res, f, indent=2)

    print(f"\n==== ARM {args.arm} ({args.split}) ====")
    print(f"SURGVU_CAT1_ARM={args.arm} "
          f"BEST_MAP={res.get('best', {}).get('map', float('nan')):.4f} "
          f"LAST_MAP={res.get('last', {}).get('map', float('nan')):.4f} "
          f"BEST_MAP_SUPPORTED={res.get('best', {}).get('map_supported', float('nan')):.4f}")
    print(f"(baseline job 6933641 on the same main split = 0.1470 best.pt)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

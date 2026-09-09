"""Cat-1 weakly-supervised detector (autonomous ladder, lever 3).

There is NO 2024 bbox ground truth. The recognised SurgToolLoc/SurgVU recipe is:
  tool-PRESENCE spine -> Grad-CAM -> connected-component PSEUDO-boxes
  -> train a YOLO detector on the pseudo-boxes
  -> (later) refine on the small 2026 bbox val set when it lands.

This script:
  1) builds a YOLO dataset from cam_pseudoboxes.json (frame PNGs + per-frame boxes),
     with a case-disjoint train/val split (so val frames are from UNSEEN cases);
  2) trains ultralytics YOLO (from the cached yolo11n.pt -- NO download needed) on
     the pseudo-boxes;
  3) reports COCO mAP@[.5:.95] / mAP50 on the held-out pseudo-box val split.

IMPORTANT HONESTY NOTE (logged): with no real boxes, this mAP is *self-consistency*
on the spine's own pseudo-labels -- it measures whether a detector can REPRODUCE
the Grad-CAM localisation, i.e. the ceiling a weakly-sup detector can reach from
this supervision. It is NOT mAP vs human boxes (impossible until the 2026 bbox val
set lands). Reported as such; the real number replaces it once bbox GT is available.

Run (SLURM gpu node):
  python train_detector.py --pseudo .../cam_vis/pseudo_boxes.json \
      --img_dir .../cam_vis --epochs 40
"""
import argparse
import json
import os
import random
import shutil

from train_spine import TOOLS

CACHED_YOLO = "/scratch/USERNAME/AILET/from_pc/weights/yolo11n.pt"


def build_yolo_dataset(pseudo_json, img_dir, out_root, val_frac=0.2, seed=0):
    """Convert pseudo_boxes.json (list of {frame, case, boxes:[{class_id,xywh_norm}]})
    into a YOLO-format dataset, case-disjoint train/val. cam_pseudoboxes writes
    cam_{i:03d}.png next to the json; xywh_norm is TOP-LEFT x,y + w,h normalized."""
    data = json.load(open(pseudo_json))
    # case-disjoint split
    cases = sorted({d["case"] for d in data})
    rng = random.Random(seed)
    rng.shuffle(cases)
    n_val = max(1, int(len(cases) * val_frac))
    val_cases = set(cases[:n_val])

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(out_root, sub), exist_ok=True)

    n_tr = n_va = n_box = 0
    for d in data:
        if not d.get("boxes"):
            continue  # only frames with at least one pseudo-box are useful supervision
        split = "val" if d["case"] in val_cases else "train"
        src = os.path.join(img_dir, f"cam_{d['frame']:03d}.png")
        if not os.path.exists(src):
            continue
        stem = f"f{d['frame']:05d}"
        shutil.copy(src, os.path.join(out_root, f"images/{split}", stem + ".png"))
        lines = []
        for b in d["boxes"]:
            x, y, w, h = b["xywh_norm"]
            # YOLO wants CENTER x,y + w,h (normalized). pseudo is top-left x,y.
            cx, cy = x + w / 2.0, y + h / 2.0
            cx = min(max(cx, 0.0), 1.0); cy = min(max(cy, 0.0), 1.0)
            w = min(w, 1.0); h = min(h, 1.0)
            lines.append(f"{b['class_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            n_box += 1
        with open(os.path.join(out_root, f"labels/{split}", stem + ".txt"), "w") as fh:
            fh.write("\n".join(lines))
        if split == "train":
            n_tr += 1
        else:
            n_va += 1

    yaml_path = os.path.join(out_root, "data.yaml")
    with open(yaml_path, "w") as fh:
        fh.write(f"path: {out_root}\ntrain: images/train\nval: images/val\n")
        fh.write(f"nc: {len(TOOLS)}\nnames: {TOOLS}\n")
    print(f"# YOLO dataset: train={n_tr} val={n_va} frames, {n_box} pseudo-boxes, "
          f"{len(val_cases)} val cases (case-disjoint)")
    return yaml_path, n_tr, n_va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pseudo", default="/scratch/USERNAME/miccai-2026/SurgVU/models/cam_vis/pseudo_boxes.json")
    ap.add_argument("--img_dir", default="/scratch/USERNAME/miccai-2026/SurgVU/models/cam_vis")
    ap.add_argument("--out_root", default="/scratch/USERNAME/miccai-2026/SurgVU/models/detector")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--weights", default=CACHED_YOLO)
    # Optional augmentation knobs (lever 2). Defaults = ultralytics defaults so
    # that runs which DON'T pass these reproduce the ladder8 baseline exactly.
    ap.add_argument("--mosaic", type=float, default=None)
    ap.add_argument("--mixup", type=float, default=None)
    ap.add_argument("--hsv_h", type=float, default=None)
    ap.add_argument("--hsv_s", type=float, default=None)
    ap.add_argument("--hsv_v", type=float, default=None)
    ap.add_argument("--degrees", type=float, default=None)
    ap.add_argument("--translate", type=float, default=None)
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--fliplr", type=float, default=None)
    ap.add_argument("--patience", type=int, default=15)
    args = ap.parse_args()

    if not os.path.exists(args.pseudo):
        raise SystemExit(f"pseudo-box json missing: {args.pseudo} "
                         "(run cam_pseudoboxes.py first)")
    os.makedirs(args.out_root, exist_ok=True)
    ds_root = os.path.join(args.out_root, "yolo_ds")
    if os.path.exists(ds_root):
        shutil.rmtree(ds_root)
    yaml_path, n_tr, n_va = build_yolo_dataset(args.pseudo, args.img_dir, ds_root)
    if n_tr == 0 or n_va == 0:
        print("DETECTOR_mAP50_95=nan  (insufficient pseudo-boxes for a train/val split)")
        raise SystemExit("not enough pseudo-boxes to train a detector")

    from ultralytics import YOLO
    weights = args.weights if os.path.exists(args.weights) else "yolo11n.pt"
    print(f"# training YOLO from {weights}")
    model = YOLO(weights)
    # only pass aug overrides that were explicitly set (None => ultralytics default)
    aug = {k: v for k, v in {
        "mosaic": args.mosaic, "mixup": args.mixup,
        "hsv_h": args.hsv_h, "hsv_s": args.hsv_s, "hsv_v": args.hsv_v,
        "degrees": args.degrees, "translate": args.translate,
        "scale": args.scale, "fliplr": args.fliplr,
    }.items() if v is not None}
    if aug:
        print(f"# augmentation overrides: {aug}")
    model.train(data=yaml_path, epochs=args.epochs, imgsz=args.imgsz, batch=args.bs,
                project=args.out_root, name="train", exist_ok=True, verbose=True,
                patience=args.patience, device=0, **aug)
    metrics = model.val(data=yaml_path, project=args.out_root, name="val",
                        exist_ok=True, device=0)
    m5095 = float(metrics.box.map)     # mAP@[.5:.95]
    m50 = float(metrics.box.map50)
    print(f"\n# Cat-1 weakly-sup detector (pseudo-box self-consistency, case-disjoint val)")
    print(f"# mAP@[.5:.95]={m5095:.4f}  mAP50={m50:.4f}")
    print(f"# NOTE: self-consistency vs Grad-CAM pseudo-boxes (NO human bbox GT yet);")
    print(f"#       replace with real mAP once the 2026 bbox val set lands.")
    with open(os.path.join(args.out_root, "detector_metrics.json"), "w") as fh:
        json.dump({"map50_95": m5095, "map50": m50, "n_train": n_tr, "n_val": n_va,
                   "note": "pseudo-box self-consistency, no human GT"}, fh, indent=2)
    print(f"DETECTOR_mAP50_95={m5095:.4f}")
    print(f"DETECTOR_mAP50={m50:.4f}")


if __name__ == "__main__":
    main()

"""
SurgVU Cat-1 — train a detector on REAL HUMAN BOXES (video-disjoint), score real COCO mAP.

WHY THIS EXISTS
---------------
Every Cat-1 number in the dossier (0.7746 -> 0.9788) was self-consistency against the
pipeline's OWN Grad-CAM pseudo-boxes. Measured against the real human COCO GT in
`data/external/cat1_test_set/` those detectors score **mAP@[.5:.95] = 0.000**, and the
mechanism is arithmetic, not a bug:

    human GT box area   = 0.0215 of frame (median 0.0141), ~2.19 boxes/frame
    our Grad-CAM boxes  = 0.1538 of frame   -> 7.2x too large by area (2.7x per side)
    => a PERFECTLY CENTRED pseudo-box caps at IoU ~ 0.140, far below the 0.5 floor
       where AP@[.5:.95] starts. mAP is forced to ~0 regardless of classification.
    (the seg-cascade at 0.3404 is worse still: IoU cap ~ 0.063)

So the bottleneck is LOCALIZATION GRANULARITY: the target is the tool *clevis*, ~1.4% of
the frame, and weak Grad-CAM supervision cannot resolve it. This is a supervision-quality
problem, and it is exactly the one the challenge's own design fixes -- 2026 Cat-1 provides
"bounding box labels ... in the small validation set". This script is the dry run of that
recipe on the data we already hold.

DESIGN
------
7 annotated videos. Train on 5, hold out 2 (VIDEO-DISJOINT -- no frame from a validation
video is ever trained on). Report real COCO mAP@[.5:.95] on the held-out videos, plus the
weak-supervision baseline for the same held-out split so the comparison is like-for-like.
"""

import argparse
import json
import os
import shutil
import sys

import cv2
import numpy as np

EXT = "/scratch/sc20osc/miccai-2026/SurgVU/data/external/cat1_test_set"
VIDEOS = [1, 2, 3, 4, 5, 6, 7]


def norm(s):
    return s.lower().replace("-", " ").replace("/", " ").replace("_", " ").strip().replace(" ", "_")


def load_video_coco(v):
    return json.load(open(f"{EXT}/{v}_fps1_coco.json"))


def build_class_list():
    c = load_video_coco(VIDEOS[0])
    cats = sorted(c["categories"], key=lambda x: x["id"])
    return [norm(x["name"]) for x in cats], {x["id"]: i for i, x in enumerate(cats)}


def export_split(videos, split, root, names, catid_to_idx, stride=1):
    """Dump frames + YOLO labels for the given videos into <root>/{images,labels}/<split>."""
    imdir = os.path.join(root, "images", split)
    lbdir = os.path.join(root, "labels", split)
    os.makedirs(imdir, exist_ok=True)
    os.makedirs(lbdir, exist_ok=True)
    n_img = n_box = 0
    for v in videos:
        c = load_video_coco(v)
        meta = {i["id"]: i for i in c["images"]}
        anns = {}
        for a in c["annotations"]:
            anns.setdefault(a["image_id"], []).append(a)
        want = {int(os.path.splitext(meta[i]["file_name"])[0]): i for i in meta}
        keep = sorted(want)[::stride]
        keepset = {want[k] for k in keep}

        cap = cv2.VideoCapture(f"{EXT}/{v}_fps1.mp4")
        fidx, maxf = 0, max(want) if want else -1
        while True:
            ok, frame = cap.read()
            if not ok or fidx > maxf:
                break
            iid = want.get(fidx)
            if iid is not None and iid in keepset:
                im = meta[iid]
                W, H = im["width"], im["height"]
                fr = frame if (frame.shape[1] == W and frame.shape[0] == H) else \
                    cv2.resize(frame, (W, H))
                stem = f"v{v}_{fidx:08d}"
                cv2.imwrite(os.path.join(imdir, stem + ".jpg"), fr)
                lines = []
                for a in anns.get(iid, []):
                    x, y, w, h = a["bbox"]
                    cx, cy = (x + w / 2) / W, (y + h / 2) / H
                    lines.append(f"{catid_to_idx[a['category_id']]} {cx:.6f} {cy:.6f} "
                                 f"{w / W:.6f} {h / H:.6f}")
                with open(os.path.join(lbdir, stem + ".txt"), "w") as f:
                    f.write("\n".join(lines))
                n_img += 1
                n_box += len(lines)
            fidx += 1
        cap.release()
    return n_img, n_box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_videos", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--val_videos", type=int, nargs="+", default=[6, 7])
    ap.add_argument("--model", default="/scratch/sc20osc/miccai-2026/SurgVU/models/yolo26m.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--root", default="/scratch/sc20osc/miccai-2026/SurgVU/data/cat1_realbox")
    ap.add_argument("--out", default="/scratch/sc20osc/miccai-2026/SurgVU/logs/cat1_realbox.json")
    ap.add_argument("--project", default="/scratch/sc20osc/miccai-2026/SurgVU/models/cat1_realbox")
    args = ap.parse_args()

    assert not (set(args.train_videos) & set(args.val_videos)), "train/val videos must be disjoint"

    names, catid_to_idx = build_class_list()
    print(f"[classes] {len(names)}: {names}", flush=True)

    if os.path.isdir(args.root):
        shutil.rmtree(args.root)
    ntr, btr = export_split(args.train_videos, "train", args.root, names, catid_to_idx, args.stride)
    nva, bva = export_split(args.val_videos, "val", args.root, names, catid_to_idx, 1)
    print(f"[data] train {ntr} imgs / {btr} boxes (videos {args.train_videos})", flush=True)
    print(f"[data] val   {nva} imgs / {bva} boxes (videos {args.val_videos})  VIDEO-DISJOINT",
          flush=True)

    yaml_path = os.path.join(args.root, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {args.root}\ntrain: images/train\nval: images/val\n")
        f.write(f"nc: {len(names)}\nnames: {json.dumps(names)}\n")

    from ultralytics import YOLO
    model = YOLO(args.model)
    model.train(data=yaml_path, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                project=args.project, name="train", exist_ok=True, pretrained=True,
                verbose=True, device=0)

    m = model.val(data=yaml_path, imgsz=args.imgsz, device=0, split="val")
    res = {
        "train_videos": args.train_videos, "val_videos": args.val_videos,
        "n_train_images": ntr, "n_train_boxes": btr,
        "n_val_images": nva, "n_val_boxes": bva,
        "mAP": float(m.box.map), "mAP50": float(m.box.map50), "mAP75": float(m.box.map75),
        "per_class_mAP": {names[i]: float(x) for i, x in enumerate(m.box.maps)}
            if hasattr(m.box, "maps") else {},
        "note": "REAL human COCO boxes, VIDEO-DISJOINT split. Directly comparable to the "
                "grand-challenge Cat-1 metric. Weak Grad-CAM pipeline scores 0.000 here.",
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)

    print("\n==== Cat-1 REAL-BOX result (video-disjoint) ====")
    print(f"mAP@[.5:.95] = {res['mAP']:.4f} | mAP50 = {res['mAP50']:.4f} "
          f"| mAP75 = {res['mAP75']:.4f}")
    print(f"SURGVU_CAT1_REALBOX_MAP={res['mAP']:.4f}")
    print(f"(weak Grad-CAM supervision on the same real GT = 0.0000)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

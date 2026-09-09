#!/usr/bin/env python
"""HONEST Cat-1 validation against the REAL public 2024 SurgVU Cat-1 test set.

Until now Cat-1 detector mAP was *self-consistency* vs the spine's own Grad-CAM
pseudo-boxes (no human GT) -> 0.7746-0.9788, all UNVALIDATED. The public
`cat1_test_set_public.zip` (7 held-out videos at 1 fps, 640x512) ships HUMAN
COCO bbox annotations (`*_coco.json`) in the SAME 14-tool taxonomy. This script
runs a trained YOLO detector on those frames and computes the REAL COCO
mAP@[.5:.95] (the exact grand-challenge Cat-1 metric) against human boxes.

Same-domain, held-out, never-trained-on -> this is the first honest Cat-1 number.

Usage (SLURM gpu node):
  python cat1_honest_val.py --weights <best.pt> --out <results.json>
"""
import argparse, json, os, sys
import numpy as np
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

EXT = "/scratch/USERNAME/miccai-2026/SurgVU/data/external/cat1_test_set"
VIDEOS = [1, 2, 3, 4, 5, 6, 7]

# detector class order == train_spine.TOOLS == data.yaml names (underscored)
DET_NAMES = ['needle_driver', 'monopolar_curved_scissor', 'force_bipolar', 'clip_applier',
             'tip_up_fenestrated_grasper', 'cadiere_forceps', 'bipolar_forceps', 'vessel_sealer',
             'suction_irrigator', 'bipolar_dissector', 'prograsp_forceps', 'stapler',
             'permanent_cautery_hook_spatula', 'grasping_retractor']


def norm(s):
    return s.lower().replace('-', ' ').replace('/', ' ').replace('_', ' ').strip().replace(' ', '_')


def build_merged_gt():
    """Concatenate the 7 per-video COCO files into ONE, offsetting image ids.
    Returns (merged_coco_dict, coco_catname_by_id, img_meta) where img_meta maps
    merged_image_id -> (video, frame_index)."""
    merged = {"images": [], "annotations": [], "categories": None}
    catname_by_id = None
    img_meta = {}
    ann_id = 0
    for v in VIDEOS:
        c = json.load(open(f"{EXT}/{v}_fps1_coco.json"))
        if merged["categories"] is None:
            merged["categories"] = c["categories"]
            catname_by_id = {cat["id"]: cat["name"] for cat in c["categories"]}
        off = v * 1_000_000
        for im in c["images"]:
            gid = off + im["id"]
            frame_idx = int(os.path.splitext(im["file_name"])[0])
            merged["images"].append({"id": gid, "width": im["width"], "height": im["height"],
                                     "file_name": f"v{v}_{im['file_name']}"})
            img_meta[gid] = (v, frame_idx, im["id"])
        for a in c["annotations"]:
            a2 = dict(a)
            a2["image_id"] = off + a["image_id"]
            a2["id"] = ann_id
            ann_id += 1
            merged["annotations"].append(a2)
    return merged, catname_by_id, img_meta


def det_to_coco_catid(catname_by_id):
    """Map detector class index -> COCO category id by normalized name."""
    coco_by_norm = {norm(n): cid for cid, n in catname_by_id.items()}
    mapping = {}
    for di, dn in enumerate(DET_NAMES):
        key = norm(dn)
        if key not in coco_by_norm:
            print(f"[WARN] detector class '{dn}' ({key}) has no COCO match", file=sys.stderr)
            continue
        mapping[di] = coco_by_norm[key]
    assert len(mapping) == 14, f"only mapped {len(mapping)}/14 classes: {mapping}"
    return mapping


def run(weights, conf, imgsz, out):
    from ultralytics import YOLO
    merged, catname_by_id, img_meta = build_merged_gt()
    det2coco = det_to_coco_catid(catname_by_id)
    print(f"[map] det->coco: {det2coco}")

    # wanted frames per video
    want = {}
    for gid, (v, fidx, oid) in img_meta.items():
        want.setdefault(v, {})[fidx] = gid

    model = YOLO(weights)
    dets = []
    for v in VIDEOS:
        cap = cv2.VideoCapture(f"{EXT}/{v}_fps1.mp4")
        wmap = want[v]
        maxf = max(wmap) if wmap else -1
        fidx = 0
        nrun = 0
        while True:
            ok, frame = cap.read()
            if not ok or fidx > maxf:
                break
            if fidx in wmap:
                gid = wmap[fidx]
                r = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False, device=0)[0]
                b = r.boxes
                if b is not None and len(b):
                    xyxy = b.xyxy.cpu().numpy()
                    cls = b.cls.cpu().numpy().astype(int)
                    sc = b.conf.cpu().numpy()
                    for (x1, y1, x2, y2), cl, s in zip(xyxy, cls, sc):
                        if cl not in det2coco:
                            continue
                        dets.append({"image_id": int(gid), "category_id": int(det2coco[cl]),
                                     "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                                     "score": float(s)})
                nrun += 1
            fidx += 1
        cap.release()
        print(f"[video {v}] scored {nrun} annotated frames")

    # write merged GT + dets to disk for pycocotools
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    gt_path = out.replace(".json", "_gt.json")
    dt_path = out.replace(".json", "_dt.json")
    json.dump(merged, open(gt_path, "w"))
    json.dump(dets, open(dt_path, "w"))

    cocoGt = COCO(gt_path)
    cocoDt = cocoGt.loadRes(dt_path) if dets else cocoGt.loadRes([])
    E = COCOeval(cocoGt, cocoDt, "bbox")
    E.evaluate(); E.accumulate(); E.summarize()
    map5095, map50, map75 = float(E.stats[0]), float(E.stats[1]), float(E.stats[2])

    # per-class AP@[.5:.95]
    per_class = {}
    precision = E.eval["precision"]  # [T,R,K,A,M]
    cat_ids = cocoGt.getCatIds()
    for ki, cid in enumerate(cat_ids):
        p = precision[:, :, ki, 0, -1]
        p = p[p > -1]
        per_class[catname_by_id[cid]] = float(p.mean()) if p.size else float("nan")

    # per-video breakdown (rerun eval restricted to each video's image ids)
    per_video = {}
    for v in VIDEOS:
        ids = [gid for gid, (vv, _, _) in img_meta.items() if vv == v]
        Ev = COCOeval(cocoGt, cocoDt, "bbox")
        Ev.params.imgIds = ids
        Ev.evaluate(); Ev.accumulate(); Ev.summarize()
        per_video[v] = {"map5095": float(Ev.stats[0]), "map50": float(Ev.stats[1])}

    res = {"weights": weights, "conf": conf, "imgsz": imgsz,
           "n_images": len(merged["images"]), "n_gt_ann": len(merged["annotations"]),
           "n_det": len(dets),
           "map5095": map5095, "map50": map50, "map75": map75,
           "per_class_ap5095": per_class, "per_video": per_video,
           "note": "REAL human-bbox COCO mAP on public 2024 Cat-1 test set (7 held-out videos)"}
    json.dump(res, open(out, "w"), indent=2)
    print("\n==== HONEST Cat-1 result ====")
    print(f"mAP@[.5:.95] = {map5095:.4f} | mAP50 = {map50:.4f} | mAP75 = {map75:.4f}")
    print(f"saved -> {out}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/scratch/USERNAME/miccai-2026/SurgVU/models/detector_thr0.7_yolo26m_BEST_long120aug/train/weights/best.pt")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="/scratch/USERNAME/miccai-2026/SurgVU/logs/cat1_honest_val.json")
    a = ap.parse_args()
    run(a.weights, a.conf, a.imgsz, a.out)

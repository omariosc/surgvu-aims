"""Cat-1 stage (a): SAM2-prompted SEG pseudo-boxes (HRI_MV #1 cascade, arXiv 2305.07152).

UPGRADE (2026-06-22): the instance-segmentation step is now **SAM2** (the lever the
HRI_MV SurgToolLoc winners actually used), prompted from saliency peaks, replacing the
generic-COCO yolo*-seg.pt step. SAM2 returns TIGHTER single-instance masks than a
COCO-class detector that has never seen a surgical instrument -> tighter pseudo-boxes.

Replaces the coarse Grad-CAM connected-component blobs with TIGHT, single-instance,
clevis-centred boxes derived from an instance-segmentation model, CONDITIONED to the
tool classes the presence-spine predicts present in that frame.

Recipe (in-hand substitute for HRI_MV's dVRK/EndoVis17-18 surgical seg models, which we
have NO data/weights for on disk -> SAM2 is the class-agnostic stand-in):
  1. spine (train_spine.Spine) -> tool-PRESENCE set for the frame (the ONLY label we have);
  2. derive POINT PROMPTS from metal-saliency: the top-K bright/high-saturation foreground
     peaks inside the central FOV (bright instrument metal vs the dark UI border) where
     K = #spine-present tools (cap <=3); prompt SAM2 (`points=`/`labels=`, one positive
     point per candidate instrument location). Fall back to a coarse saliency BOX prompt
     when point prompts yield no mask;
  3. SAM2 returns tight instance masks -> mask_to_box -> tight boxes. ASSIGN masks to
     spine-predicted tool classes:
       - rank SAM2 masks by a "tool-likeness" score (elongation toward image centre +
         foreground saliency vs. the dark UI border = a clevis/instrument prior);
       - assign the top-K masks (K = #spine-present tools, capped <=3 per SurgVU) to the
         spine's present tools, highest-confidence tool first;
       - if SAM2 yields NO usable mask for a present tool, FALL BACK to a centred
         foreground/clevis-region box from saliency_box (so every present tool still gets
         a box -> coverage parity with Grad-CAM).
  4. emit `seg_pseudo_boxes.json` in the SAME schema as cam_pseudoboxes.py
     ([{frame, case, boxes:[{class, class_id, xywh_norm(top-left), score, src}]}]) AND write
     the SAME cam_{i:03d}.png frame images train_detector.py reads -> drops straight in.

The HYPOTHESIS this tests: SAM2 masks give TIGHTER, SINGLE-INSTANCE, clevis-centred boxes
than Grad-CAM's 15%-of-frame multi-instance blobs (2.71 boxes/frame, area 0.1538) AND
than the old generic-COCO-seg substitute (area 0.2180). Box stats (count, area,
instances/frame) are printed for the Grad-CAM / generic-COCO / SAM2 quality compare.

LICENSE NOTE: SAM2 (facebook/sam2.1-hiera-large, Apache-2.0) is used OFFLINE to GENERATE
pseudo-boxes only; the DETECTOR trained on them (train_detector.py) is the deliverable.
No YOLO-World / GroundingDINO (CC-BY-NC) is used anywhere.

Usage (full gen goes to sbatch; few-frame login smoke OK):
  python seg_pseudoboxes.py --ckpt .../spine_full/spine_best.pt \
      --manifest .../frame_manifest_full.csv --out_dir .../seg_vis_sam2 --n 600
"""
import argparse
import csv
import json
import os
import random

import numpy as np
import torch
import cv2

from train_spine import Spine, FrameDS, TOOLS

# Verified-working offline load path (ultralytics keys SAM by FILENAME -> symlink to a
# canonical sam2.1_l.pt name pointing at the HF sam2.1_hiera_large.pt checkpoint).
SAM_WEIGHTS = "/scratch/sc20osc/miccai-2026/SurgVU/models/sam_weights/sam2.1_l.pt"


def mask_to_box(mask):
    """tight top-left xywh (pixel) from a binary instance mask, plus area frac."""
    ys, xs = np.where(mask > 0.5)
    if xs.size == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)


def tool_likeness(box, H, W):
    """Prior that a seg instance is a surgical INSTRUMENT rather than background:
    - reward boxes whose centre is in the surgical FOV (not stuck on the dark border);
    - reward moderate size (instruments are not the whole frame, not a speck);
    - reward elongation (instrument shafts/clevises are elongated).
    Returns a scalar in ~[0, 2]; higher = more instrument-like."""
    x, y, w, h = box
    cx, cy = (x + w / 2) / W, (y + h / 2) / H
    centre = 1.0 - (abs(cx - 0.5) + abs(cy - 0.5))            # 1 at centre -> 0 at corner
    area = (w * h) / (H * W)
    size = float(np.exp(-((area - 0.12) ** 2) / (2 * 0.10 ** 2)))  # peak ~12% of frame
    elong = max(w, h) / (min(w, h) + 1e-6)
    elong = min(elong / 3.0, 1.0)                            # cap, instruments are elongated
    return centre + 0.5 * size + 0.5 * elong


def saliency_box(vis_bgr):
    """Fallback clevis/foreground box when seg gives no instance for a present tool:
    the brightest-saturation foreground component inside the central FOV (instruments
    are bright metal vs the dark UI border + duller tissue), then TIGHTENED toward its
    high-saliency core so the fallback box approximates a clevis, not the whole blob."""
    g = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2GRAY)
    H, W = g.shape
    thr = max(60, int(np.percentile(g, 75)))           # higher pct -> brighter (metal) only
    m = (g > thr).astype(np.uint8)
    bm = np.zeros_like(m)
    by, bx = int(H * 0.10), int(W * 0.10)
    bm[by:H - by, bx:W - bx] = 1
    m = m * bm
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # drop specks
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    best, barea = None, 0
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if 0.005 * H * W < area < 0.35 * H * W and area > barea:
            best, barea = k, area
    if best is None:  # last resort: a centred default box
        return (int(W * 0.35), int(H * 0.35), int(W * 0.3), int(H * 0.3))
    # tighten: keep the pixels of the chosen component, take their 5-95 percentile extent
    ys, xs = np.where(lab == best)
    x0, x1 = int(np.percentile(xs, 5)), int(np.percentile(xs, 95))
    y0, y1 = int(np.percentile(ys, 5)), int(np.percentile(ys, 95))
    return (x0, y0, max(4, x1 - x0), max(4, y1 - y0))


def saliency_points(vis_bgr, k):
    """Up-to-k POSITIVE point prompts for SAM2 at the brightest-metal foreground peaks
    inside the central FOV (same metal/clevis saliency prior as saliency_box, but emits
    point locations to prompt SAM2 rather than a single box). Returns a list of (x, y)
    pixel coords (length 1..k); de-duplicates peaks that fall in the same blob so each
    point seeds a DISTINCT instrument instance."""
    g = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2GRAY)
    H, W = g.shape
    thr = max(60, int(np.percentile(g, 75)))           # brighter (metal) only
    m = (g > thr).astype(np.uint8)
    bm = np.zeros_like(m)
    by, bx = int(H * 0.10), int(W * 0.10)
    bm[by:H - by, bx:W - bx] = 1                        # drop the dark UI border
    m = m * bm
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    comps = []
    for c in range(1, n):
        x, y, w, h, area = stats[c]
        if 0.003 * H * W < area < 0.45 * H * W:
            comps.append((area, c))
    comps.sort(key=lambda t: -t[0])                     # largest metal blobs first
    pts = []
    for _, c in comps[:k]:
        cy, cx = cent[c][1], cent[c][0]                 # centroid (a clevis-ish seed)
        pts.append((float(cx), float(cy)))
    if not pts:                                         # last resort: a centred seed
        pts.append((float(W * 0.5), float(H * 0.5)))
    return pts[:max(1, k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", default="/scratch/sc20osc/miccai-2026/SurgVU/models/seg_vis_sam2")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--sam_weights", default=SAM_WEIGHTS)  # verified offline SAM2 load path
    ap.add_argument("--seg_conf", type=float, default=0.05)  # ignored (SAM2 is prompt-driven); kept for CLI byte-compat
    ap.add_argument("--max_tools", type=int, default=3)      # SurgVU: <=3 tools/frame
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = Spine(ck["args"]["backbone"]).to(device).eval()
    model.load_state_dict(ck["model"])
    tool_thr = ck.get("tool_thr", 0.5)
    img_size = ck["args"]["img_size"]

    from ultralytics import SAM
    sam = SAM(args.sam_weights)  # facebook/sam2.1-hiera-large (Apache-2.0), offline

    rows = [r for r in csv.DictReader(open(args.manifest)) if r["video"]]
    random.Random(0).shuffle(rows)
    ds = FrameDS(rows, img_size, train=False)

    def _masks_from_result(r, H, W):
        """SAM2 result -> list of (box, tool_likeness) for usable instance masks."""
        boxes = []
        if r is None or r.masks is None:
            return boxes
        masks = r.masks.data.cpu().numpy()  # (n, h, w)
        mh, mw = masks.shape[1], masks.shape[2]
        for mk in masks:
            if (mh, mw) != (H, W):
                mk = cv2.resize(mk.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
            bb = mask_to_box(mk)
            if bb is None:
                continue
            bw, bh = bb[2], bb[3]
            if bw * bh >= 0.85 * H * W:   # reject a near-full-frame "whole-scene" mask
                continue
            boxes.append((bb, tool_likeness(bb, H, W)))
        return boxes

    out, n_box, n_seg_assigned, n_fallback = [], 0, 0, 0
    N = min(args.n, len(rows))
    for i in range(N):
        x, _, _ = ds[i]
        xb = x.unsqueeze(0).to(device)
        with torch.no_grad():
            lt, _ = model(xb)
            prob = torch.sigmoid(lt)[0]
        present = (prob > tool_thr).nonzero().flatten().tolist()
        present = sorted(present, key=lambda c: -float(prob[c]))[: args.max_tools]

        # the exact image the spine + detector see (same normalisation inverse as cam_pseudoboxes)
        vis = ((x.cpu().numpy().transpose(1, 2, 0) *
                np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
               ).clip(0, 255).astype(np.uint8)[:, :, ::-1].copy()  # RGB->BGR for cv2.imwrite
        H, W = vis.shape[:2]

        frame_boxes = []
        if present:
            K = len(present)
            pts = saliency_points(vis, K)           # up-to-K positive instrument seeds
            seg_boxes = []
            # PRIMARY: point-prompt SAM2 (one positive point per candidate instrument).
            try:
                pp = [[float(px), float(py)] for (px, py) in pts]
                ll = [1] * len(pp)
                r = sam.predict(vis, points=pp, labels=ll, verbose=False, device=device)[0]
                seg_boxes = _masks_from_result(r, H, W)
            except Exception:
                seg_boxes = []
            # FALLBACK: box-prompt SAM2 from a coarse saliency box if points gave no mask.
            if not seg_boxes:
                try:
                    bx0, by0, bw0, bh0 = saliency_box(vis)
                    bbox = [float(bx0), float(by0), float(bx0 + bw0), float(by0 + bh0)]
                    r = sam.predict(vis, bboxes=[bbox], verbose=False, device=device)[0]
                    seg_boxes = _masks_from_result(r, H, W)
                except Exception:
                    seg_boxes = []
            # rank SAM2 instances by instrument-likeness, assign best to highest-conf tools
            seg_boxes.sort(key=lambda t: -t[1])
            for j, c in enumerate(present):
                if j < len(seg_boxes):
                    (bx, by, bw, bh), score = seg_boxes[j]
                    src = "seg"
                    n_seg_assigned += 1
                else:
                    bx, by, bw, bh = saliency_box(vis)
                    score = 0.30
                    src = "fallback"
                    n_fallback += 1
                frame_boxes.append({
                    "class": TOOLS[c], "class_id": int(c),
                    "xywh_norm": [bx / W, by / H, bw / W, bh / H],
                    "score": float(min(score, 1.0)), "src": src,
                })
                n_box += 1
        cv2.imwrite(os.path.join(args.out_dir, f"cam_{i:03d}.png"), vis)
        out.append({"frame": i, "case": rows[i]["case"], "boxes": frame_boxes})

    json.dump(out, open(os.path.join(args.out_dir, "seg_pseudo_boxes.json"), "w"), indent=2)

    # quality stats for the Grad-CAM-vs-seg pseudo-box comparison
    nb = [len(o["boxes"]) for o in out]
    areas = [b["xywh_norm"][2] * b["xywh_norm"][3] for o in out for b in o["boxes"]]
    print(f"wrote {len(out)} frames -> {args.out_dir}/seg_pseudo_boxes.json")
    print(f"SEG_total_boxes={n_box}  sam2_assigned={n_seg_assigned}  fallback={n_fallback}")
    print(f"SEG_mean_boxes_per_frame={np.mean(nb):.3f}")
    print(f"SEG_mean_box_area_frac={np.mean(areas):.4f}" if areas else "SEG_mean_box_area_frac=nan")
    print(f"# refs: Grad-CAM 2.71 boxes/frame area 0.1538 | generic-COCO-seg area 0.2180 "
          f"-> SAM2 should be TIGHTER (area < 0.1538 = beats Grad-CAM)")


if __name__ == "__main__":
    main()

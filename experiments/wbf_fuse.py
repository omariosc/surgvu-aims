"""Cat-1 stage (d): detection + tracking FUSION via Weighted-Boxes-Fusion (HRI_MV #1).

The +0.009 mAP that WON SurgToolLoc came from fusing the trained detector's RAW
predictions with the tracker-propagated boxes. We reproduce that fusion step with WBF
(`ensemble_boxes.weighted_boxes_fusion`).

Pipeline:
  1. load a trained YOLO detector (from train_detector.py's run) + its case-disjoint
     YOLO val dataset (built from the SAME seg pseudo-boxes so the GT is the seg
     supervision -> self-consistency, NO human bbox GT);
  2. for each val image: get detector boxes AND tracker-propagated boxes (we re-run the
     seg+track proposer on the val image as the "tracking" source, matching HRI_MV's
     two-source fusion);
  3. WBF-fuse the two box sets per image;
  4. score detector-only vs WBF-fused mAP@[.5:.95] against the seg-pseudo-box GT labels
     -> report the FUSION DELTA (the winning margin was +0.009).

HONESTY: all mAP here is self-consistency vs the seg pseudo-box GT (NO human boxes for
2024). The number that matters is the *delta* WBF adds, and the pseudo-box QUALITY -
both are pseudo-box-relative until the gated 2026 bbox-val set lands.

Usage (sbatch):
  python wbf_fuse.py --weights .../detector_seg/train/weights/best.pt \
      --yolo_ds .../detector_seg/yolo_ds --ckpt .../spine_best.pt
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import cv2

from ensemble_boxes import weighted_boxes_fusion
from train_spine import Spine, TOOLS
from seg_pseudoboxes import mask_to_box, tool_likeness, saliency_box, SEG_WEIGHTS


def load_yolo_label(txt, W, H):
    """read a YOLO label file -> GT boxes in xyxy-norm + class ids."""
    boxes, labels = [], []
    if not os.path.exists(txt):
        return boxes, labels
    for ln in open(txt):
        p = ln.split()
        if len(p) != 5:
            continue
        c, cx, cy, w, h = int(p[0]), *map(float, p[1:])
        boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        labels.append(c)
    return boxes, labels


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def coco_map(preds, gts, n_cls):
    """minimal COCO mAP@[.5:.05:.95] over the val images.
    preds/gts: list per image of (boxes_xyxy_norm, labels, scores[preds only])."""
    threshs = np.arange(0.5, 1.0, 0.05)
    aps = []
    for c in range(n_cls):
        ap_t = []
        # gather this class
        n_gt = sum(int((np.array(g[1]) == c).sum()) for g in gts)
        if n_gt == 0:
            continue
        for th in threshs:
            scored = []  # (score, is_tp)
            for im, (pb, pl, ps) in enumerate(preds):
                gb = [b for b, l in zip(gts[im][0], gts[im][1]) if l == c]
                used = [False] * len(gb)
                order = np.argsort(-np.array(ps)) if ps else []
                for k in order:
                    if pl[k] != c:
                        continue
                    best, bj = th, -1
                    for j, g in enumerate(gb):
                        if used[j]:
                            continue
                        v = iou(pb[k], g)
                        if v >= best:
                            best, bj = v, j
                    if bj >= 0:
                        used[bj] = True
                        scored.append((ps[k], 1))
                    else:
                        scored.append((ps[k], 0))
            if not scored:
                ap_t.append(0.0)
                continue
            scored.sort(key=lambda t: -t[0])
            tp = np.cumsum([s[1] for s in scored])
            fp = np.cumsum([1 - s[1] for s in scored])
            rec = tp / (n_gt + 1e-9)
            prec = tp / (tp + fp + 1e-9)
            # 101-point interpolation
            ap = 0.0
            for rt in np.linspace(0, 1, 101):
                p = prec[rec >= rt].max() if np.any(rec >= rt) else 0.0
                ap += p / 101
            ap_t.append(ap)
        aps.append(np.mean(ap_t))
    return float(np.mean(aps)) if aps else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)            # trained detector best.pt
    ap.add_argument("--yolo_ds", required=True)            # train_detector's yolo_ds (val split)
    ap.add_argument("--ckpt", required=True)               # spine for the tracking source
    ap.add_argument("--out", default="/scratch/USERNAME/miccai-2026/SurgVU/models/wbf_fusion.json")
    ap.add_argument("--seg_conf", type=float, default=0.10)
    ap.add_argument("--max_tools", type=int, default=3)
    ap.add_argument("--det_w", type=float, default=2.0)    # WBF weights: detector vs tracker
    ap.add_argument("--trk_w", type=float, default=1.0)
    ap.add_argument("--iou_thr", type=float, default=0.5)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    spine = Spine(ck["args"]["backbone"]).to(device).eval()
    spine.load_state_dict(ck["model"])
    tool_thr = ck.get("tool_thr", 0.5)
    img_size = ck["args"]["img_size"]

    from ultralytics import YOLO
    det = YOLO(args.weights)
    seg = YOLO(SEG_WEIGHTS)

    val_imgs = sorted(glob.glob(os.path.join(args.yolo_ds, "images/val", "*.png")))
    if not val_imgs:
        print("WBF_DELTA=nan  (no val images)")
        raise SystemExit("no val images in yolo_ds/images/val")

    gts, det_only, fused = [], [], []
    for ip in val_imgs:
        img = cv2.imread(ip)
        H, W = img.shape[:2]
        stem = os.path.splitext(os.path.basename(ip))[0]
        txt = os.path.join(args.yolo_ds, "labels/val", stem + ".txt")
        gb, gl = load_yolo_label(txt, W, H)
        gts.append((gb, gl))

        # --- detector boxes ---
        dr = det.predict(img, conf=0.05, verbose=False, device=device, imgsz=img_size)[0]
        db, dl, dscore = [], [], []
        if dr.boxes is not None and len(dr.boxes):
            xyxy = dr.boxes.xyxy.cpu().numpy()
            for (x0, y0, x1, y1), cl, sc in zip(xyxy, dr.boxes.cls.cpu().numpy(),
                                                dr.boxes.conf.cpu().numpy()):
                db.append([x0 / W, y0 / H, x1 / W, y1 / H])
                dl.append(int(cl))
                dscore.append(float(sc))
        det_only.append((db, dl, dscore))

        # --- tracker/seg-propagated boxes on the same val image (the 2nd fusion source) ---
        xt = torch.from_numpy(np.ascontiguousarray(
            ((cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
              - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
             ).transpose(2, 0, 1))).float()
        with torch.no_grad():
            lt, _ = spine(xt.unsqueeze(0).to(device))
            prob = torch.sigmoid(lt)[0]
        present = sorted((prob > tool_thr).nonzero().flatten().tolist(),
                         key=lambda c: -float(prob[c]))[: args.max_tools]
        tb, tl, tscore = [], [], []
        if present:
            sr = seg.predict(img, conf=args.seg_conf, verbose=False, device=device,
                             imgsz=img_size, retina_masks=True)[0]
            sboxes = []
            if sr.masks is not None:
                masks = sr.masks.data.cpu().numpy()
                mh, mw = masks.shape[1], masks.shape[2]
                for mk in masks:
                    if (mh, mw) != (H, W):
                        mk = cv2.resize(mk, (W, H), interpolation=cv2.INTER_NEAREST)
                    bb = mask_to_box(mk)
                    if bb:
                        sboxes.append((bb, tool_likeness(bb, H, W)))
            sboxes.sort(key=lambda t: -t[1])
            for j, c in enumerate(present):
                if j < len(sboxes):
                    (bx, by, bw, bh), score = sboxes[j]
                else:
                    bx, by, bw, bh = saliency_box(img)
                    score = 0.30
                tb.append([bx / W, by / H, (bx + bw) / W, (by + bh) / H])
                tl.append(int(c))
                tscore.append(float(min(score, 1.0)))

        # --- WBF fuse detector + tracker ---
        bl = [b for b in (db, tb)]
        sl = [s for s in (dscore, tscore)]
        ll = [l for l in (dl, tl)]
        nonempty = [(b, s, l, w) for b, s, l, w in
                    zip(bl, sl, ll, (args.det_w, args.trk_w)) if b]
        if nonempty:
            bb, ss, ll2, ww = zip(*nonempty)
            fb, fs, fl = weighted_boxes_fusion(
                list(bb), list(ss), list(ll2), weights=list(ww),
                iou_thr=args.iou_thr, skip_box_thr=0.0)
            fused.append(([list(x) for x in fb], [int(x) for x in fl], [float(x) for x in fs]))
        else:
            fused.append(([], [], []))

    m_det = coco_map(det_only, gts, len(TOOLS))
    m_fused = coco_map(fused, gts, len(TOOLS))
    delta = m_fused - m_det
    res = {"map_det_only": m_det, "map_wbf_fused": m_fused, "wbf_delta": delta,
           "n_val": len(val_imgs), "det_w": args.det_w, "trk_w": args.trk_w,
           "note": "self-consistency vs seg pseudo-box GT (NO human bbox GT)"}
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"# WBF fusion (self-consistency vs seg pseudo-boxes, case-disjoint val)")
    print(f"DET_ONLY_mAP={m_det:.4f}")
    print(f"WBF_FUSED_mAP={m_fused:.4f}")
    print(f"WBF_DELTA={delta:+.4f}  (HRI_MV winning margin was +0.009)")


if __name__ == "__main__":
    main()

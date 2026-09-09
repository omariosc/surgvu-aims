"""
SurgVU Cat-1 semi-supervised box proposal — shared helpers.

WHAT THIS MODULE ENCODES (all of it MEASURED on 2026-08-12, not assumed)
------------------------------------------------------------------------
1. GEOMETRY.  The annotated Cat-1 frames (640x512, 1 fps, videos 1-7 of
   `cat1_test_set`) are a deterministic crop+resize of the 1280x720/60fps
   training corpus:  corpus[0:720, 192:1088]  (896x720, ar 1.2444)  ->  640x512
   (ar 1.2500).  Verified on 14 randomly-drawn corpus parts: the content column
   range is x[192,1087] in EVERY one of them, and the da Vinci HUD arm-boxes land
   at identical x positions in both domains after the transform.  Without this
   crop the proposer would see a 1.78-aspect frame it was never trained on.

2. PRESENCE LABELS ARE STRONGER THAN "NOISY PER-CLIP TOOL PRESENCE".
   `tools.csv` carries per-ARM install/uninstall TIMESTAMPS plus
   `install_case_part`, so presence is resolvable to the FRAME, not the clip.
   Alignment verified: 8,833 of 8,885 rows fall inside their part's true decoded
   duration (2 overshoot, 50 unusable/nan) => `install_case_part` indexes
   `case_XXX_video_part_00P.mp4` and the times are that part's own clock.

3. PRESENCE IS A SUPERSET CONSTRAINT, NEVER A POSITIVE.  A tool that is
   *installed on an arm* need not be *inside the endoscope view* (confirmed by
   eye: annotated video 6 HUD lists Cadiere Forceps on arm 4, yet video 6 has
   zero cadiere_forceps GT boxes).  So presence can only ever be used to REJECT
   a proposal or to NAME an already-detected object -- never to assert that an
   object is there.  Both gating rules below respect that direction.

4. OUT-OF-VOCABULARY TOOLS EXIST.  25 intervals name tools outside the 14-class
   list (synchroseal 15, curved scissors 4, crocodile grasper 2, potts scissors
   2, tenaculum forceps 2) and 144 rows name nothing at all.  Any frame where one
   of those is installed is DISQUALIFIED from the singleton-assign rule, because
   "exactly one in-vocabulary tool installed" would be a lie there.
"""

import csv
import glob
import os
import re

import cv2
import numpy as np

# ---------------------------------------------------------------- paths
CORPUS = "/scratch/USERNAME/miccai-2026/SurgVU/data/surgvu24_videos/surgvu24"
LABELS = ("/scratch/USERNAME/miccai-2026/SurgVU/data/external/cat2_train_labels/"
          "SURGVU25_train_labels")
ANN_POOL = "/scratch/USERNAME/miccai-2026/SurgVU/data/cat1_pool"
SSL = "/scratch/USERNAME/miccai-2026/SurgVU/data/cat1_ssl"

# ---------------------------------------------------------------- geometry
CROP_X0, CROP_X1 = 192, 1088          # measured, uniform across 14 sampled parts
CROP_Y0, CROP_Y1 = 0, 720
OUT_W, OUT_H = 640, 512               # the annotated Cat-1 frame size

# ---------------------------------------------------------------- classes
# Order is the ultralytics class-index order of the annotated pool. DO NOT REORDER.
CLASSES = ['grasping_retractor', 'cadiere_forceps', 'bipolar_forceps', 'force_bipolar',
           'clip_applier', 'stapler', 'permanent_cautery_hook_spatula',
           'monopolar_curved_scissor', 'vessel_sealer', 'tip_up_fenestrated_grasper',
           'bipolar_dissector', 'needle_driver', 'prograsp_forceps', 'suction_irrigator']
CLS_IDX = {c: i for i, c in enumerate(CLASSES)}

# tools.csv `groundtruth_toolname` -> our class name
TOOLNAME_MAP = {
    'needle driver': 'needle_driver',
    'monopolar curved scissors': 'monopolar_curved_scissor',
    'clip applier': 'clip_applier',
    'bipolar forceps': 'bipolar_forceps',
    'cadiere forceps': 'cadiere_forceps',
    'prograsp forceps': 'prograsp_forceps',
    'stapler': 'stapler',
    'vessel sealer': 'vessel_sealer',
    'grasping retractor': 'grasping_retractor',
    'permanent cautery hook/spatula': 'permanent_cautery_hook_spatula',
    'force bipolar': 'force_bipolar',
    'suction irrigator': 'suction_irrigator',
    'tip-up fenestrated grasper': 'tip_up_fenestrated_grasper',
    'bipolar dissector': 'bipolar_dissector',
}
OOV = "__OOV__"     # an installed tool we cannot name -> disqualifies singleton frames


def crop_resize(frame):
    """corpus 1280x720 BGR frame -> the annotated Cat-1 640x512 geometry."""
    return cv2.resize(frame[CROP_Y0:CROP_Y1, CROP_X0:CROP_X1], (OUT_W, OUT_H),
                      interpolation=cv2.INTER_AREA)


HASH_BITS = 256
HASH_BYTES = HASH_BITS // 8


def dhash(img_bgr, size=16):
    """256-bit difference hash -> uint8[32].

    WHY 256 AND NOT THE USUAL 64.  Measured on this corpus: at 64 bits only 4,198
    of the 5,178 annotated frames hash uniquely (surgical video is near-static, so
    19% collide with each other), and an unrelated corpus frame sat at min-distance
    15 from the annotated set -- barely outside a leak threshold of 12.  A near-dup
    test with that little headroom cannot separate "same shot" from "same surgery
    looks alike", and it would either leak or ban the whole corpus.  256 bits
    restores the margin (see the calibrated null distribution in the scan log)."""
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (size + 1, size), interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return np.packbits(bits)


_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_to_set(h, table):
    """min Hamming distance from a uint8[32] hash to every row of table (N,32)."""
    if table is None or table.size == 0:
        return HASH_BITS
    x = np.bitwise_xor(table, h[None, :])
    return int(_POP[x].sum(axis=1).min())


def _t2s(s):
    s = (s or "").strip()
    m = re.match(r"^(\d+):(\d+):(\d+(?:\.\d+)?)$", s)
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else None


def load_intervals(part_duration, stats=None):
    """-> {(case, part): [(t0, t1, label), ...]}  label in CLASSES or OOV.

    Camera rows are dropped (the endoscope is never an object in its own view).

    TWO DEFECTS IN tools.csv THAT MUST BE HANDLED CONSERVATIVELY, both found by
    reading the raw rows on 2026-08-12:

      (a) 144 rows across 62 cases carry commercial_toolname "Unknown Instrument"
          and an EMPTY groundtruth_toolname.  Something IS installed on that arm.
          Dropping those rows would manufacture fake "exactly one tool installed"
          frames, so they are mapped to OOV, which DISQUALIFIES the frame from the
          singleton-assign rule.

      (b) A substantial number of rows have uninstall_case_time EARLIER than
          install_case_time (e.g. case_000: install 00:23:03, uninstall 00:06:57).
          The row is corrupt, not empty.  We extend it to the end of the part.
          That is the conservative direction in BOTH rules: for the singleton rule
          it keeps the arm "occupied" and so disqualifies frames rather than
          inventing clean ones; for the filter rule it only widens the allowed
          class set, which can never turn a rejected proposal into an accepted
          wrong one that presence would otherwise have caught.
    """
    out = {}
    for f in sorted(glob.glob(LABELS + "/*/tools.csv")):
        case = os.path.basename(os.path.dirname(f))
        with open(f) as fh:
            for r in csv.DictReader(fh):
                raw = (r.get("groundtruth_toolname") or "").strip().lower()
                if raw.startswith("nan(camera"):
                    continue
                try:
                    part = int(float(r["install_case_part"]))
                except (TypeError, ValueError):
                    continue
                key = (case, part)
                dur = part_duration.get(key)
                if dur is None:
                    continue
                t0 = _t2s(r["install_case_time"])
                if t0 is None:
                    continue
                t1 = _t2s(r["uninstall_case_time"])
                corrupt = (t1 is None or t1 <= t0)
                if corrupt:
                    t1 = dur                      # defect (b): extend, conservatively
                t1 = min(t1, dur)
                lab = TOOLNAME_MAP.get(raw, OOV)  # defect (a): unknown -> OOV
                if stats is not None:
                    stats["rows"] = stats.get("rows", 0) + 1
                    if corrupt:
                        stats["corrupt_uninstall"] = stats.get("corrupt_uninstall", 0) + 1
                    if lab is OOV:
                        stats["oov"] = stats.get("oov", 0) + 1
                out.setdefault(key, []).append((t0, t1, lab))
    return out


def installed_at(ivs, t):
    """-> (frozenset of in-vocab class names, oov_present: bool)"""
    act = {lab for a, b, lab in ivs if a <= t <= b}
    return frozenset(a for a in act if a != OOV), (OOV in act)


def part_key(path):
    b = os.path.basename(path)
    return b[:8], int(b.split("part_")[1][:3])


def probe_duration(path):
    cap = cv2.VideoCapture(path)
    n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return (n / fps) if fps > 0 else None, fps, n

"""SurgVU-2024 grounding-spine frame-label pipeline (Phase 1, lever 1).

Maps each case's ``tools.csv`` install/uninstall intervals + ``tasks.csv`` task
segments into a per-timestamp multi-label record:

    {video, case, t_sec, tools[14 multi-hot], task[8 one-hot or -1]}

then samples frames *inside* those intervals at a fixed stride, producing a
frame manifest (CSV) with a case-disjoint train/val split. Frames are decoded
later (extract_frames.py) only for sampled timestamps, so we never touch the
full >18M-frame corpus.

Label spaces (canonical, from the official Cat-1 submission template tool list):
  TOOLS (14): needle_driver, monopolar_curved_scissor, force_bipolar,
    clip_applier, tip_up_fenestrated_grasper, cadiere_forceps, bipolar_forceps,
    vessel_sealer, suction_irrigator, bipolar_dissector, prograsp_forceps,
    stapler, permanent_cautery_hook_spatula, grasping_retractor
  TASKS (8): Suturing, Uterine horn, Suspensory ligaments, Rectal artery/vein,
    Skills application, Range of motion, Retraction and collision avoidance, Other

Tool labels are noisy (robot-derived, <=3 tools/clip) -> weak labels. Non-canonical
rows ("nan(camera in)", "", and rare off-list tools) are dropped from the tool
target but never *suppress* a co-present canonical tool.

Usage:
  python build_frame_labels.py --labels_dir <...> --videos_dir <...> \
      --out_manifest <...> --stride_sec 2 --val_frac 0.15
The video dir is scanned to (a) confirm which cases have a local mp4 and (b)
read true duration. If a case has no local video yet, it is written to the
manifest with video="" (so training can scale up as the download completes by
just re-running with --require_video).
"""
import argparse
import csv
import glob
import os
import random
import re
from collections import defaultdict

# ---- canonical label spaces -------------------------------------------------
TOOLS = [
    "needle_driver", "monopolar_curved_scissor", "force_bipolar", "clip_applier",
    "tip_up_fenestrated_grasper", "cadiere_forceps", "bipolar_forceps",
    "vessel_sealer", "suction_irrigator", "bipolar_dissector", "prograsp_forceps",
    "stapler", "permanent_cautery_hook_spatula", "grasping_retractor",
]
TOOL_IDX = {t: i for i, t in enumerate(TOOLS)}

# map the noisy groundtruth_toolname surface forms -> canonical key
TOOL_ALIASES = {
    "needle driver": "needle_driver",
    "monopolar curved scissors": "monopolar_curved_scissor",
    "monopolar curved scissor": "monopolar_curved_scissor",
    "force bipolar": "force_bipolar",
    "clip applier": "clip_applier",
    "tip-up fenestrated grasper": "tip_up_fenestrated_grasper",
    "cadiere forceps": "cadiere_forceps",
    "bipolar forceps": "bipolar_forceps",
    "vessel sealer": "vessel_sealer",
    "suction irrigator": "suction_irrigator",
    "bipolar dissector": "bipolar_dissector",
    "prograsp forceps": "prograsp_forceps",
    "stapler": "stapler",
    "permanent cautery hook/spatula": "permanent_cautery_hook_spatula",
    "permanent cautery hook spatula": "permanent_cautery_hook_spatula",
    "grasping retractor": "grasping_retractor",
    # rare / off-canonical -> dropped (not in the 14): synchroseal,
    # tenaculum forceps, curved scissors, potts scissors, crocodile grasper,
    # "nan(camera in)", "" -> all return None below.
}

TASKS = [
    "Suturing", "Uterine horn", "Suspensory ligaments", "Rectal artery/vein",
    "Skills application", "Range of motion", "Retraction and collision avoidance",
    "Other",
]
TASK_IDX = {t.lower(): i for i, t in enumerate(TASKS)}
OTHER_IDX = TASK_IDX["other"]


def canon_tool(raw):
    if raw is None:
        return None
    k = raw.strip().lower()
    if not k or k.startswith("nan"):
        return None
    return TOOL_ALIASES.get(k)  # None for off-canonical


def canon_task(raw):
    if raw is None:
        return OTHER_IDX
    k = raw.strip().lower()
    return TASK_IDX.get(k, OTHER_IDX)


def hms_to_sec(s):
    """'HH:MM:SS.ffffff' -> float seconds. Returns None if unparseable."""
    s = (s or "").strip()
    m = re.match(r"^(\d+):(\d{2}):(\d{2}(?:\.\d+)?)$", s)
    if not m:
        try:
            return float(s)  # already seconds in some rows
        except ValueError:
            return None
    h, mi, se = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(se)


# -----------------------------------------------------------------------------
def _part(raw, default=1):
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


def load_tool_intervals(tools_csv):
    """Return list of (part, canon_tool_idx, start_sec, stop_sec). Times are
    PART-LOCAL seconds (videos are split into case_XXX_video_part_NNN.mp4).
    Drops rows with bad/absent times, non-canonical tools, stop<=start, or rows
    whose install/uninstall straddle different parts (a few corrupted rows)."""
    out = []
    with open(tools_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            ti = canon_tool(row.get("groundtruth_toolname"))
            if ti is None:
                continue
            p_in = _part(row.get("install_case_part"))
            p_out = _part(row.get("uninstall_case_part"), p_in)
            if p_in != p_out:
                continue  # cross-part row -> ambiguous timeline, skip
            s = hms_to_sec(row.get("install_case_time"))
            e = hms_to_sec(row.get("uninstall_case_time"))
            if s is None or e is None or e <= s:
                continue
            out.append((p_in, TOOL_IDX[ti], s, e))
    return out


def load_task_intervals(tasks_csv):
    """Return list of (part, task_idx, start_sec, stop_sec) (part-local seconds)."""
    out = []
    with open(tasks_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                s = float(row["start_time"]); e = float(row["stop_time"])
            except (KeyError, ValueError, TypeError):
                continue
            p_in = _part(row.get("start_part"))
            p_out = _part(row.get("stop_part"), p_in)
            if p_in != p_out or e <= s:
                continue
            out.append((p_in, canon_task(row.get("groundtruth_taskname")), s, e))
    return out


def tools_at(part, t, tool_intervals):
    vec = [0] * len(TOOLS)
    for p, idx, s, e in tool_intervals:
        if p == part and s <= t < e:
            vec[idx] = 1
    return vec


def task_at(part, t, task_intervals):
    for p, idx, s, e in task_intervals:
        if p == part and s <= t < e:
            return idx
    return OTHER_IDX  # unannotated span = Other


def find_video(videos_dir, case_id, part):
    """Locate case_XXX_video_part_NNN.mp4 for a given case + 1-based part.
    Video tree: <videos_dir>/surgvu24/case_XXX/case_XXX_video_part_NNN.mp4
    (also tolerant of a flat layout). Returns "" if not local yet."""
    if not videos_dir or not os.path.isdir(videos_dir):
        return ""
    pp = f"{part:03d}"
    pats = (
        f"{case_id}/{case_id}_video_part_{pp}.mp4",
        f"**/{case_id}/{case_id}_video_part_{pp}.mp4",
        f"**/{case_id}_video_part_{pp}.mp4",
        f"**/{case_id}*part*{pp}*.mp4",
        f"**/{case_id}*part*{part}*.mp4",
    )
    for pat in pats:
        hits = sorted(glob.glob(os.path.join(videos_dir, pat), recursive=True))
        if hits:
            return hits[0]
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_dir", default="/scratch/USERNAME/miccai-2026/SurgVU/data/surgvu24_labels/labels")
    ap.add_argument("--videos_dir", default="/scratch/USERNAME/miccai-2026/SurgVU/data/surgvu24_videos")
    ap.add_argument("--out_manifest", default="/scratch/USERNAME/miccai-2026/SurgVU/data/frame_manifest.csv")
    ap.add_argument("--stride_sec", type=float, default=2.0,
                    help="sample one frame every N seconds inside labelled spans")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--require_video", action="store_true",
                    help="only emit rows for cases whose mp4 is already local")
    args = ap.parse_args()

    case_dirs = sorted(glob.glob(os.path.join(args.labels_dir, "case_*")))
    case_dirs = [d for d in case_dirs if ".ipynb" not in d]
    cases = [os.path.basename(d) for d in case_dirs]

    # case-disjoint split (deterministic)
    rng = random.Random(args.seed)
    shuffled = cases[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * args.val_frac)))
    val_cases = set(shuffled[:n_val])

    rows = []
    stat_tool = [0] * len(TOOLS)
    stat_task = [0] * len(TASKS)
    n_local_vid = 0
    cases_with_frames = set()

    for cdir, cid in zip(case_dirs, cases):
        tools_csv = os.path.join(cdir, "tools.csv")
        tasks_csv = os.path.join(cdir, "tasks.csv")
        if not (os.path.exists(tools_csv) and os.path.exists(tasks_csv)):
            continue
        tool_iv = load_tool_intervals(tools_csv)
        task_iv = load_task_intervals(tasks_csv)
        if not task_iv and not tool_iv:
            continue
        split = "val" if cid in val_cases else "train"
        parts = sorted({p for p, *_ in tool_iv} | {p for p, *_ in task_iv})
        case_has_local = False
        for part in parts:
            vid = find_video(args.videos_dir, cid, part)
            if vid:
                case_has_local = True
            if args.require_video and not vid:
                continue
            p_tool = [(idx, s, e) for p, idx, s, e in tool_iv if p == part]
            p_task = [(idx, s, e) for p, idx, s, e in task_iv if p == part]
            extents = p_tool + p_task
            if not extents:
                continue
            t_min = min(s for _, s, _ in extents)
            t_max = max(e for _, _, e in extents)
            if t_max <= t_min:
                continue
            t = t_min
            while t < t_max:
                tv = tools_at(part, t, tool_iv)
                tk = task_at(part, t, task_iv)
                # keep a frame only if it carries SOME label signal (>=1 tool,
                # or a non-Other task). Pure-empty frames add weak-label noise.
                if sum(tv) > 0 or tk != OTHER_IDX:
                    rows.append((cid, part, vid, f"{t:.3f}", tk,
                                 "".join(map(str, tv)), split))
                    for i, b in enumerate(tv):
                        stat_tool[i] += b
                    stat_task[tk] += 1
                    cases_with_frames.add(cid)
                t += args.stride_sec
        if case_has_local:
            n_local_vid += 1

    out_dir = os.path.dirname(os.path.abspath(args.out_manifest))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out_manifest, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "part", "video", "t_sec", "task_idx", "tools_multihot", "split"])
        w.writerows(rows)

    n_train = sum(1 for r in rows if r[6] == "train")
    n_val_rows = len(rows) - n_train
    print(f"cases total={len(cases)}  with-labels-emitting-frames={len(cases_with_frames)}")
    print(f"local videos matched={n_local_vid}/{len(cases)}  (require_video={args.require_video})")
    print(f"frames: total={len(rows)}  train={n_train}  val={n_val_rows}  "
          f"val_cases={len(val_cases)} stride={args.stride_sec}s")
    print("tool freq (frames present):")
    for i, t in enumerate(TOOLS):
        print(f"  {t:34s} {stat_tool[i]}")
    print("task freq:")
    for i, t in enumerate(TASKS):
        print(f"  {t:34s} {stat_task[i]}")
    print(f"\nmanifest -> {args.out_manifest}")


if __name__ == "__main__":
    main()

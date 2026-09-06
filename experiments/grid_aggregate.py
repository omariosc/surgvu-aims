"""Aggregate a class-imbalance grid (lever 1) -> pick the best variant.

Each grid cell trains a spine with a (focal-gamma, sample-strength) combo and runs
diagnose_spine.py -> per_class_diag.json. This reads ALL the cells' diagnostics,
ranks them by the RECOVERABLE macro-F1 (tool_macro_f1_excl_floor -- the 3 zero-val
data-floor tools removed, exactly the metric the brief asks to maximise), and emits
a single machine-readable BEST line for the slurm wrapper to append.

Run:
  python grid_aggregate.py --root .../models/grid1 --base_diag .../spine_full/per_class_diag.json
"""
import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="dir containing g<gamma>_s<strength>/per_class_diag.json subdirs")
    ap.add_argument("--base_diag", default=None,
                    help="baseline (uniform) per_class_diag.json for the delta")
    args = ap.parse_args()

    base_excl = None
    if args.base_diag and os.path.exists(args.base_diag):
        try:
            base_excl = json.load(open(args.base_diag))["tool_macro_f1_excl_floor"]
        except Exception:
            base_excl = None

    rows = []
    for diag in sorted(glob.glob(os.path.join(args.root, "*", "per_class_diag.json"))):
        cell = os.path.basename(os.path.dirname(diag))
        try:
            d = json.load(open(diag))
        except Exception as e:
            print(f"  [skip] {cell}: {e}")
            continue
        rows.append(dict(cell=cell,
                         excl=d.get("tool_macro_f1_excl_floor", float("nan")),
                         all14=d.get("tool_macro_f1", float("nan")),
                         task=d.get("task_macro_f1", float("nan"))))

    if not rows:
        print("GRID_BEST cell=NONE excl=nan (no diagnostics found)")
        return

    rows.sort(key=lambda r: (r["excl"] if r["excl"] == r["excl"] else -1), reverse=True)
    print(f"# class-imbalance grid ({len(rows)} cells) ranked by recoverable macro-F1 (excl 3 data-floor tools):")
    for r in rows:
        print(f"#   {r['cell']:14s} excl-floor={r['excl']:.4f}  all14={r['all14']:.4f}  task={r['task']:.4f}")

    best = rows[0]
    delta = ""
    verdict = ""
    if base_excl is not None:
        d = best["excl"] - base_excl
        delta = f" delta_vs_uniform={d:+.4f}"
        verdict = (" CONFIRMED >=+0.05" if d >= 0.05
                   else " REFUTED (<+0.05 -> data-floor not loss)")
    print(f"GRID_BEST cell={best['cell']} excl-floor={best['excl']:.4f} "
          f"all14={best['all14']:.4f} task={best['task']:.4f}{delta}{verdict}")


if __name__ == "__main__":
    main()

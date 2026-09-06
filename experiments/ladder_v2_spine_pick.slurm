#!/bin/bash
#SBATCH --job-name=surgvu_v2pick
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/sc20osc/miccai-2026/SurgVU/logs/v2pick_%j.out
#SBATCH --error=/scratch/sc20osc/miccai-2026/SurgVU/logs/v2pick_%j.err
# NO mail directives (self-monitored). LEVER 2 -- spine-accuracy + final Cat-2 spine pick.
# Diagnoses the TWO undiagnosed spine checkpoints (spine_full_bal = balanced+focal;
# spine_tf_efficientnetv2_m = heavier backbone) AND re-runs the Cat-2 V2 grounded-BLEU
# (lever-1 v2 path) with each, to see whether a better spine fixes case124 forceps-
# confusion and/or raises grounded-BLEU. Picks the FINAL Cat-2 spine = highest grounded-
# BLEU (tie-break: highest tool excl-floor macro-F1). GPU needed for decode+forward.
set -e
mkdir -p /scratch/sc20osc/miccai-2026/SurgVU/logs
export HF_HOME=/scratch/sc20osc/hf_cache
export TORCH_HOME=/scratch/sc20osc/hf_cache/torch
export OMP_NUM_THREADS=4
PY=/scratch/sc20osc/surgvu_env/bin/python
EXP=/users/sc20osc/SurgVU/experiments
D=/scratch/sc20osc/miccai-2026/SurgVU/data
MAN=$D/frame_manifest_full.csv
RESULTS=/users/sc20osc/AUTONOMOUS_RESULTS.md
M=/scratch/sc20osc/miccai-2026/SurgVU/models

echo "=== node $(hostname) ==="; nvidia-smi -L
cd $EXP

# checkpoints to evaluate (the two undiagnosed) + spine_full reference for the V2 number
declare -A CKPTS=(
  [spine_full_bal]=$M/spine_full_bal/spine_best.pt
  [spine_tf_efficientnetv2_m]=$M/spine_tf_efficientnetv2_m/spine_best.pt
)

for TAG in spine_full_bal spine_tf_efficientnetv2_m; do
  CKPT=${CKPTS[$TAG]}
  echo "############ ${TAG} : ${CKPT} ############"
  if [ ! -f "$CKPT" ]; then echo "MISSING $CKPT -- skip"; continue; fi

  echo "=== STEP A: per-class diagnosis (${TAG}) ==="
  $PY $EXP/diagnose_spine.py --ckpt $CKPT --manifest $MAN --max_val 40000 || echo "diag failed for $TAG"
  DIAG=$(dirname $CKPT)/per_class_diag.json

  echo "=== STEP B: Cat-2 V2 action-grounded BLEU (${TAG}, nf4_thr0.3_mean) ==="
  PARSE=/scratch/sc20osc/miccai-2026/SurgVU/logs/v2pick_${SLURM_JOB_ID}_${TAG}.parse
  $PY $EXP/spine_grounding_bleu_v2.py --ckpt $CKPT --n_frames 4 --thr 0.3 --pool mean \
      --out $(dirname $CKPT)/grounded_bleu_v2.json 2>&1 | tee $PARSE || echo "bleu failed for $TAG"
  BLEU=$(grep -oE 'GROUNDED_BLEU_V2=[0-9.]+' $PARSE | tail -1 | cut -d= -f2)
  [ -z "$BLEU" ] && BLEU="nan"

  # pull macro-F1 numbers from the diagnosis json
  $PY - "$DIAG" "$SLURM_JOB_ID" "$TAG" "$BLEU" "$RESULTS" <<'PYEOF'
import json, sys
diag, jid, tag, bleu, res = sys.argv[1:6]
try:
    d = json.load(open(diag))
    all14 = d.get("tool_macro_f1")
    excl = d.get("tool_macro_f1_excl_floor")
    # forceps-confusion: top FP pairs involving a forceps family
    conf = [c for c in d.get("tool_confusion_top", [])
            if "forceps" in c.get("fp_tool","") or "forceps" in c.get("true_present","")]
    cstr = "; ".join(f"{c['fp_tool']}<-{c['true_present']}({c['n']})" for c in conf[:3]) or "none"
    line = (f"[SurgVU][{jid}] Cat-2 spine-pick {tag}: V2-grounded-BLEU(n=11,nf4_thr0.3_mean)={bleu} | "
            f"tool macroF1 all14={all14:.4f}/excl-floor={excl:.4f} | forceps-conf[{cstr}]")
except Exception as e:
    line = f"[SurgVU][{jid}] Cat-2 spine-pick {tag}: V2-grounded-BLEU={bleu} | diag parse FAILED: {e}"
open(res, "a").write(line + "\n")
print("appended:", line)
PYEOF
  echo "=== appended ${TAG} ==="; tail -1 $RESULTS
done

# ---- FINAL PICK across {spine_full (existing v2 json), spine_full_bal, effv2m} ----
echo "=== FINAL Cat-2 spine pick ==="
$PY - "$M" "$SLURM_JOB_ID" "$RESULTS" <<'PYEOF'
import json, os, sys
M, jid, res = sys.argv[1:4]
cands = ["spine_full", "spine_full_bal", "spine_tf_efficientnetv2_m"]
rows = []
for tag in cands:
    gj = os.path.join(M, tag, "grounded_bleu_v2.json")
    dj = os.path.join(M, tag, "per_class_diag.json")
    bleu = excl = None
    if os.path.exists(gj):
        try: bleu = json.load(open(gj)).get("grounded_mean_bleu")
        except Exception: pass
    if os.path.exists(dj):
        try: excl = json.load(open(dj)).get("tool_macro_f1_excl_floor")
        except Exception: pass
    rows.append((tag, bleu, excl))
# rank by grounded-BLEU desc, tie-break excl-floor macro-F1 desc
def key(r):
    b = r[1] if r[1] is not None else -1
    e = r[2] if r[2] is not None else -1
    return (b, e)
ranked = sorted(rows, key=key, reverse=True)
best = ranked[0]
tbl = " | ".join(f"{t}: BLEU={'%.4f'%b if b is not None else 'NA'}/exclF1={'%.4f'%e if e is not None else 'NA'}"
                 for t, b, e in rows)
line = (f"[SurgVU][{jid}] Cat-2 FINAL spine PICK = {best[0]} "
        f"(V2-grounded-BLEU={'%.4f'%best[1] if best[1] is not None else 'NA'}, "
        f"excl-floor-F1={'%.4f'%best[2] if best[2] is not None else 'NA'}) | candidates: {tbl}")
open(res, "a").write(line + "\n")
print("appended:", line)
PYEOF
echo "=== FINAL pick appended ==="; tail -1 $RESULTS
echo "=== DONE $(date) ==="

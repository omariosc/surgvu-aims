#!/bin/bash
#SBATCH --job-name=surgvu_l12hp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/sc20osc/miccai-2026/SurgVU/logs/ladder12_%j.out
#SBATCH --error=/scratch/sc20osc/miccai-2026/SurgVU/logs/ladder12_%j.err
# NO mail directives (self-monitored). NEW LEVER -- bipolar<-cadiere HARD-PAIR head.
# Targets the #1 spine confusion (per_class_diag bipolar<-cadiere n=1106; the case124
# Cat-2 error "The forceps type is bipolar forceps", BLEU 0.537). Adds a confusable-pair
# contrastive hinge (--hardpair) on top of the focal+balanced recipe; everything else
# identical to the imbalance grid so the delta isolates the hard-pair term.
#   PRE-REGISTERED KILL CRITERION: the bipolar<-cadiere confusion count must DROP and the
#   recoverable (excl-floor) macro-F1 must NOT regress vs the spine_full baseline diag;
#   else the look-alike confusion is encoder-capacity-bound (escalate backbone), not a
#   loss-term gap. Same train budget as the grid (5 ep / 150k frames) -- relative signal.
set -e
mkdir -p /scratch/sc20osc/miccai-2026/SurgVU/logs
export HF_HOME=/scratch/sc20osc/hf_cache
export TORCH_HOME=/scratch/sc20osc/hf_cache/torch
export OMP_NUM_THREADS=4
PY=/scratch/sc20osc/surgvu_env/bin/python
EXP=/users/sc20osc/SurgVU/experiments
D=/scratch/sc20osc/miccai-2026/SurgVU/data
MAN=$D/frame_manifest_full.csv
OUT=/scratch/sc20osc/miccai-2026/SurgVU/models/spine_hardpair
RESULTS=/users/sc20osc/AUTONOMOUS_RESULTS.md
BASE=/scratch/sc20osc/miccai-2026/SurgVU/models/spine_full/per_class_diag.json

echo "=== node $(hostname) ==="; nvidia-smi -L
cd $EXP
mkdir -p $OUT

echo "=== SMOKE (hardpair ON) ==="
$PY $EXP/train_spine.py --manifest $MAN --smoke --workers 6 --bs 32 --img_size 384 \
    --balanced_sampling --sample_strength 1.0 --focal --gamma 2 \
    --hardpair --hardpair_w 0.5 --hardpair_margin 2.0 --out_dir $OUT

echo "=== TRAIN hardpair spine (focal g2 + balanced s1.0 + hardpair w0.5 m2.0) ==="
$PY $EXP/train_spine.py --manifest $MAN --epochs 5 --bs 64 --img_size 384 \
    --workers 12 --max_train 150000 --max_val 40000 \
    --balanced_sampling --sample_strength 1.0 --focal --gamma 2 \
    --hardpair --hardpair_w 0.5 --hardpair_margin 2.0 --out_dir $OUT

echo "=== DIAGNOSE hardpair spine ==="
$PY $EXP/diagnose_spine.py --ckpt $OUT/spine_best.pt --manifest $MAN --max_val 40000

# extract bipolar<-cadiere confusion count + recoverable F1 from both diags (best-effort)
DIAG=$OUT/per_class_diag.json
parse_diag () {  # $1=json $2=label
  $PY - "$1" "$2" <<'PYEOF'
import json,sys
path,label=sys.argv[1],sys.argv[2]
try:
    d=json.load(open(path))
    conf=d.get("tool_confusion_top",[])
    # bipolar<-cadiere = fp_tool bipolar_forceps, true_present cadiere_forceps
    n_bc=None
    for c in conf:
        if "bipolar" in str(c.get("fp_tool","")) and "cadiere" in str(c.get("true_present","")):
            n_bc=c.get("n"); break
    rec=d.get("tool_macro_f1_excl_floor")
    rs=f"{rec:.4f}" if isinstance(rec,(int,float)) else str(rec)
    print(f"{label}: bipolar<-cadiere_n={n_bc} excl_floor_F1={rs}")
except Exception as e:
    print(f"{label}: parse-failed:{e}")
PYEOF
}
HPN=$(parse_diag "$DIAG" "hardpair")
BASEN=$(parse_diag "$BASE" "baseline")

echo "[SurgVU][${SLURM_JOB_ID}] Cat hard-pair head (bipolar<-cadiere contrastive hinge, focal g2+bal s1.0+hp w0.5) -> ${HPN} | ${BASEN} -> compare confusion-drop + excl-floor-F1 vs spine_full baseline" >> $RESULTS
echo "=== appended to $RESULTS ==="; tail -1 $RESULTS
echo "=== cell hardpair DONE $(date) ==="

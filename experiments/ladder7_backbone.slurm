#!/bin/bash
#SBATCH --job-name=surgvu_l7bb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=30:00:00
#SBATCH --array=0-1
#SBATCH --output=/scratch/USERNAME/miccai-2026/SurgVU/logs/ladder7_%A_%a.out
#SBATCH --error=/scratch/USERNAME/miccai-2026/SurgVU/logs/ladder7_%A_%a.err
# NO mail directives (self-monitored). NEW LEVER C (brief lever 2) -- STRONGER backbone +
# LONGER schedule for the rare tools. Array of 2 cached backbones (verified loadable
# offline from /scratch/USERNAME/hf_cache):
#   0 = convnextv2_tiny.fcmae_ft_in22k_in1k   (modern ConvNeXt, FCMAE+in22k pretrain)
#   1 = tf_efficientnetv2_m.in21k_ft_in1k     (larger EffNetV2-M spine, 1280-d feats)
# Both with focal(gamma=2) + balanced sampling (the established imbalance config) and a
# LONGER 12-epoch schedule (vs spine_full's 10) on 250k train frames. Diagnoses each.
# KILL CRITERION: neither backbone beats spine_full's all14 macro-F1 by >=+0.02
#   -> EfficientNetV2-S is already at the encoder frontier for this data -> escalate.
set -e
mkdir -p /scratch/USERNAME/miccai-2026/SurgVU/logs
export HF_HOME=/scratch/USERNAME/hf_cache
export TORCH_HOME=/scratch/USERNAME/hf_cache/torch
export OMP_NUM_THREADS=4
PY=/scratch/USERNAME/surgvu_env/bin/python
EXP=/users/USERNAME/SurgVU/experiments
D=/scratch/USERNAME/miccai-2026/SurgVU/data
MAN=$D/frame_manifest_full.csv
RESULTS=/users/USERNAME/AUTONOMOUS_RESULTS.md

BBS=(convnextv2_tiny.fcmae_ft_in22k_in1k tf_efficientnetv2_m.in21k_ft_in1k)
BB=${BBS[$SLURM_ARRAY_TASK_ID]}
TAG=$(echo $BB | cut -d. -f1)
OUT=/scratch/USERNAME/miccai-2026/SurgVU/models/spine_$TAG
BASE_DIAG=/scratch/USERNAME/miccai-2026/SurgVU/models/spine_full/per_class_diag.json
PARSE=/scratch/USERNAME/miccai-2026/SurgVU/logs/ladder7_${SLURM_JOB_ID}.parse

echo "=== node $(hostname)  backbone=$BB ==="; nvidia-smi -L
cd $EXP
mkdir -p $OUT

echo "=== SMOKE ($BB) ==="
$PY $EXP/train_spine.py --manifest $MAN --smoke --workers 6 --bs 32 --img_size 384 \
    --backbone $BB --balanced_sampling --focal --gamma 2 --out_dir $OUT

echo "=== LONGER TRAIN ($BB, 12 ep, 250k frames) ==="
# EffNetV2-M is heavier -> drop batch to 48 to stay in 48 GB; ConvNeXtV2-T fits 64
BS=64; [ "$SLURM_ARRAY_TASK_ID" = "1" ] && BS=48
$PY $EXP/train_spine.py --manifest $MAN --epochs 12 --bs $BS --img_size 384 \
    --workers 12 --max_train 250000 --max_val 40000 \
    --backbone $BB --balanced_sampling --focal --gamma 2 --out_dir $OUT

echo "=== DIAGNOSE ($BB) ==="
$PY $EXP/diagnose_spine.py --ckpt $OUT/spine_best.pt --manifest $MAN --max_val 40000 2>&1 | tee $PARSE

CMP=/scratch/USERNAME/miccai-2026/SurgVU/logs/ladder7_${SLURM_JOB_ID}_cmp.py
cat > $CMP <<'PYEOF'
import json, sys, os
new_p, base_p, bb = sys.argv[1], sys.argv[2], sys.argv[3]
n = json.load(open(new_p))
nall = n["tool_macro_f1"]; nexcl = n["tool_macro_f1_excl_floor"]; ntask = n["task_macro_f1"]
if os.path.exists(base_p):
    b = json.load(open(base_p)); ball = b["tool_macro_f1"]
    d = nall - ball
    v = "CONFIRMED stronger backbone helps" if d >= 0.02 else "REFUTED (<+0.02 -> EffV2-S at frontier)"
    print(f"BB_RESULT {bb}: all14={nall:.4f}(base {ball:.4f}, delta {d:+.4f}) excl-floor={nexcl:.4f} task={ntask:.4f}; {v}")
else:
    print(f"BB_RESULT {bb}: all14={nall:.4f} excl-floor={nexcl:.4f} task={ntask:.4f} (no base diag)")
PYEOF
$PY $CMP "$OUT/per_class_diag.json" "$BASE_DIAG" "$BB" 2>&1 | tee -a $PARSE
RES=$(grep -E '^BB_RESULT' $PARSE | tail -1 | sed 's/^BB_RESULT //')
echo "[SurgVU][${SLURM_JOB_ID}] Cat backbone/longer-train -> ${RES:-see log}" >> $RESULTS
echo "=== appended to $RESULTS ==="; tail -1 $RESULTS
echo "=== $BB DONE $(date) ==="

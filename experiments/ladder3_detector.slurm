#!/bin/bash
#SBATCH --job-name=surgvu_l3det
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/USERNAME/miccai-2026/SurgVU/logs/ladder3_%j.out
#SBATCH --error=/scratch/USERNAME/miccai-2026/SurgVU/logs/ladder3_%j.err
# NO mail directives (self-monitored). LADDER STEP 3 -- Cat-1 weakly-sup detector.
# (a) cam_pseudoboxes.py at FULL scale: Grad-CAM on the trained spine -> pseudo-boxes
# (b) train_detector.py: YOLO on the pseudo-boxes (from cached yolo11n.pt, NO download)
#     -> COCO mAP on a case-disjoint held-out pseudo-box split (self-consistency; NO
#        human bbox GT exists for 2024 -- honest ceiling for weakly-sup localisation).
# Chained afterok off the Cat-2 grounding step.
set -e
mkdir -p /scratch/USERNAME/miccai-2026/SurgVU/logs
export HF_HOME=/scratch/USERNAME/hf_cache
export TORCH_HOME=/scratch/USERNAME/hf_cache/torch
export OMP_NUM_THREADS=4
export YOLO_CONFIG_DIR=/scratch/USERNAME/hf_cache/ultralytics
PY=/scratch/USERNAME/surgvu_env/bin/python
EXP=/users/USERNAME/SurgVU/experiments
D=/scratch/USERNAME/miccai-2026/SurgVU/data
CKPT=/scratch/USERNAME/miccai-2026/SurgVU/models/spine_full/spine_best.pt
CAM=/scratch/USERNAME/miccai-2026/SurgVU/models/cam_vis
RESULTS=/users/USERNAME/AUTONOMOUS_RESULTS.md
PARSE=/scratch/USERNAME/miccai-2026/SurgVU/logs/ladder3_${SLURM_JOB_ID}.parse

echo "=== node $(hostname) ==="; nvidia-smi -L
cd $EXP

echo "=== (a) Grad-CAM pseudo-boxes at FULL scale (600 frames) ==="
$PY $EXP/cam_pseudoboxes.py --ckpt $CKPT --manifest $D/frame_manifest_full.csv \
    --out_dir $CAM --n 600 --thr 0.4 2>&1 | tee $PARSE

echo "=== (b) train YOLO detector on pseudo-boxes ==="
# tolerate detector failure (e.g. too few pseudo-boxes) without aborting the append
set +e
$PY $EXP/train_detector.py --pseudo $CAM/pseudo_boxes.json --img_dir $CAM \
    --out_root /scratch/USERNAME/miccai-2026/SurgVU/models/detector --epochs 40 2>&1 | tee -a $PARSE
set -e

MAP=$(grep -oE 'DETECTOR_mAP50_95=[0-9.nan]+' $PARSE | tail -1 | cut -d= -f2)
MAP50=$(grep -oE 'DETECTOR_mAP50=[0-9.nan]+' $PARSE | tail -1 | cut -d= -f2)
NBOX=$(grep -oE '[0-9]+ pseudo-boxes' $PARSE | tail -1 | grep -oE '^[0-9]+')
[ -z "$MAP" ] && MAP="nan"
[ -z "$MAP50" ] && MAP50="nan"
echo "[SurgVU][${SLURM_JOB_ID}] Cat-1 weakly-sup detector (YOLO on ${NBOX:-?} Grad-CAM pseudo-boxes, case-disjoint val, self-consistency NO human GT) -> mAP@[.5:.95]=${MAP} mAP50=${MAP50}" >> $RESULTS
echo "=== appended to $RESULTS ==="; tail -1 $RESULTS
echo "=== DONE $(date) ==="

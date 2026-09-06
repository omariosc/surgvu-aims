#!/bin/bash
#SBATCH --job-name=svu_c1val
#SBATCH --partition=gpu
#SBATCH --nice=0
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-4
#SBATCH --output=/scratch/sc20osc/miccai-2026/SurgVU/logs/cat1_honest_%A_%a.out
#SBATCH --error=/scratch/sc20osc/miccai-2026/SurgVU/logs/cat1_honest_%A_%a.err
# HONEST Cat-1 re-validation: trained YOLO detectors vs the REAL human-bbox public
# 2024 Cat-1 test set (7 held-out videos, COCO mAP@[.5:.95] = the grand-challenge metric).
# Replaces the self-consistency-vs-Grad-CAM mAP (0.7746-0.9788, no human GT, UNVALIDATED).
# Array tests whether self-consistency gains transfer to real boxes (diagnostic).
set -e
export HF_HOME=/scratch/sc20osc/hf_cache
export TORCH_HOME=/scratch/sc20osc/hf_cache/torch
export YOLO_CONFIG_DIR=/scratch/sc20osc/hf_cache/ultralytics
export OMP_NUM_THREADS=4
PY=/scratch/sc20osc/surgvu_env/bin/python
EXP=/users/sc20osc/SurgVU/experiments
M=/scratch/sc20osc/miccai-2026/SurgVU/models
LOG=/scratch/sc20osc/miccai-2026/SurgVU/logs

WEIGHTS=(
  "$M/detector/train/weights/best.pt"
  "$M/detector_thr0.5_yolo26m/train/weights/best.pt"
  "$M/detector_thr0.7_yolo26m_BEST_long120aug/train/weights/best.pt"
  "$M/detector_seg/train/weights/best.pt"
  "$M/detector_sam2_full/train/weights/best.pt"
)
TAGS=(base_detector thr0.5_yolo26m thr0.7_BEST_long120aug seg_cascade sam2_cascade)
W=${WEIGHTS[$SLURM_ARRAY_TASK_ID]}
TAG=${TAGS[$SLURM_ARRAY_TASK_ID]}
OUT="$LOG/cat1_honest_${TAG}.json"

echo "[$(date)] honest Cat-1 val  tag=$TAG  weights=$W"
$PY $EXP/cat1_honest_val.py --weights "$W" --out "$OUT"
echo "[$(date)] DONE $TAG -> $OUT"

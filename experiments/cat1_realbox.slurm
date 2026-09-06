#!/bin/bash
#SBATCH --job-name=surgvu_c1real
#SBATCH --partition=gpu
#SBATCH --nice=0
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/users/sc20osc/SurgVU/experiments/logs/c1real_%j.out
#SBATCH --error=/users/sc20osc/SurgVU/experiments/logs/c1real_%j.out

set -uo pipefail
export HF_HOME=/scratch/sc20osc/hf_cache
export TORCH_HOME=/scratch/sc20osc/hf_cache/torch
export YOLO_CONFIG_DIR=/scratch/sc20osc/hf_cache/ultralytics
export YOLO_OFFLINE=1
export OMP_NUM_THREADS=4

PY=/scratch/sc20osc/surgvu_env/bin/python
cd /users/sc20osc/SurgVU/experiments

echo "=== SurgVU Cat-1: detector trained on REAL HUMAN BOXES, video-disjoint ==="
echo "Weak Grad-CAM pipeline scores mAP=0.000 on this same GT (boxes 7.2x too large by"
echo "area -> IoU capped at ~0.14, below the 0.5 floor). This is the honest baseline."
echo "host=$(hostname) job=${SLURM_JOB_ID}"
date

$PY cat1_realbox_train.py \
  --train_videos 1 2 3 4 5 \
  --val_videos 6 7 \
  --model /scratch/sc20osc/AILET/from_pc/weights/yolo26m.pt \
  --epochs 60 --imgsz 640 --batch 16

echo "DONE $(date)"

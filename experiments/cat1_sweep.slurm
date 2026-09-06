#!/bin/bash
#SBATCH --job-name=surgvu_c1sweep
#SBATCH --partition=gpu
#SBATCH --nice=60
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --array=0-7%2
#SBATCH --output=/users/sc20osc/SurgVU/experiments/logs/c1sweep_%A_%a.out
#SBATCH --error=/users/sc20osc/SurgVU/experiments/logs/c1sweep_%A_%a.out

# SurgVU Cat-1 headroom sweep. ONE KNOB PER ARM, all on the SAME video-disjoint
# `main` split (train v1-5, val v6+7) that produced the honest 0.1470 baseline.
# nice=60 so ORENA (priority ~233-242) always wins contention; %2 caps us at two
# concurrent GPUs.
set -uo pipefail
export HF_HOME=/scratch/sc20osc/hf_cache
export TORCH_HOME=/scratch/sc20osc/hf_cache/torch
export YOLO_CONFIG_DIR=/scratch/sc20osc/hf_cache/ultralytics
export YOLO_OFFLINE=1
export OMP_NUM_THREADS=4

PY=/scratch/sc20osc/surgvu_env/bin/python
W=/scratch/sc20osc/AILET/from_pc/weights
cd /users/sc20osc/SurgVU/experiments

# arm | model | epochs | imgsz | seed
ARMS=(
  "S0_seed0|$W/yolo26m.pt|60|640|0"
  "S0_seed1|$W/yolo26m.pt|60|640|1"
  "S0_seed2|$W/yolo26m.pt|60|640|2"
  "E200|$W/yolo26m.pt|200|640|0"
  "R960|$W/yolo26m.pt|60|960|0"
  "R1280|$W/yolo26m.pt|60|1280|0"
  "CAP_n|$W/yolo26n.pt|60|640|0"
  "CAP_l|$W/yolo26l.pt|60|640|0"
)

IFS='|' read -r ARM MODEL EPOCHS IMGSZ SEED <<< "${ARMS[$SLURM_ARRAY_TASK_ID]}"

echo "=== SurgVU Cat-1 headroom arm ${SLURM_ARRAY_TASK_ID}: ${ARM} ==="
echo "model=$(basename $MODEL) epochs=$EPOCHS imgsz=$IMGSZ seed=$SEED"
echo "baseline to beat = 0.1470 mAP@[.5:.95] (job 6933641, same main split)"
echo "host=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
date

$PY cat1_headroom_sweep.py \
  --arm "$ARM" --split main \
  --model "$MODEL" --epochs "$EPOCHS" --imgsz "$IMGSZ" --batch 16 --seed "$SEED"

echo "DONE $(date)"

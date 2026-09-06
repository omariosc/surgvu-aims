#!/bin/bash
#SBATCH --job-name=surgvu_c1lovo
#SBATCH --partition=gpu
#SBATCH --nice=90
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-6%2
#SBATCH --output=/users/sc20osc/SurgVU/experiments/logs/c1lovo_%A_%a.out
#SBATCH --error=/users/sc20osc/SurgVU/experiments/logs/c1lovo_%A_%a.out

# SurgVU Cat-1 INSTRUMENT FIX — 7-fold leave-one-video-out at the baseline config.
#
# This does NOT weaken the honesty property the parent asked me to preserve: every
# fold is still strictly VIDEO-DISJOINT.  It is an ADDITION to the v1-5 / v6-7 main
# split, reported as its own protocol, because that main split turns out to be a
# very fragile instrument: its val set contains only 4 of 14 classes, and 2 of them
# (clip_applier, vessel_sealer) have ZERO train instances, so 50% of the headline
# denominator is pinned at 0.000 by construction and cannot move for ANY arm.
# LOVO puts every class into a val fold, restores train support for clip_applier
# (present in both v6 and v7), and yields a 7-fold error bar instead of a single
# point estimate.  nice=90 => yields to ORENA AND to the headroom sweep.
set -uo pipefail
export HF_HOME=/scratch/sc20osc/hf_cache
export TORCH_HOME=/scratch/sc20osc/hf_cache/torch
export YOLO_CONFIG_DIR=/scratch/sc20osc/hf_cache/ultralytics
export YOLO_OFFLINE=1
export OMP_NUM_THREADS=4

PY=/scratch/sc20osc/surgvu_env/bin/python
W=/scratch/sc20osc/AILET/from_pc/weights
cd /users/sc20osc/SurgVU/experiments

FOLD=$(( SLURM_ARRAY_TASK_ID + 1 ))

echo "=== SurgVU Cat-1 LOVO fold ${FOLD} (hold out video ${FOLD}) ==="
echo "host=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
date

$PY cat1_headroom_sweep.py \
  --arm "LOVO_v${FOLD}" --split "lovo${FOLD}" \
  --model "$W/yolo26m.pt" --epochs 60 --imgsz 640 --batch 16 --seed 0

echo "DONE $(date)"

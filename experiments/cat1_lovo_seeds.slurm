#!/bin/bash
#SBATCH --job-name=surgvu_lovoseed
#SBATCH --partition=gpu
#SBATCH --nice=90
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-13%4
#SBATCH --output=/users/sc20osc/SurgVU/experiments/logs/lovoseed_%A_%a.out
#SBATCH --error=/users/sc20osc/SurgVU/experiments/logs/lovoseed_%A_%a.out

# ============================================================================
# THE FOLD-NOISE ESTIMATE.  Nothing else here; this job exists only to give the
# 7-fold LOVO comparison an error bar it does not currently have.
#
# The main split's Delta_noise = 0.0365 CANNOT be reused: it is a 3-seed spread of
# best.pt on a 4-class val set whose ceiling is 0.500, i.e. a different estimator
# on a different protocol.  LOVO's estimand is the 7-fold MEAN, and the arms will
# be compared PAIRED per fold, so what we need is the WITHIN-FOLD seed SD.
#
# Seed 0 x 7 folds already exists (job 7050543).  This adds seeds 1 and 2, giving
# 3 seeds x 7 folds.  From that we get, per fold, the seed SD; the pre-registered
# bar for any pseudo-label arm is then:
#     (i)  paired mean gain over the 7 folds > 2 * SE(paired mean), where
#          SE = sqrt(sum_f 2*s_f^2 / 3) / 7  using the per-fold seed variances, AND
#     (ii) >= 6 of 7 folds move in the same direction (the sign-consistency gate
#          that MultiBypass used to separate a real fusion gain from a fit artifact).
# Both must hold on UNSELECTED last.pt.  best.pt is reported but never gates.
# ============================================================================
set -uo pipefail
export HF_HOME=/scratch/sc20osc/hf_cache
export TORCH_HOME=/scratch/sc20osc/hf_cache/torch
export YOLO_CONFIG_DIR=/scratch/sc20osc/hf_cache/ultralytics
export YOLO_OFFLINE=1
export OMP_NUM_THREADS=4

PY=/scratch/sc20osc/surgvu_env/bin/python
W=/scratch/sc20osc/AILET/from_pc/weights
cd /users/sc20osc/SurgVU/experiments

FOLD=$(( SLURM_ARRAY_TASK_ID % 7 + 1 ))
SEED=$(( SLURM_ARRAY_TASK_ID / 7 + 1 ))

echo "=== LOVO fold ${FOLD} seed ${SEED} (fold-noise replication) ==="
echo "host=$(hostname) job=${SLURM_JOB_ID}"; date

$PY cat1_headroom_sweep.py \
  --arm "LOVO_v${FOLD}_s${SEED}" --split "lovo${FOLD}" \
  --model "$W/yolo26m.pt" --epochs 60 --imgsz 640 --batch 16 --seed "${SEED}"

echo "DONE $(date)"

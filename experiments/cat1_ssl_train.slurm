#!/bin/bash
#SBATCH --job-name=surgvu_ssltrain
#SBATCH --partition=gpu
#SBATCH --nice=90
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-14%4
#SBATCH --output=/users/USERNAME/SurgVU/experiments/logs/ssltrain_%A_%a.out
#SBATCH --error=/users/USERNAME/SurgVU/experiments/logs/ssltrain_%A_%a.out
# STAGE 4 -- 5 arms x 3 seeds on the MAIN split.  Self-gating: every task reads
# GATE.json first and exits without training if the label-free check failed.
# Baseline needs no run: S0_seed0/1/2 already exist (0.1444 best / 0.1110 last,
# with clip_applier and vessel_sealer at exactly 0.000 in all three).
set -uo pipefail
export HF_HOME=/scratch/USERNAME/hf_cache
export TORCH_HOME=/scratch/USERNAME/hf_cache/torch
export YOLO_CONFIG_DIR=/scratch/USERNAME/hf_cache/ultralytics
export YOLO_OFFLINE=1
export OMP_NUM_THREADS=4
PY=/scratch/USERNAME/surgvu_env/bin/python
cd /users/USERNAME/SurgVU/experiments
ARMS=(FILTER ASSIGN COMBINED NEG_UNCONSTRAINED NEG_SHUFFLE)
ARM=${ARMS[$(( SLURM_ARRAY_TASK_ID % 5 ))]}
SEED=$(( SLURM_ARRAY_TASK_ID / 5 ))
echo "=== arm=${ARM} seed=${SEED} split=main ==="
echo "host=$(hostname) job=${SLURM_JOB_ID}"; date
$PY cat1_ssl_train.py --arm "${ARM}" --split main --tau 0.25 --seed "${SEED}"
echo "DONE $(date)"

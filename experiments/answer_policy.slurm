#!/bin/bash
#SBATCH --job-name=surgvu_policy
#SBATCH --partition=nodes
#SBATCH --nice=0
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/users/USERNAME/SurgVU/experiments/logs/policy_%j.out
#SBATCH --error=/users/USERNAME/SurgVU/experiments/logs/policy_%j.out

set -euo pipefail
export HF_HOME=/scratch/USERNAME/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OUT=/scratch/USERNAME/miccai-2026/SurgVU/models/answer_policy_surface.json

PY=/scratch/USERNAME/surgvu_env/bin/python
cd /users/USERNAME/SurgVU/experiments

echo "=== SurgVU 2026 Cat-2 METRIC RE-CALIBRATION (BLEU -> BERTScore-F1) ==="
echo "host=$(hostname) job=${SLURM_JOB_ID}"
date

$PY answer_policy_surface.py

echo "DONE $(date)"

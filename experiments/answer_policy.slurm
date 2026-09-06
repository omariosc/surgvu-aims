#!/bin/bash
#SBATCH --job-name=surgvu_policy
#SBATCH --partition=nodes
#SBATCH --nice=0
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/users/sc20osc/SurgVU/experiments/logs/policy_%j.out
#SBATCH --error=/users/sc20osc/SurgVU/experiments/logs/policy_%j.out

set -euo pipefail
export HF_HOME=/scratch/sc20osc/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OUT=/scratch/sc20osc/miccai-2026/SurgVU/models/answer_policy_surface.json

PY=/scratch/sc20osc/surgvu_env/bin/python
cd /users/sc20osc/SurgVU/experiments

echo "=== SurgVU 2026 Cat-2 METRIC RE-CALIBRATION (BLEU -> BERTScore-F1) ==="
echo "host=$(hostname) job=${SLURM_JOB_ID}"
date

$PY answer_policy_surface.py

echo "DONE $(date)"

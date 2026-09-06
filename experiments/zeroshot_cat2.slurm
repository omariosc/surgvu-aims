#!/bin/bash
#SBATCH --job-name=surgvu_zs26
#SBATCH --partition=gpu
#SBATCH --nice=0
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/users/sc20osc/SurgVU/experiments/logs/zs26_%j.out
#SBATCH --error=/users/sc20osc/SurgVU/experiments/logs/zs26_%j.out

set -euo pipefail
export HF_HOME=/scratch/sc20osc/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

PY=/scratch/sc20osc/surgvu_env/bin/python
cd /users/sc20osc/SurgVU/experiments

MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
OUT="${OUT:-/scratch/sc20osc/miccai-2026/SurgVU/models/zeroshot_cat2_2026.json}"

echo "=== SurgVU 2026 Cat-2 ZERO-SHOT baseline ==="
echo "host=$(hostname) job=${SLURM_JOB_ID} model=${MODEL}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
date

$PY zeroshot_cat2_2026.py --model "$MODEL" --frames 1 5 --out "$OUT"

echo "DONE $(date)"

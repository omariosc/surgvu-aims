#!/bin/bash
#SBATCH --job-name=surgvu_c2hyb
#SBATCH --partition=gpu
#SBATCH --nice=60
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/users/USERNAME/SurgVU/experiments/logs/c2hyb_%j.out
#SBATCH --error=/users/USERNAME/SurgVU/experiments/logs/c2hyb_%j.out

# SurgVU 2026 Cat-2: VLM-as-short-span-extractor -> syntactic template.
# Tests the architecture implied by 6980942: the best zero-shot free-text config
# (0.7236) scores BELOW a ~7-word declarative that is WRONG (0.7932), because the
# VLM cannot be prompted into the reference register (it emitted 13-45 words).
# Includes the no-vision template control that isolates the value of vision.
# nice=60 => ORENA (priority ~233-242) always wins contention.  ~15 min.
set -uo pipefail
export HF_HOME=/scratch/USERNAME/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

PY=/scratch/USERNAME/surgvu_env/bin/python
cd /users/USERNAME/SurgVU/experiments

MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"

echo "=== SurgVU 2026 Cat-2 HYBRID (extractor -> syntactic template) ==="
echo "host=$(hostname) job=${SLURM_JOB_ID} model=${MODEL}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
date

$PY cat2_hybrid_entity.py --model "$MODEL" --frames 1 5 --k_samples 5

echo "DONE $(date)"

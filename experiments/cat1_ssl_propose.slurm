#!/bin/bash
#SBATCH --job-name=surgvu_sslprop
#SBATCH --partition=gpu
#SBATCH --nice=90
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/users/USERNAME/SurgVU/experiments/logs/sslpropose_%j.out
#SBATCH --error=/users/USERNAME/SurgVU/experiments/logs/sslpropose_%j.out
# STAGE 3 -- run all 8 per-split proposers, measure agreement / firing / yield,
# write GATE.json.  ~20 GPU-min, and it can kill the lever before any training.
set -uo pipefail
export HF_HOME=/scratch/USERNAME/hf_cache
export TORCH_HOME=/scratch/USERNAME/hf_cache/torch
export YOLO_CONFIG_DIR=/scratch/USERNAME/hf_cache/ultralytics
export YOLO_OFFLINE=1
export OMP_NUM_THREADS=4
PY=/scratch/USERNAME/surgvu_env/bin/python
cd /users/USERNAME/SurgVU/experiments
echo "host=$(hostname) job=${SLURM_JOB_ID}"; date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY cat1_ssl_propose.py --proposers all
echo "DONE $(date)"

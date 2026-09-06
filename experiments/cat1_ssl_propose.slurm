#!/bin/bash
#SBATCH --job-name=surgvu_sslprop
#SBATCH --partition=gpu
#SBATCH --nice=90
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/users/sc20osc/SurgVU/experiments/logs/sslpropose_%j.out
#SBATCH --error=/users/sc20osc/SurgVU/experiments/logs/sslpropose_%j.out
# STAGE 3 -- run all 8 per-split proposers, measure agreement / firing / yield,
# write GATE.json.  ~20 GPU-min, and it can kill the lever before any training.
set -uo pipefail
export HF_HOME=/scratch/sc20osc/hf_cache
export TORCH_HOME=/scratch/sc20osc/hf_cache/torch
export YOLO_CONFIG_DIR=/scratch/sc20osc/hf_cache/ultralytics
export YOLO_OFFLINE=1
export OMP_NUM_THREADS=4
PY=/scratch/sc20osc/surgvu_env/bin/python
cd /users/sc20osc/SurgVU/experiments
echo "host=$(hostname) job=${SLURM_JOB_ID}"; date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY cat1_ssl_propose.py --proposers all
echo "DONE $(date)"

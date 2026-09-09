#!/bin/bash
#SBATCH --job-name=cat2_taskcls
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH --exclude=gpu013
#SBATCH -o /users/USERNAME/joblogs/cat2_taskcls_%j.out
set -euo pipefail
# eio-guard: an unreadable HOME scripts dir means the NFS export is flapping -> abort cleanly
# rather than burn the allocation (learned 2026-08-17).
[ -r /users/USERNAME/SurgVU/experiments ] || { echo "[eio-guard] HOME scripts dir not readable -> abort"; exit 3; }
echo "node=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/scratch/USERNAME/surgvu_env/bin/python /users/USERNAME/SurgVU/experiments/train_cat2_task_classifier.py \
  --epochs 8 --k 4 --bs 8 --lr 1e-4
echo "done=$(date -Is)"

#!/bin/bash
#SBATCH --job-name=surgvu_c1pool
#SBATCH --partition=nodes
#SBATCH --nice=0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/users/USERNAME/SurgVU/experiments/logs/c1pool_%j.out
#SBATCH --error=/users/USERNAME/SurgVU/experiments/logs/c1pool_%j.out

# CPU-only gate job: decode the 7 annotated Cat-1 videos ONCE into a shared frame
# pool + image-list splits.  Runs on the `nodes` (CPU) partition so it takes ZERO
# GPU away from ORENA.  Everything downstream depends on this with afterok.
set -uo pipefail
export HF_HOME=/scratch/USERNAME/hf_cache
export OMP_NUM_THREADS=8

PY=/scratch/USERNAME/surgvu_env/bin/python
cd /users/USERNAME/SurgVU/experiments

echo "=== SurgVU Cat-1: build shared frame pool + split index files ==="
echo "host=$(hostname) job=${SLURM_JOB_ID}"
date
$PY cat1_build_pool.py
echo "DONE $(date)"

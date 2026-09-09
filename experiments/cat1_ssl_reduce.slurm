#!/bin/bash
#SBATCH --job-name=surgvu_sslred
#SBATCH --partition=nodes
#SBATCH --nice=90
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/users/USERNAME/SurgVU/experiments/logs/sslreduce_%j.out
#SBATCH --error=/users/USERNAME/SurgVU/experiments/logs/sslreduce_%j.out
# STAGE 2 -- leak verdict + bounded candidate set. CPU only.
set -uo pipefail
PY=/scratch/USERNAME/surgvu_env/bin/python
cd /users/USERNAME/SurgVU/experiments
echo "host=$(hostname) job=${SLURM_JOB_ID}"; date
$PY cat1_ssl_reduce.py
echo "DONE $(date)"

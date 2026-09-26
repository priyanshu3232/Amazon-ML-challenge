#!/bin/bash
# SLURM job for PARAM Ganga (IIT Roorkee): full pipeline on one high-memory node
# (2x Xeon 8268 = 48 cores, 768 GB RAM). Submit from code/business_entity_resolution:
#     sbatch run_paramganga.sbatch
#SBATCH --job-name=er_v2
#SBATCH --partition=highmemory
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --time=20:00:00
#SBATCH --output=er_v2.%J.out
#SBATCH --error=er_v2.%J.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR/src"
source "$SLURM_SUBMIT_DIR/../../.venv/bin/activate"
export OMP_NUM_THREADS=48
N_QUERIES="${N_QUERIES:-500000}"
echo "start $(date) on $(hostname), n_queries=$N_QUERIES"
python prepare.py --split train
python prepare.py --split test
python train.py --n-queries "$N_QUERIES" --tag _v2
python predict.py --tag _v2
echo "FINISHED $(date)"

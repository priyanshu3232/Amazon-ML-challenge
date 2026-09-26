#!/bin/bash
# Full pipeline on one Linux box (EC2 or similar), all output to work/run<tag>.log.
# Usage, from anywhere, detached so it survives a dropped SSH session:
#     nohup bash ~/er/code/business_entity_resolution/run_aws.sh 300000 _v2 >/dev/null 2>&1 &
#     tail -f ~/er/work/run_v2.log
# n_queries by RAM: 120000 (32 GB), 300000 (64 GB), 500000 (128 GB+).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
N="${1:-300000}"; TAG="${2:-_v2}"
cd "$ROOT/code/business_entity_resolution/src"
source "$ROOT/.venv/bin/activate"
export OMP_NUM_THREADS="$(nproc)"
mkdir -p "$ROOT/work"
LOG="$ROOT/work/run${TAG}.log"
{
  echo "start $(date) host=$(hostname) cores=$(nproc) n_queries=$N tag=$TAG"
  python prepare.py --split train
  python prepare.py --split test
  python train.py --n-queries "$N" --tag "$TAG"
  python predict.py --tag "$TAG"
  echo "FINISHED $(date)"
} >"$LOG" 2>&1

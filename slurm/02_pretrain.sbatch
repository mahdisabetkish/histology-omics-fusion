#!/bin/bash
#SBATCH --job-name=hof-ssl
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=7
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --array=0-2

# Cross-modal contrastive pretraining: the main run and its two ablations.
# Roughly half an hour each on a GTX 1080 Ti, less on anything newer.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

source slurm/_env.sh

ARGS=(
  "--tag ssl"
  "--tag ssl_no_imagenet --no-imagenet"
  "--tag ssl_no_neighbour_mask --neighbour-radius 0"
)

IDX="${SLURM_ARRAY_TASK_ID:-0}"
echo "running: ${ARGS[$IDX]}"

# shellcheck disable=SC2086
python -m src.pretrain_ssl \
  ${ARGS[$IDX]} \
  --epochs 60 \
  --batch-size 256 \
  --workers "$NUM_WORKERS" \
  --seed 0

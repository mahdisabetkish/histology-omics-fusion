#!/bin/bash
#SBATCH --job-name=hof-train
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=7
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --array=0-27%8

# The full run matrix as one array: supervised baselines, the transfer runs
# that start from a pretrained checkpoint, the label-efficiency sweep and the
# two cross-modal regressions.
#
# %8 throttles to eight concurrent tasks. Raise it if the queue allows; nothing
# here shares state between tasks, each one writes to its own runs/<tag>/.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

source slurm/_env.sh

SSL=runs/ssl/checkpoint.pt
BASE="--epochs 30 --workers $NUM_WORKERS --seed 0"
FUSED="--modality both --fusion gated"

JOBS=(
  # supervised baselines
  "src.train --tag expression_only    --modality expression"
  "src.train --tag image_only         --modality image"
  "src.train --tag image_only_scratch --modality image --no-imagenet"
  "src.train --tag fusion_concat      --modality both --fusion concat"
  "src.train --tag fusion_gated       $FUSED"
  "src.train --tag fusion_bilinear    --modality both --fusion bilinear"

  # started from the self-supervised checkpoint
  "src.train --tag fusion_gated_ssl     $FUSED --init $SSL"
  "src.train --tag expression_only_ssl  --modality expression --init $SSL"
  "src.train --tag image_only_ssl       --modality image --init $SSL"

  # frozen encoders: measures the representation rather than the fine-tuning
  "src.train --tag probe_ssl                    $FUSED --init $SSL --freeze-encoders --lr 1e-3"
  "src.train --tag probe_ssl_no_imagenet        $FUSED --init runs/ssl_no_imagenet/checkpoint.pt --freeze-encoders --lr 1e-3 --no-imagenet"
  "src.train --tag probe_ssl_no_neighbour_mask  $FUSED --init runs/ssl_no_neighbour_mask/checkpoint.pt --freeze-encoders --lr 1e-3"

  # label efficiency, both arms
  "src.train --tag eff_scratch_001 $FUSED --label-fraction 0.01"
  "src.train --tag eff_scratch_002 $FUSED --label-fraction 0.02"
  "src.train --tag eff_scratch_005 $FUSED --label-fraction 0.05"
  "src.train --tag eff_scratch_010 $FUSED --label-fraction 0.10"
  "src.train --tag eff_scratch_025 $FUSED --label-fraction 0.25"
  "src.train --tag eff_scratch_050 $FUSED --label-fraction 0.50"
  "src.train --tag eff_scratch_100 $FUSED --label-fraction 1.00"
  "src.train --tag eff_ssl_001 $FUSED --label-fraction 0.01 --init $SSL"
  "src.train --tag eff_ssl_002 $FUSED --label-fraction 0.02 --init $SSL"
  "src.train --tag eff_ssl_005 $FUSED --label-fraction 0.05 --init $SSL"
  "src.train --tag eff_ssl_010 $FUSED --label-fraction 0.10 --init $SSL"
  "src.train --tag eff_ssl_025 $FUSED --label-fraction 0.25 --init $SSL"
  "src.train --tag eff_ssl_050 $FUSED --label-fraction 0.50 --init $SSL"
  "src.train --tag eff_ssl_100 $FUSED --label-fraction 1.00 --init $SSL"

  # expression regressed from histology
  "src.train_expression --tag expr_from_image"
  "src.train_expression --tag expr_from_image_ssl --init $SSL"
)

# Defaulted so the script is also runnable outside an array for a single job:
#   SLURM_ARRAY_TASK_ID=4 bash slurm/03_train.sbatch
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$IDX" -ge "${#JOBS[@]}" ]; then
  echo "no job at index $IDX (matrix has ${#JOBS[@]})"
  exit 0
fi

echo "running: ${JOBS[$IDX]}"
# shellcheck disable=SC2086
python -m ${JOBS[$IDX]} $BASE

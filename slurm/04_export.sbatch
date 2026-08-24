#!/bin/bash
#SBATCH --job-name=hof-export
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

# Score every checkpoint on the held-out donor, then rebuild docs/.
# The GPU is only for the evaluation passes; the export itself is CPU work.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

source slurm/_env.sh

TAGS=(
  expression_only image_only image_only_scratch
  fusion_concat fusion_gated fusion_bilinear
  fusion_gated_ssl expression_only_ssl image_only_ssl
  probe_ssl probe_ssl_no_imagenet probe_ssl_no_neighbour_mask
)

for tag in "${TAGS[@]}"; do
  if [ -f "runs/$tag/checkpoint.pt" ]; then
    for split in test val train; do
      python -m src.evaluate --run "runs/$tag" --split "$split" \
        --workers "$NUM_WORKERS"
    done
  else
    echo "skip $tag: no checkpoint"
  fi
done

python -m src.export_dashboard
python scripts/make_report.py | tee results_tables.md

echo "docs/ rebuilt"

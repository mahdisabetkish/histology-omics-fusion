#!/bin/bash
# Submit the whole pipeline with the dependencies wired up.
#
#   bash slurm/submit_all.sh              # from data download onward
#   bash slurm/submit_all.sh --skip-data  # data already on disk

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

DEP=""
if [ "${1:-}" != "--skip-data" ]; then
  jid=$(sbatch --parsable slurm/01_data.sbatch)
  echo "data      $jid"
  DEP="--dependency=afterok:$jid"
fi

# shellcheck disable=SC2086
jid_ssl=$(sbatch --parsable $DEP slurm/02_pretrain.sbatch)
echo "pretrain  $jid_ssl"

jid_train=$(sbatch --parsable --dependency=afterok:$jid_ssl slurm/03_train.sbatch)
echo "train     $jid_train"

jid_export=$(sbatch --parsable --dependency=afterok:$jid_train slurm/04_export.sbatch)
echo "export    $jid_export"

echo
echo "watch with: squeue -u \$USER -o '%.10i %.14j %.8T %.10M %R'"

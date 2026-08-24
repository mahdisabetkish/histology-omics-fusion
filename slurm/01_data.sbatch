#!/bin/bash
#SBATCH --job-name=hof-data
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Download, crop and normalise. No GPU is used in this stage.
#
# The download is throttled by the host rather than by bandwidth, so the eight
# CPUs are for the TIFF decoding and patch extraction in build_dataset, not for
# the transfer. Give this a generous time limit: the transfer took about two
# hours from a well-connected machine and the host does not go faster.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

source slurm/_env.sh

python -m src.data.download
python scripts/check_data.py
python -m src.data.build_dataset
python -m src.data.features --n-hvg 3000

echo "data stage complete"

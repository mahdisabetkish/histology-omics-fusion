# Sourced by every job script. Edit this file, not the individual scripts.
#
# Two ways to get an interpreter; pick one and delete the other. The module
# names are placeholders, since they differ at every site.

# --- option A: conda -----------------------------------------------------
# module load anaconda3
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate histology-omics-fusion

# --- option B: a virtualenv on shared storage ----------------------------
# module load python/3.11 cuda/12.6
# source "$PWD/.venv/bin/activate"

# Keep every library's thread pool inside the CPU allocation. Without this,
# OpenBLAS and MKL each spawn a thread per core on the node rather than per
# core in the allocation, and oversubscribe badly on shared nodes.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# DataLoader workers. One fewer than the allocation leaves room for the main
# process; the loaders are not the bottleneck on a GPU node anyway.
export NUM_WORKERS="$(( ${SLURM_CPUS_PER_TASK:-5} - 1 ))"

# torchvision caches the ImageNet weights here. Point it at shared storage so
# every array task does not download its own copy, and so compute nodes with no
# outbound network still find them.
export TORCH_HOME="${TORCH_HOME:-$PWD/.torch}"

export PYTHONUNBUFFERED=1

echo "host       $(hostname)"
echo "job        ${SLURM_JOB_ID:-none} ${SLURM_ARRAY_TASK_ID:+task $SLURM_ARRAY_TASK_ID}"
echo "cpus       ${SLURM_CPUS_PER_TASK:-?}  workers $NUM_WORKERS"
python -c "import torch;print('torch     ',torch.__version__,'cuda',torch.cuda.is_available())"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

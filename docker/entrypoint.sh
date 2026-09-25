#!/bin/bash
# Container entrypoint. Short names for each pipeline stage; anything it does
# not recognise is executed as given, so `docker run IMAGE python -m src.train
# ...` still works exactly as it does outside the container.
#
# Stages read and write data/, runs/ and docs/ under /app. Mount those from the
# host, or everything the container produces disappears with it.

set -euo pipefail
cd /app

# Mirrors slurm/04_export.sbatch: the dashboard reads per-spot predictions for
# all three splits, so every checkpoint is scored on each before exporting.
EVAL_TAGS=(
  expression_only image_only image_only_scratch
  fusion_concat fusion_gated fusion_bilinear
  fusion_gated_ssl expression_only_ssl image_only_ssl
  probe_ssl probe_ssl_no_imagenet probe_ssl_no_neighbour_mask
)

usage() {
  cat <<'EOF'
histology-omics-fusion

usage: docker run [docker options] IMAGE <command> [args...]

pipeline stages, in order
  download     fetch the raw spatialLIBD files (~6.6 GB, resumable)
  check        report which raw files are present and intact
  prepare      crop patches, align counts and annotations, select genes
  train        run the experiment matrix (scripts/run_all.py; takes its flags)
  export       score every checkpoint, rebuild docs/ and print the tables

shortcuts
  data         download + check + prepare
  pipeline     data + train + export, the whole study end to end

other
  report       results tables as markdown (scripts/make_report.py)
  info         versions, device and thread counts
  help         this message
  <anything>   run as a command, e.g. `python -m src.train --help` or `bash`

mount data/, runs/ and docs/ at /app/data, /app/runs and /app/docs, and give
DataLoader workers shared memory with --shm-size=2g (or --ipc=host).
EOF
}

info() {
  python - <<'EOF'
import os, platform
import numpy, sklearn, torch, torchvision
print(f"python       {platform.python_version()}")
print(f"torch        {torch.__version__}  (image variant: {os.environ.get('HOF_TORCH_VARIANT', '?')})")
print(f"torchvision  {torchvision.__version__}")
print(f"numpy        {numpy.__version__}")
print(f"scikit-learn {sklearn.__version__}")
print(f"cpu threads  {torch.get_num_threads()} of {os.cpu_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"cuda:{i}       {p.name}, {p.total_memory / 1e9:.1f} GB")
else:
    print("cuda         not available, running on CPU")
EOF
}

stage_download() { python -m src.data.download "$@"; }
stage_check()    { python scripts/check_data.py "$@"; }

stage_prepare() {
  python -m src.data.build_dataset
  python -m src.data.features --n-hvg 3000 "$@"
}

stage_train() { python scripts/run_all.py "$@"; }

stage_export() {
  for tag in "${EVAL_TAGS[@]}"; do
    if [ -f "runs/$tag/checkpoint.pt" ]; then
      for split in test val train; do
        python -m src.evaluate --run "runs/$tag" --split "$split" "$@"
      done
    else
      echo "skip $tag: no checkpoint"
    fi
  done
  python -m src.export_dashboard
  python scripts/make_report.py
}

# A bind mount whose host directory does not exist yet is created by the Docker
# daemon, owned by root, and this user cannot write to it. Say so up front
# rather than failing deep inside a stage with a bare PermissionError.
check_writable() {
  local bad=()
  for dir in data runs docs; do
    [ -w "$dir" ] || bad+=("$dir")
  done
  if [ "${#bad[@]}" -gt 0 ]; then
    echo "warning: not writable by uid $(id -u): ${bad[*]}" >&2
    echo "  Create the host directories before the first run (mkdir -p data runs docs)," >&2
    echo "  or fix ones Docker already created: sudo chown -R \"\$(id -u):\$(id -g)\" data runs docs" >&2
  fi
}

command="${1:-help}"
[ "$#" -gt 0 ] && shift

case "$command" in
  help|-h|--help|info) ;;
  *) check_writable ;;
esac

case "$command" in
  download) stage_download "$@" ;;
  check)    stage_check "$@" ;;
  prepare)  stage_prepare "$@" ;;
  train)    stage_train "$@" ;;
  export)   stage_export "$@" ;;
  report)   exec python scripts/make_report.py "$@" ;;
  data)     stage_download; stage_check; stage_prepare ;;
  pipeline) stage_download; stage_check; stage_prepare
            stage_train "$@"; stage_export ;;
  info)     info ;;
  help|-h|--help) usage ;;
  *)        exec "$command" "$@" ;;
esac

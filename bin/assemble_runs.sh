#!/usr/bin/env bash
# Rebuild the runs/<tag>/ layout that src.export_dashboard and
# scripts/make_report.py read, from the separate task outputs Nextflow stages.
#
#   assemble_runs.sh TRAINED_DIR EVALUATED_DIR OUT_DIR
#
# TRAINED_DIR holds one directory per run, named by tag. EVALUATED_DIR holds
# <tag>__eval_<split>/ directories. Everything is linked, not copied: the
# checkpoints alone run to gigabytes.

set -euo pipefail

trained="$1"
evaluated="$2"
out="$3"

mkdir -p "$out"

for run in "$trained"/*/; do
    tag="$(basename "$run")"
    mkdir -p "$out/$tag"
    for entry in "$run"*; do
        ln -s "$(readlink -f "$entry")" "$out/$tag/$(basename "$entry")"
    done
done

for result in "$evaluated"/*__eval_*/; do
    [ -d "$result" ] || continue
    name="$(basename "$result")"
    tag="${name%%__eval_*}"
    split="${name##*__eval_}"
    if [ ! -d "$out/$tag" ]; then
        echo "evaluation $name has no matching training run" >&2
        exit 1
    fi
    ln -s "$(readlink -f "$result")" "$out/$tag/eval_$split"
done

echo "assembled $(find "$out" -mindepth 1 -maxdepth 1 | wc -l) runs in $out"

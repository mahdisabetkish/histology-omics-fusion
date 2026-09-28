// Score one trained model on one split.
//
// src.evaluate writes eval_<split>/ inside the run directory it is given. The
// run is linked into a scratch directory so the training task's output is
// never modified, and only the new eval_<split>/ is emitted.
process EVALUATE {
    tag "${meta.tag}:${split}"
    label 'process_gpu'
    label 'process_short'

    input:
    tuple val(meta), path(run, stageAs: 'trained'), val(split)
    path processed

    output:
    tuple val(meta), path("${meta.tag}__eval_${split}"), emit: eval

    script:
    """
    mkdir scratch
    for entry in trained/*; do
        ln -s "\$(readlink -f "\$entry")" "scratch/\$(basename "\$entry")"
    done

    python -m src.evaluate \\
        --run scratch \\
        --processed ${processed} \\
        --split ${split} \\
        --workers ${Math.max(task.cpus - 1, 0)}

    mv scratch/eval_${split} ${meta.tag}__eval_${split}
    """

    stub:
    """
    mkdir -p ${meta.tag}__eval_${split}
    touch ${meta.tag}__eval_${split}/metrics.json ${meta.tag}__eval_${split}/predictions.csv
    """
}

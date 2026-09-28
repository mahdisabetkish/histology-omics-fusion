// One cell of the experiment matrix: a supervised classifier (src.train) or the
// expression-from-histology regression (src.train_expression), optionally
// initialised from a self-supervised checkpoint.
//
// Arguments from the matrix come after the pipeline-wide defaults, so a row
// that fixes its own --epochs or --lr wins.
process TRAIN {
    tag "${meta.tag}"
    label 'process_gpu'

    input:
    tuple val(meta), path(init, stageAs: 'init/*')
    path processed

    output:
    tuple val(meta), path("${meta.tag}"), emit: run

    script:
    def init_arg = init ? "--init ${init}" : ''
    """
    python -m src.${meta.entry} \\
        --processed ${processed} \\
        --out . \\
        --tag ${meta.tag} \\
        --epochs ${params.epochs} \\
        --workers ${Math.max(task.cpus - 1, 0)} \\
        --seed ${params.seed} \\
        ${init_arg} \\
        ${meta.args} ${params.train_extra_args}
    """

    stub:
    """
    mkdir -p ${meta.tag}
    touch ${meta.tag}/checkpoint.pt ${meta.tag}/summary.json
    """
}

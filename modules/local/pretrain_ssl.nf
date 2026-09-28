// Cross-modal contrastive pretraining. Reads no layer labels.
process PRETRAIN_SSL {
    tag "${tag}"
    label 'process_gpu'

    input:
    tuple val(tag), val(args)
    path processed

    output:
    tuple val(tag), path("${tag}"), emit: run

    script:
    """
    python -m src.pretrain_ssl \\
        --processed ${processed} \\
        --out . \\
        --tag ${tag} \\
        --epochs ${params.ssl_epochs} \\
        --batch-size ${params.ssl_batch_size} \\
        --workers ${Math.max(task.cpus - 1, 0)} \\
        --seed ${params.seed} \\
        ${args} ${params.ssl_extra_args}
    """

    stub:
    """
    mkdir -p ${tag}
    touch ${tag}/checkpoint.pt ${tag}/summary.json ${tag}/history.json ${tag}/probes.json
    """
}

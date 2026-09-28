// Normalise, select variable genes and standardise, with every statistic fitted
// on the training sections only.
//
// src.data.features writes its output next to its input. The per-section
// arrays are linked into a fresh directory rather than modified in place, so
// the upstream task stays intact and -resume can trust it.
process SELECT_FEATURES {
    tag "n_hvg=${params.n_hvg}"
    label 'process_medium'

    input:
    path built, stageAs: 'built'

    output:
    path 'processed', emit: processed

    script:
    """
    mkdir processed
    for entry in built/*; do
        ln -s "\$(readlink -f "\$entry")" "processed/\$(basename "\$entry")"
    done

    python -m src.data.features \\
        --processed processed \\
        --n-hvg ${params.n_hvg} \\
        --min-spots ${params.min_spots}
    """

    stub:
    """
    mkdir -p processed
    touch processed/expression.npy processed/expression_meta.npz processed/feature_config.json
    """
}

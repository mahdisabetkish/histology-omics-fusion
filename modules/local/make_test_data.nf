// A small synthetic stand-in for the output of BUILD_DATASET, so the whole
// pipeline can be exercised in minutes on a CPU (-profile test).
process MAKE_TEST_DATA {
    tag "synthetic"
    label 'process_single'

    output:
    path 'processed', emit: processed

    script:
    """
    make_test_data.py \\
        --out processed \\
        --spots ${params.test_spots} \\
        --genes ${params.test_genes} \\
        --patch-size ${params.patch_size} \\
        --seed ${params.seed}
    """

    stub:
    """
    mkdir -p processed
    """
}

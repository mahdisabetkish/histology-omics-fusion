// Crop an H&E patch around every spot and align it with the spot's counts and
// manual layer annotation.
process BUILD_DATASET {
    tag "spatialLIBD"
    label 'process_data'

    input:
    path raw
    path check_report

    output:
    path 'processed', emit: processed

    script:
    """
    python -m src.data.build_dataset \\
        --raw ${raw} \\
        --out processed \\
        --patch-size ${params.patch_size}
    """

    stub:
    """
    mkdir -p processed
    touch processed/index.csv processed/genes.csv
    """
}

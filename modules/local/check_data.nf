// Fail early on a truncated or corrupt download. A part-finished TIFF has the
// right size on disk and would otherwise only break halfway through cropping.
process CHECK_DATA {
    tag "spatialLIBD"
    label 'process_single'

    input:
    path raw, stageAs: 'data/raw'

    output:
    path 'check_data.txt', emit: report

    script:
    """
    python -m scripts.check_data | tee check_data.txt
    """

    stub:
    """
    touch check_data.txt
    """
}

// Fetch the raw spatialLIBD files. The transfer is resumable, and the host
// rather than the bandwidth sets the pace, so this is the one step worth
// caching outside the work directory (see params.raw).
process DOWNLOAD {
    tag "spatialLIBD"
    label 'process_data'
    label 'process_network'

    output:
    path 'raw', emit: raw

    script:
    """
    python -m src.data.download --out raw
    """

    stub:
    """
    mkdir -p raw
    """
}

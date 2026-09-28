// Rebuild the dashboard payload and the markdown results tables from the
// complete set of runs.
process EXPORT_RESULTS {
    tag "${runs.size()} runs"
    label 'process_export'

    input:
    path runs, stageAs: 'trained/*'
    path evals, stageAs: 'evaluated/*'
    path processed
    path raw, stageAs: 'raw'
    path site, stageAs: 'site/*'

    output:
    path 'docs', emit: docs
    path 'results_tables.md', emit: tables

    script:
    """
    assemble_runs.sh trained evaluated runs

    mkdir docs
    cp -rL site/. docs/

    python -m src.export_dashboard \\
        --processed ${processed} \\
        --raw raw \\
        --runs runs \\
        --out docs

    python -m scripts.make_report > results_tables.md
    """

    stub:
    """
    mkdir -p docs/data
    touch docs/data/manifest.json results_tables.md
    """
}

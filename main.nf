#!/usr/bin/env nextflow
/*
 * histology-omics-fusion
 *
 * Cortical layer prediction from paired Visium transcriptomes and H&E
 * histology on spatialLIBD, from raw download to the published dashboard.
 *
 *   nextflow run . -profile docker                  # full study, CPU
 *   nextflow run . -profile docker,gpu              # full study, one GPU per task
 *   nextflow run . -profile slurm,apptainer,gpu     # on a cluster
 *   nextflow run . -profile test,docker             # synthetic data, minutes on a laptop
 *
 * The run matrix lives in assets/pretraining.csv and assets/experiments.csv.
 * Adding a model is a new row there, not a code change.
 */

include { DOWNLOAD        } from './modules/local/download'
include { CHECK_DATA      } from './modules/local/check_data'
include { BUILD_DATASET   } from './modules/local/build_dataset'
include { MAKE_TEST_DATA  } from './modules/local/make_test_data'
include { SELECT_FEATURES } from './modules/local/select_features'
include { PRETRAIN_SSL    } from './modules/local/pretrain_ssl'
include { TRAIN           } from './modules/local/train'
include { EVALUATE        } from './modules/local/evaluate'
include { EXPORT_RESULTS  } from './modules/local/export_results'


/*
 * Raw files to model-ready arrays. Three ways in: synthetic data for the test
 * profile, a raw directory already on disk, or a fresh download.
 */
workflow PREPARE_DATA {
    main:
    raw = channel.empty()

    if( params.test ) {
        built = MAKE_TEST_DATA().processed
    }
    else {
        raw = params.raw
            ? channel.fromPath(params.raw, type: 'dir', checkIfExists: true)
            : DOWNLOAD().raw
        CHECK_DATA(raw)
        built = BUILD_DATASET(raw, CHECK_DATA.out.report).processed
    }

    processed = SELECT_FEATURES(built).processed

    emit:
    processed = processed
    raw       = raw
}


/*
 * Self-supervised pretraining, then every supervised and regression run in
 * the matrix. A run that names a checkpoint in the `init` column starts as
 * soon as that one pretraining run finishes, not after all of them.
 */
workflow TRAIN_MODELS {
    take:
    processed
    pretraining
    experiments

    main:
    PRETRAIN_SSL(pretraining, processed)

    checkpoints = PRETRAIN_SSL.out.run
        .map { tag, run -> [tag, run.resolve('checkpoint.pt')] }

    from_scratch = experiments
        .filter { meta -> !meta.init }
        .map { meta -> [meta, []] }

    from_checkpoint = experiments
        .filter { meta -> meta.init }
        .map { meta -> [meta.init, meta] }
        .combine(checkpoints, by: 0)
        .map { _init, meta, checkpoint -> [meta, checkpoint] }

    TRAIN(from_scratch.mix(from_checkpoint), processed)

    emit:
    ssl_runs = PRETRAIN_SSL.out.run
    runs     = TRAIN.out.run
}


/*
 * Score the selected runs on each split, then rebuild the dashboard and the
 * results tables from everything.
 */
workflow EVALUATE_AND_REPORT {
    take:
    processed
    raw
    ssl_runs
    runs

    main:
    splits = channel.fromList(params.eval_splits.tokenize(','))

    to_evaluate = runs
        .filter { meta, _run -> meta.evaluate }
        .combine(splits)

    EVALUATE(to_evaluate, processed)

    all_runs = runs.map { _meta, run -> run }
        .mix(ssl_runs.map { _tag, run -> run })
        .collect()

    all_evals = EVALUATE.out.eval
        .map { _meta, result -> result }
        .collect()
        .ifEmpty([])

    site = channel.fromPath(['index.html', 'css', 'js'].collect { name -> "${projectDir}/docs/${name}" },
                            checkIfExists: true)
        .collect()

    raw_or_none = raw.ifEmpty([]).first()

    EXPORT_RESULTS(all_runs, all_evals, processed, raw_or_none, site)

    emit:
    docs   = EXPORT_RESULTS.out.docs
    tables = EXPORT_RESULTS.out.tables
}


/*
 * Rows of the two run-matrix files, checked before anything is scheduled. A
 * mistyped init tag would otherwise drop its run silently, since the join on
 * the checkpoint simply never matches.
 */
def readMatrix(pretrainingFile, experimentsFile) {
    def pretraining = file(pretrainingFile, checkIfExists: true).splitCsv(header: true)
    def experiments = file(experimentsFile, checkIfExists: true).splitCsv(header: true)

    def sslTags = pretraining.collect { row -> row.tag }
    def entries = ['train', 'train_expression']
    def seen = [] as Set

    experiments.each { row ->
        if( !row.tag )
            error("${experimentsFile}: a row has no tag")
        if( !seen.add(row.tag) || row.tag in sslTags )
            error("${experimentsFile}: tag '${row.tag}' is used more than once")
        if( !(row.entry in entries) )
            error("${experimentsFile}: '${row.tag}' has entry '${row.entry}', expected one of ${entries}")
        if( row.init && !(row.init in sslTags) )
            error("${experimentsFile}: '${row.tag}' starts from '${row.init}', which is not in ${pretrainingFile}")
        if( !(row.evaluate in ['true', 'false']) )
            error("${experimentsFile}: '${row.tag}' has evaluate '${row.evaluate}', expected true or false")
    }

    def pretrainingRows = pretraining.collect { row -> [row.tag, row.args ?: ''] }
    def experimentRows = experiments.collect { row ->
        [
            tag     : row.tag,
            entry   : row.entry,
            init    : row.init ?: null,
            evaluate: row.evaluate == 'true',
            args    : row.args ?: '',
        ]
    }
    return [pretrainingRows, experimentRows]
}


workflow {
    main:
    log.info """
        histology-omics-fusion  v${workflow.manifest.version}
        ----------------------------------------------------
        input        : ${params.test ? 'synthetic test data' : (params.processed ?: (params.raw ?: 'download'))}
        pretraining  : ${params.pretraining}
        experiments  : ${params.experiments}
        epochs       : ${params.epochs} (ssl ${params.ssl_epochs})
        gpu          : ${params.gpu}
        outdir       : ${params.outdir}
        """.stripIndent()

    def (pretrainingRows, experimentRows) = readMatrix(params.pretraining, params.experiments)

    if( params.processed ) {
        processed = channel.fromPath(params.processed, type: 'dir', checkIfExists: true).first()
        raw = params.raw ? channel.fromPath(params.raw, type: 'dir', checkIfExists: true) : channel.empty()
    }
    else {
        PREPARE_DATA()
        processed = PREPARE_DATA.out.processed.first()
        raw = PREPARE_DATA.out.raw
    }

    TRAIN_MODELS(processed, channel.fromList(pretrainingRows), channel.fromList(experimentRows))

    if( params.export ) {
        EVALUATE_AND_REPORT(processed, raw, TRAIN_MODELS.out.ssl_runs, TRAIN_MODELS.out.runs)
    }

    workflow.onComplete = {
        log.info(workflow.success
            ? "done in ${workflow.duration}; results in ${params.outdir}"
            : "failed after ${workflow.duration}: ${workflow.errorMessage}")
    }
}

# Cortical layers from paired transcriptomes and histology

Every spot on a 10x Visium slide is measured twice over. The assay captures the
mRNA released by the handful of cells sitting under the spot, and the same
tissue is photographed before capture, so each spot carries a transcriptome and
a piece of H&E histology recorded at a known pixel coordinate. The pairing is
physical rather than inferred, which makes this a convenient place to ask a
question that is usually hard to answer cleanly: what does each modality
actually contribute, and does putting them together help?

The task is laminar identity. Human dorsolateral prefrontal cortex is arranged
in six layers over white matter, the layers differ in cell density, cell size
and gene expression, and two neuroanatomists annotated every spot in the
spatialLIBD dataset by hand. That gives a seven-class problem with real ground
truth on both modalities at once.

This repository trains three kinds of model on it: expression alone, histology
alone, and a fused model with a gate that weighs the two per spot. It then adds
a self-supervised stage that pretrains both encoders on the expression–image
pairing without reading a single layer label, and measures whether that
initialisation is worth anything downstream.

**[Interactive results dashboard](https://mahdisabetkish.github.io/histology-omics-fusion/)** |
**[Docker image](https://hub.docker.com/r/mahdisabetkish/histology-omics-fusion)**

### What came out of it

Fusing the two modalities did not help. The best fused model reaches 0.620
macro-F1 on a held-out donor against 0.619 for expression alone, and three
different fusion schemes all landed in the same place. Histology on its own
reaches 0.375, and the cross-modal regression shows why the combination gains
so little: only 4.2% of genes can be predicted from the crop above r = 0.3, and
those are almost all myelin genes. The image knows where the white matter is
and not much else about the transcriptome.

Self-supervised pretraining produced the clearest effect in the project, and
only at small label budgets. With 246 labelled spots it is worth +0.143
macro-F1; the advantage reverses by 1,200 spots and is 0.063 *worse* with the
full training set.

The reasoning behind both is in [Reading the
results](#reading-the-results).

---

## The data

[spatialLIBD](http://spatial.libd.org/spatialLIBD/) (Maynard et al., *Nature
Neuroscience* 24, 425–436, 2021). Twelve Visium sections from three neurotypical
donors, four sections each, taken as two adjacent pairs 300 µm apart in the
tissue block.

| | |
|---|---|
| Spots | 47,681 |
| Annotated | 47,329 (99.3%) |
| Genes measured | 33,538 |
| Genes modelled | 3,000 most variable, selected on training sections only |
| Image per spot | 112 px crop, cut at twice the spot diameter from the full-resolution slide |
| Classes | L1–L6, white matter |

Raw files come from the public `spatial-dlpfc` S3 bucket (counts, full-resolution
TIFFs) and the HumanPilot repository (spot coordinates, layer annotations). The
download is about 6.6 GB and the host throttles each connection hard, so
`src/data/download.py` pulls the large images as parallel byte ranges and
records completed ranges next to the file so an interrupted run resumes.

### Splitting

Donor Br8100 is held out completely. Nothing from it is used for fitting or for
choosing a checkpoint, and every number reported below is measured on it.

This matters more than it might seem. Sections from one donor were cut from the
same block, and the two members of an adjacent pair are close to the same cells
sliced twice. A random split over spots, or even over sections, puts nearly
identical tissue on both sides and produces numbers that say very little about
whether the model would work on a new brain.

Early stopping needs a validation set, and with only three donors, spending one
on it costs a third of the training data. One section is withheld from each
training donor instead. Those sections share a donor with the training data, so
their scores run optimistic; they are used to pick the stopping epoch and are
never reported as a result.

## Models

Expression goes through a two-layer MLP over the standardised variable-gene
panel. Images go through a ResNet-18 with the max pool removed from its stem,
which puts the first residual stage at 56×56 instead of 28×28. At a 112 px
input the stock stem downsamples fourfold before computing any features, and
nuclei are small enough that most of what distinguishes one layer from the next
is gone by then. Replacing the stem outright with a stride-1 convolution keeps
more again, but benchmarked about ten times slower for the whole study on the
GPU used here and throws away the pretrained stem weights, so it was not worth
it.

Three fusion schemes are compared. Concatenation is the baseline. The gated
version computes a per-spot weight for each modality from both embeddings, on
the reasoning that the two are not equally informative everywhere: expression
separates the middle layers well, histology is clearer at the white matter
boundary, and over a spot with poor RNA capture the expression vector is close
to noise. Low-rank bilinear pooling is included as an ablation; it is the most
expressive and the most eager to overfit.

### Self-supervised stage

The pretraining objective is symmetric InfoNCE over expression–image pairs with
a learned temperature. A spot's transcriptome and its own crop are the positive
pair, everything else in the batch is a negative, and no layer annotation is
read at any point.

One correction is needed. Visium spots sit 100 µm apart, close enough that a
spot and its immediate neighbours often contain overlapping cells. Left alone
the loss pushes those apart as hard negatives, working directly against the
structure the objective is supposed to find, so pairs within two lattice steps
on the same section are dropped from the denominator. The ablation without this
masking is reported.

A note on what this can and cannot show here. Self-supervised pretraining is
usually justified by unlabelled data that would otherwise go to waste, and that
argument does not apply to this dataset: the annotators covered 99.3% of spots,
leaving no meaningful unlabelled pool. What can be tested is whether the pairing
alone teaches the encoders enough that fewer labels are needed afterwards, and
whether the representation holds up with nothing fitted on top of it. Both are
measured, at seven label fractions from 1% to 100% and with frozen-encoder
probes, retrieval scores and a nearest-neighbour vote.

## Results

All figures are on donor Br8100, 14,243 annotated spots the models never saw.
Layer frequencies are uneven, so macro-F1 and balanced accuracy are the numbers
to read; plain accuracy is there for reference. `scripts/make_report.py`
regenerates these tables from the run artefacts.

| Model | Macro F1 | Balanced acc | Accuracy | κ | ARI | ±1 layer | Coherence |
|---|---|---|---|---|---|---|---|
| Expression only, SSL init ⁺ | **0.639** | **0.631** | 0.691 | **0.625** | **0.482** | 0.930 | 0.773 |
| Fusion, gated | 0.620 | 0.614 | 0.681 | 0.611 | 0.465 | 0.917 | 0.797 |
| Expression only | 0.619 | 0.608 | 0.685 | 0.614 | 0.467 | 0.919 | 0.775 |
| Fusion, concatenation | 0.618 | 0.603 | **0.694** | 0.624 | 0.472 | 0.908 | 0.796 |
| Fusion, bilinear | 0.597 | 0.581 | 0.681 | 0.607 | 0.448 | 0.897 | **0.802** |
| Fusion, gated, SSL init ⁺ | 0.560 | 0.555 | 0.640 | 0.559 | 0.408 | 0.884 | 0.779 |
| Frozen SSL, no neighbour masking ⁺ | 0.529 | 0.522 | 0.610 | 0.520 | 0.356 | 0.847 | 0.755 |
| Frozen SSL encoders + linear head ⁺ | 0.512 | 0.507 | 0.605 | 0.512 | 0.343 | 0.840 | 0.770 |
| Frozen SSL, no ImageNet init ⁺ | 0.508 | 0.501 | 0.567 | 0.468 | 0.299 | 0.826 | 0.705 |
| Histology only | 0.375 | 0.385 | 0.468 | 0.342 | 0.208 | 0.643 | 0.711 |
| Histology only, SSL init ⁺ | 0.357 | 0.371 | 0.444 | 0.310 | 0.193 | 0.641 | 0.753 |
| Histology only, no ImageNet init | 0.306 | 0.317 | 0.369 | 0.222 | 0.150 | 0.579 | 0.697 |

⁺ encoders initialised from the self-supervised checkpoint.
Spatial coherence of the manual annotation itself is 0.951.

### F1 by layer

| Model | L1 | L2 | L3 | L4 | L5 | L6 | WM |
|---|---|---|---|---|---|---|---|
| Expression only, SSL init | 0.54 | 0.56 | 0.72 | 0.42 | 0.74 | 0.65 | 0.85 |
| Fusion, gated | 0.44 | 0.59 | 0.71 | 0.41 | 0.76 | 0.63 | 0.81 |
| Expression only | 0.46 | 0.46 | 0.70 | 0.45 | 0.74 | 0.67 | 0.85 |
| Fusion, concatenation | 0.43 | 0.46 | 0.72 | 0.43 | 0.74 | 0.68 | 0.87 |
| Histology only | 0.29 | 0.17 | 0.54 | 0.15 | 0.21 | 0.39 | **0.87** |

### Label efficiency

Same architecture, same schedule, same data; only the initialisation differs.

| Labels | Spots | From scratch | SSL init | Δ |
|---|---|---|---|---|
| 1% | 246 | 0.209 | 0.353 | **+0.143** |
| 2% | 492 | 0.360 | 0.424 | **+0.064** |
| 5% | 1,230 | 0.525 | 0.496 | −0.029 |
| 10% | 2,460 | 0.584 | 0.494 | −0.090 |
| 25% | 6,151 | 0.596 | 0.550 | −0.046 |
| 50% | 12,302 | 0.595 | 0.546 | −0.048 |
| 100% | 24,603 | 0.624 | 0.560 | −0.063 |

### Expression predicted from histology

Mean Pearson r across the 3,000 modelled genes is 0.106, median 0.081. Only
4.2% of genes clear r = 0.3 and 0.9% clear r = 0.5. The ones that do are almost
entirely a single biological programme:

| Gene | r | | Gene | r |
|---|---|---|---|---|
| MBP | 0.76 | | CRYAB | 0.68 |
| PLP1 | 0.73 | | MAG | 0.63 |
| MOBP | 0.71 | | CLDND1 | 0.62 |
| CNP | 0.69 | | GFAP | 0.62 |
| TF | 0.68 | | CLDN11 | 0.60 |

## Reading the results

The fused model reaches 0.620 macro-F1 and expression alone reaches 0.619.
Whatever the histology contributes, the transcriptome was already carrying it.
Three fusion schemes finished within 0.023 of each other, so this is not a case
of having picked the wrong combination rule.

That does not make the images uninformative, only narrow in what they inform.
Histology on its own reaches 0.375, comfortably above the 0.143 a balanced guess
would give, and the per-class figures show where it earns that: white matter F1
of 0.87, higher than any expression model manages on that class, against 0.15
for layer 4 and 0.21 for layer 5. Myelinated tissue looks different under H&E.
Which of the six cortical layers you are standing in largely does not.

The gate reached the same conclusion without being told. It assigns 65% of its
weight to expression averaged over the held-out donor, but the split moves with
the class: 79% expression in layer 5, falling to 39% over white matter, where
morphology carries most of the decision.

Regressing expression from the crop puts a number on the same limit. Mean
correlation across the 3,000 modelled genes is 0.106, and only 4.2% clear
r = 0.3. The genes that do clear it are not scattered. MBP, PLP1, MOBP, CNP,
MAG, CLDN11 and TF are all myelin and oligodendrocyte transcripts, running up to
r = 0.76. H&E stains myelin, the model reads myelin, and the rest of the
transcriptome stays invisible to it.

Pretraining is worth a great deal when labels are scarce and nothing at all when
they are not. Starting from the contrastive checkpoint with 246 labelled spots
lifts macro-F1 from 0.209 to 0.353. By roughly 1,200 spots the two arms have
crossed, and on the full training set the pretrained model finishes 0.063 behind
one initialised from ImageNet. With eight annotated sections in hand the
pretraining is wasted effort. With one, it would be the difference between a
usable model and an unusable one.

Oddly, the same checkpoint improves the expression-only model at full labels,
0.619 to 0.639, while costing the fused model 0.060. My reading is that the
contrastive objective pulls the expression encoder toward the coarse structure
the two modalities share, which here is laminar, and leaves an image encoder
that starts from a worse place than ImageNet and drags the fusion down with it.
These runs do not test that, so it stays a guess.

Two things I expected to matter turned out not to. Excluding a spot's
neighbours from the contrastive negatives was meant to stop the loss tearing
apart near-duplicate tissue; with frozen encoders the masked model scores 0.512
against 0.529 unmasked, so the correction is at best unnecessary. Cross-modal
retrieval then fails outright across donors. On the held-out brain the correct
crop for a given expression vector sits at median rank 2,064 out of 4,096, which
is chance, even though the model matches its own training batches 95% of the
time. Whatever lets it pair one specific spot with one specific crop appears
tied to donor-specific staining rather than to anything a second brain shares.

One number sits behind all of this. Spatial coherence runs 0.71 to 0.80 for the
models against 0.951 for the manual annotation. The annotators drew continuous
bands; a classifier that sees one spot at a time produces speckle.

## Limitations

Three donors is not many. The held-out donor is a real test of transfer, but a
single held-out brain gives one sample of between-donor variation, and the
spread across the four test sections is the only handle on how much that number
would move on a different donor.

Sequencing depth differs systematically between donors, with median UMI counts
per spot running roughly 2,300, 3,450 and 4,000 for the three. Library-size
normalisation removes the first-order effect but not the differences in
detection rate that follow from it, and some of the gap between validation and
test scores is likely that rather than anything anatomical.

The layer annotations are expert judgement, not a physical measurement. Layer
boundaries in cortex are gradual, and a spot sitting on a boundary has no single
correct answer. Adjacent-layer accuracy is reported alongside plain accuracy for
that reason, and the spatial coherence of the manual annotation is reported as
the ceiling for the same statistic on predictions.

Nothing here uses spatial context. Each spot is classified on its own, while the
annotators drew contiguous bands and were plainly using the surrounding tissue.
A model with a neighbourhood term, which is what the graph-based spatial domain
methods do, should do better and would be the obvious next step.

## Dashboard

`docs/` is a static site, published through GitHub Pages, with no server or
build step behind it. Everything it draws is precomputed by
`src/export_dashboard.py` into JSON and image assets, so it stays up whether or
not anything is running locally.

It carries the section explorer (every spot on any of the twelve sections,
coloured by annotation, prediction, confidence, gate weight or marker gene, with
the tissue under the cursor shown as you hover), the model comparison table with
linked confusion matrix and training curves, the self-supervised panels and the
label-efficiency curve, and the cross-modal regression results.

## Layout

```
src/
  data/
    samples.py         section metadata, donor grouping, split definition
    download.py        parallel range downloader with resume
    build_dataset.py   crops patches, aligns counts and annotations
    features.py        normalisation and variable-gene selection
    dataset.py         torch datasets and per-modality augmentation
  models/
    encoders.py        expression MLP, image trunk, projection head
    fusion.py          concatenation, gated, bilinear
    multimodal.py      classifier, contrastive model, regression model
  train.py             supervised training
  pretrain_ssl.py      cross-modal contrastive pretraining
  train_expression.py  expression regressed from histology
  evaluate.py          held-out scoring and per-spot export
  export_dashboard.py  builds everything under docs/
scripts/
  run_all.py           the full experiment matrix, sequentially
  check_data.py        what is downloaded, complete and readable
  make_report.py       results tables as markdown, read off the run artefacts
slurm/                 the same matrix as dependent array jobs
docs/                  the published dashboard
```

## Running it

```bash
pip install -r requirements.txt

python -m src.data.download          # ~6.6 GB, resumable
python scripts/check_data.py         # confirm everything arrived intact
python -m src.data.build_dataset     # patches, counts, annotations
python -m src.data.features          # normalisation and gene selection

python scripts/run_all.py            # every model, skips what is done
python -m src.export_dashboard       # rebuild docs/
```

`run_all.py` takes `--stages` if you want a subset (`ssl`, `supervised`,
`transfer`, `efficiency`, `regression`, `evaluate`) and skips any run that
already has a `summary.json` unless given `--force`.

The full matrix is 31 training runs and takes about 6.4 hours of GPU time on the
hardware below, plus roughly two hours to download the raw data. Preprocessing
the twelve slides takes about fifteen minutes and needs around 8 GB of free
disk on top of the download.

### Docker

Two images are published to Docker Hub on every push to `main`: `cpu` (also
`latest`) and `gpu`, built with CUDA 12.6. Both pin the exact versions in
`requirements-lock.txt` and carry the ImageNet weights, so a container never
needs the network after the data stage. Stages are subcommands, and `data/`,
`runs/` and `docs/` are mounted from the checkout, so everything lands where
the commands above would put it:

```bash
mkdir -p data runs                     # before the first run, so you own them
docker compose run --rm hof info       # versions and device
docker compose run --rm hof data       # download + check + preprocess
docker compose run --rm hof train      # run_all.py; takes the same flags
docker compose run --rm hof export     # score, rebuild docs/, print tables
docker compose up dashboard            # docs/ at http://localhost:8080
```

On an NVIDIA machine with the Container Toolkit installed, use the `hof-gpu`
service instead (`docker compose --profile gpu run --rm hof-gpu train`).
Without Compose:

```bash
docker run --rm --shm-size=2g --gpus all \
  -v "$PWD/data:/app/data" -v "$PWD/runs:/app/runs" -v "$PWD/docs:/app/docs" \
  mahdisabetkish/histology-omics-fusion:gpu pipeline
```

`--shm-size` matters: DataLoader workers hand batches over through shared
memory, and Docker's 64 MB default is too small for image batches. Any command
the entrypoint does not recognise runs as given, so `python -m src.train ...`
works unchanged. To build locally, `docker compose build`, or
`docker build --build-arg TORCH_VARIANT=cu126 .` for the GPU image.
`.github/workflows/docker.yml` builds both, smoke-tests them offline and pushes
them with SBOM and provenance attestations.

### On a cluster

The experiment matrix is embarrassingly parallel: once pretraining is finished,
the runs share nothing and each writes to its own `runs/<tag>/`. `slurm/`
submits the whole thing as three dependent array jobs.

```bash
bash slurm/submit_all.sh              # data, pretrain, train matrix, export
bash slurm/submit_all.sh --skip-data  # if data/ is already populated
```

Edit `slurm/_env.sh` for your site's modules or conda environment; the job
scripts source it and do not otherwise hard-code anything local. Resubmitting a
failed array task is safe, since every stage skips runs that already produced a
`summary.json`. Point `TORCH_HOME` at shared storage so array tasks share one
copy of the ImageNet weights, which also matters if your compute nodes have no
outbound network.

Peak GPU memory is about 4 GB at the default batch sizes, so a small allocation
is enough. `slurm/README.md` has the per-stage detail.

### Environment and determinism

Results here were produced on a single workstation, not on a cluster:

| | |
|---|---|
| GPU | NVIDIA GeForce GTX 1080 Ti, 11 GB, driver 560.94 |
| CPU | Intel Xeon E5-2697 v4, 18 cores |
| RAM | 32 GB |
| OS | Windows 10, Python 3.11.8 |
| PyTorch | 2.7.1+cu126, cuDNN 9.7.1 |

`requirements-lock.txt` pins the exact versions; `requirements.txt` gives looser
ranges and `environment.yml` builds the same thing under conda.

Two hardware notes came out of benchmarking on this GPU and are worth knowing
before you port the code. Pascal cards have no tensor cores, so mixed precision
buys only a few percent rather than the usual factor, and `channels_last`
memory format is roughly six times *slower* than contiguous because its kernels
target Ampere and later. Both defaults in this repository reflect that. On a
newer card, turning `channels_last` on is likely to be a substantial win, and
the benchmark that decided it is easy to repeat.

Every entry point takes `--seed` and seeds Python, NumPy and Torch from it.
Runs are not bitwise deterministic: `torch.backends.cudnn.benchmark` is left on,
which lets cuDNN pick algorithms by timing and costs about a third of the
throughput to disable. Set `torch.backends.cudnn.deterministic = True` and
`benchmark = False` in `src/utils/common.py` if you need exact reproduction.
Run-to-run variation in held-out macro-F1 from seed alone is small relative to
the differences between models discussed above, but it is not zero, and single
runs at one seed are the main reason the smaller gaps in the tables should not
be over-read.

Nothing in the pipeline touches the network after the data stage, apart from
torchvision fetching ImageNet weights on first use.

## Reference

Maynard KR, Collado-Torres L, Weber LM, et al. Transcriptome-scale spatial gene
expression in the human dorsolateral prefrontal cortex. *Nature Neuroscience*
24, 425–436 (2021).

Data is distributed by the Lieber Institute for Brain Development under the
terms given in the [HumanPilot
repository](https://github.com/LieberInstitute/HumanPilot). Code in this
repository is MIT licensed.

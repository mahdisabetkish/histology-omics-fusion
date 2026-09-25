# histology-omics-fusion

Multimodal fusion of spatial transcriptomics and H&E histology for cortical
layer prediction on the spatialLIBD DLPFC dataset, with self-supervised
cross-modal pretraining. Code, results and write-up:
**[github.com/mahdisabetkish/histology-omics-fusion](https://github.com/mahdisabetkish/histology-omics-fusion)** |
**[interactive dashboard](https://mahdisabetkish.github.io/histology-omics-fusion/)**

## Tags

| Tag | PyTorch | Use it for |
|---|---|---|
| `latest`, `cpu` | 2.7.1 CPU | any machine; everything runs, training is slow |
| `gpu` | 2.7.1 + CUDA 12.6 | NVIDIA GPUs (driver 525 or newer and the NVIDIA Container Toolkit) |
| `X.Y.Z-cpu`, `X.Y.Z-gpu` | | pinned releases |
| `sha-<commit>-cpu`, `sha-<commit>-gpu` | | an exact commit |

Both images carry Python 3.11, the exact dependency versions from
`requirements-lock.txt`, and the ImageNet ResNet-18 weights, so nothing needs
the network after the data download. They run as an unprivileged user (UID
1000) and are published with SBOM and provenance attestations.

## Quick start

```bash
docker run --rm mahdisabetkish/histology-omics-fusion info

mkdir -p data runs docs
docker run --rm --shm-size=2g \
  -v "$PWD/data:/app/data" -v "$PWD/runs:/app/runs" -v "$PWD/docs:/app/docs" \
  mahdisabetkish/histology-omics-fusion data        # download + preprocess, ~9 GB

docker run --rm --shm-size=2g --gpus all \
  -v "$PWD/data:/app/data" -v "$PWD/runs:/app/runs" -v "$PWD/docs:/app/docs" \
  mahdisabetkish/histology-omics-fusion:gpu train   # all 31 runs
```

## Commands

| Command | Does |
|---|---|
| `download` | fetch the raw spatialLIBD files (~6.6 GB, resumable) |
| `check` | report which raw files are present and intact |
| `prepare` | crop patches, align counts and annotations, select genes |
| `train` | the experiment matrix (`scripts/run_all.py`, takes its flags) |
| `export` | score every checkpoint, rebuild `docs/`, print the results tables |
| `data` | `download` + `check` + `prepare` |
| `pipeline` | the whole study end to end |
| `info` | versions, device and threads |
| anything else | run as given, e.g. `python -m src.train --help` or `bash` |

Mount `data/`, `runs/` and `docs/` or the results leave with the container.
`--shm-size=2g` gives the DataLoader workers room to pass image batches.

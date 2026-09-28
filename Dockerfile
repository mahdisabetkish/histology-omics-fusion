# One Dockerfile, two images. TORCH_VARIANT picks the PyTorch wheel index:
#
#   docker build -t histology-omics-fusion:cpu .                                  # CPU only
#   docker build -t histology-omics-fusion:gpu --build-arg TORCH_VARIANT=cu126 .  # CUDA 12.6
#
# The CUDA wheels bundle their own CUDA and cuDNN libraries, so both images sit
# on the same slim Python base. The GPU image only needs the NVIDIA driver and
# the NVIDIA Container Toolkit on the host, not a CUDA base image.
#
# Stages are stacked rather than copied between: every dependency is a wheel,
# so there is no compiler to leave behind, and stacking means a code change
# rebuilds only the last thin layer while the multi-gigabyte dependency layers
# come from cache.

ARG PYTHON_VERSION=3.11

# --------------------------------------------------------------------------
# base: OS packages and the unprivileged user
# --------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS base

# tini reaps DataLoader worker processes and forwards Ctrl+C to Python.
# procps provides ps, which Nextflow needs to record per-task CPU and memory.
RUN apt-get update \
 && apt-get install -y --no-install-recommends tini procps \
 && rm -rf /var/lib/apt/lists/*

# UID 1000 matches the first user on most Linux hosts, so files written to
# bind-mounted data/, runs/ and docs/ stay owned by you. Any other UID works
# too (docker run --user "$(id -u):$(id -g)"): nothing the container writes
# lives outside those mounts and /tmp.
RUN groupadd --gid 1000 app \
 && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash app

ENV PATH=/opt/venv/bin:$PATH \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_HOME=/opt/torch \
    NUMBA_CACHE_DIR=/tmp/numba \
    MPLCONFIGDIR=/tmp/matplotlib

# --------------------------------------------------------------------------
# deps: the Python environment and the pretrained weights
# --------------------------------------------------------------------------
FROM base AS deps

ARG TORCH_VARIANT=cpu

# PyTorch first, from the index matching the requested variant. The lock file
# pins torch==2.7.1 too, which the +cpu / +cu126 local build already satisfies,
# so the second install leaves it alone. imagecodecs is not in the lock file;
# 2025.8.2 is the last release with Python 3.11 wheels.
COPY requirements-lock.txt /tmp/requirements-lock.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv /opt/venv \
 && pip install --index-url "https://download.pytorch.org/whl/${TORCH_VARIANT}" \
        torch==2.7.1 torchvision==0.22.1 \
 && pip install -r /tmp/requirements-lock.txt imagecodecs==2025.8.2 \
 && rm /tmp/requirements-lock.txt

# Bake the ImageNet ResNet-18 weights in, so containers never touch the network
# after the data stage (the same reason slurm/_env.sh points TORCH_HOME at
# shared storage).
RUN python -c "import torchvision; torchvision.models.resnet18(weights='IMAGENET1K_V1')" \
 && chmod -R a+rX /opt/torch

# --------------------------------------------------------------------------
# runtime: the code on top
# --------------------------------------------------------------------------
FROM deps AS runtime

ARG TORCH_VARIANT=cpu
ARG VERSION=dev
ARG REVISION=unknown

# CI overrides these with the values from docker/metadata-action.
LABEL org.opencontainers.image.title="histology-omics-fusion" \
      org.opencontainers.image.description="Multimodal fusion of spatial transcriptomics and histology for cortical layer prediction, with self-supervised pretraining." \
      org.opencontainers.image.source="https://github.com/mahdisabetkish/histology-omics-fusion" \
      org.opencontainers.image.url="https://mahdisabetkish.github.io/histology-omics-fusion/" \
      org.opencontainers.image.documentation="https://github.com/mahdisabetkish/histology-omics-fusion#docker" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}"

ENV HOF_TORCH_VARIANT=${TORCH_VARIANT}

WORKDIR /app

COPY --chmod=755 docker/entrypoint.sh /usr/local/bin/hof
COPY scripts/ scripts/
COPY src/ src/
RUN mkdir -p data runs docs && chown app:app data runs docs

USER app

ENTRYPOINT ["tini", "--", "hof"]
CMD ["help"]

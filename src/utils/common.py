"""Small helpers shared by the training and evaluation scripts."""

import json
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Input shapes are fixed here, so letting cuDNN pick kernels once is free
    # speed. Deliberately not setting deterministic mode: it costs roughly a
    # third of the throughput and the conclusions do not turn on bitwise
    # reproducibility.
    torch.backends.cudnn.benchmark = True


def get_device(prefer_cuda=True):
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe_device(device):
    if device.type != "cuda":
        return "cpu"
    index = device.index or 0
    name = torch.cuda.get_device_name(index)
    total = torch.cuda.get_device_properties(index).total_memory / 1e9
    return f"{name} ({total:.1f} GB)"


def count_parameters(model, trainable_only=True):
    return sum(p.numel() for p in model.parameters()
               if p.requires_grad or not trainable_only)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_encode))
    return path


def read_json(path):
    return json.loads(Path(path).read_text())


def _encode(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialise {type(value)}")


class RunningMean:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.total += float(value) * n
        self.count += n

    @property
    def value(self):
        return self.total / self.count if self.count else 0.0

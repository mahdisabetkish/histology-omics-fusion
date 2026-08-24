"""Paired spot dataset and the augmentations used for each modality.

One item is a single Visium spot: its standardised expression vector, the H&E
crop centred on it, and the layer label when the annotators assigned one.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Channel statistics of the training crops. H&E is nowhere near ImageNet in
# colour distribution, so the usual ImageNet constants are a poor fit even when
# the backbone starts from ImageNet weights.
HE_MEAN = np.array([0.7180, 0.5564, 0.6743], dtype=np.float32)
HE_STD = np.array([0.1832, 0.2211, 0.1743], dtype=np.float32)


class ImageAugment:
    """Flips, right-angle rotations and a mild stain jitter.

    A tissue section has no canonical orientation, so the dihedral group is
    label preserving here. The colour jitter stands in for slide-to-slide
    variation in haematoxylin and eosin intensity, which is the main nuisance
    factor when a model trained on one brain is applied to another.
    """

    def __init__(self, flip=True, rotate=True, jitter=0.25, gray=0.0):
        self.flip = flip
        self.rotate = rotate
        self.jitter = jitter
        self.gray = gray

    def __call__(self, patch, rng):
        if self.rotate:
            patch = np.rot90(patch, rng.integers(4), axes=(0, 1))
        if self.flip and rng.random() < 0.5:
            patch = patch[:, ::-1]
        patch = np.ascontiguousarray(patch).astype(np.float32) / 255.0

        if self.jitter > 0:
            scale = 1.0 + rng.uniform(-self.jitter, self.jitter, size=3).astype(np.float32)
            shift = rng.uniform(-self.jitter, self.jitter, size=3).astype(np.float32) * 0.1
            patch = np.clip(patch * scale + shift, 0.0, 1.0)
        if self.gray > 0 and rng.random() < self.gray:
            patch = np.repeat(patch.mean(axis=2, keepdims=True), 3, axis=2)

        patch = (patch - HE_MEAN) / HE_STD
        return torch.from_numpy(np.ascontiguousarray(patch.transpose(2, 0, 1)))


class ExpressionAugment:
    """Gene dropout plus multiplicative noise.

    Dropout imitates the incomplete capture that makes two spots of the same
    cell type look different, and the noise loosens the exact magnitudes so the
    encoder leans on which genes are on rather than on their precise values.
    """

    def __init__(self, dropout=0.2, noise=0.1):
        self.dropout = dropout
        self.noise = noise

    def __call__(self, vector, rng):
        vector = vector.astype(np.float32, copy=True)
        if self.dropout > 0:
            mask = rng.random(vector.shape[0]) < self.dropout
            vector[mask] = 0.0
        if self.noise > 0:
            vector *= (1.0 + rng.normal(0.0, self.noise, vector.shape[0])).astype(np.float32)
        return torch.from_numpy(vector)


def identity_image(patch, rng=None):
    patch = patch.astype(np.float32) / 255.0
    patch = (patch - HE_MEAN) / HE_STD
    return torch.from_numpy(np.ascontiguousarray(patch.transpose(2, 0, 1)))


def identity_expression(vector, rng=None):
    return torch.from_numpy(vector.astype(np.float32, copy=True))


class SpotDataset(Dataset):
    """Spots from one split, optionally restricted to annotated ones.

    The expression panel and the twelve patch arrays are together about 2.3 GB
    and are opened as memory maps rather than read into RAM. They are opened
    lazily, per process, because DataLoader workers on Windows are spawned and
    receive the dataset by pickle: a `np.memmap` attribute pickles its whole
    contents rather than a reference to the file, which either blows up the
    pipe or silently copies gigabytes into every worker. Only the paths cross
    the process boundary; `__getstate__` drops any handle already opened.
    """

    def __init__(self, root, split, labelled_only=True, image_augment=None,
                 expression_augment=None, subset=None, seed=0):
        root = Path(root)
        self.root = root
        index = pd.read_csv(root / "index.csv", dtype={"section": str})
        self._panel = None
        self._patches = None
        store = np.load(root / "expression_meta.npz", allow_pickle=False)
        self.hvg_names = store["hvg_names"].astype(str)
        labels = store["labels"]

        # pandas hands back a read-only view here, so take a copy before the
        # in-place narrowing below.
        keep = np.ones(len(index), dtype=bool) if split == "all" else \
            np.array((index["split"] == split).to_numpy(), dtype=bool)
        if labelled_only:
            keep &= labels >= 0
        rows = np.flatnonzero(keep)
        if subset is not None:
            rows = rows[np.asarray(subset)]

        self.rows = rows
        self.meta = index.iloc[rows].reset_index(drop=True)
        self.labels = labels[rows]

        # Row i of the expression panel is row i of index.csv, and the patch
        # arrays are stored per section. Offsets are derived from index.csv
        # rather than assumed, and each section has to occupy one contiguous
        # block for the mapping to be a simple subtraction.
        sections = index["section"].astype(str).to_numpy()
        self.section_order = list(dict.fromkeys(sections))
        offsets, start = {}, 0
        for section_id in self.section_order:
            where = np.flatnonzero(sections == section_id)
            if where[-1] - where[0] + 1 != len(where):
                raise ValueError(f"section {section_id} is not contiguous in index.csv")
            offsets[section_id] = where[0]
            start += len(where)

        for section_id in self.section_order:
            header = np.load(root / section_id / "patches.npy", mmap_mode="r")
            expected = int((sections == section_id).sum())
            if header.shape[0] != expected:
                raise ValueError(f"{section_id}: {header.shape[0]} patches for "
                                 f"{expected} rows in index.csv")
            del header

        self.section = sections[rows]
        self.local = rows - np.array([offsets[s] for s in self.section])
        self.section_index = np.array([self.section_order.index(s)
                                       for s in self.section])

        self.image_augment = image_augment or identity_image
        self.expression_augment = expression_augment or identity_expression
        self.seed = seed

    @property
    def panel(self):
        if self._panel is None:
            self._panel = np.load(self.root / "expression.npy", mmap_mode="r")
        return self._panel

    @property
    def patches(self):
        if self._patches is None:
            self._patches = {s: np.load(self.root / s / "patches.npy", mmap_mode="r")
                             for s in self.section_order}
        return self._patches

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_panel"] = None
        state["_patches"] = None
        return state

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, int(self.rows[i]),
                                     torch.initial_seed() % (1 << 31)))
        patch = np.asarray(self.patches[self.section[i]][self.local[i]])
        vector = self.panel[self.rows[i]]
        return {
            "image": self.image_augment(patch, rng),
            "expression": self.expression_augment(vector, rng),
            "label": int(self.labels[i]),
            "row": int(self.rows[i]),
        }

    @property
    def n_genes(self):
        return self.panel.shape[1]

    def class_counts(self, n_classes=7):
        valid = self.labels[self.labels >= 0]
        return np.bincount(valid, minlength=n_classes)


class PairedSSLDataset(SpotDataset):
    """Two independently augmented views of the same spot, one per modality.

    Adjacent Visium spots sit 100 um apart and often share cell populations, so
    a spot's own neighbours are near-duplicates rather than true negatives. The
    array coordinates are returned alongside each item so the contrastive loss
    can exclude them from the denominator.
    """

    def __getitem__(self, i):
        item = super().__getitem__(i)
        item["array_row"] = int(self.meta["array_row"].iloc[i])
        item["array_col"] = int(self.meta["array_col"].iloc[i])
        item["section_index"] = int(self.section_index[i])
        return item

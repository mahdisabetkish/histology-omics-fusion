#!/usr/bin/env python3
"""Write a small synthetic dataset in the layout src.data.build_dataset produces.

Three sections, one per split, each a grid of spots whose layer is a band of
rows. Expression carries a layer-specific signature over a subset of genes and
each H&E patch a layer-dependent tint, so every model has something to learn
and the metrics come out well away from chance. That is enough to catch wiring
errors between stages; it says nothing about the real data.

    make_test_data.py --out processed --spots 400
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.data.samples import LAYERS, donor_of, split_of

# The first training, validation and test section of the real study, so the
# split each section falls into is the one src.data.samples assigns it.
SECTIONS = ["151507", "151510", "151673"]
MARKERS = ["MBP", "PLP1", "MOBP", "SNAP25", "PCP4", "CUX2", "RORB", "GFAP"]


def make_section(section_id, n_spots, n_genes, patch_size, signatures, rng):
    side = int(np.ceil(np.sqrt(n_spots)))
    rows, cols = np.divmod(np.arange(n_spots), side)
    layer_index = np.minimum(rows * len(LAYERS) // side, len(LAYERS) - 1)

    rates = np.exp(signatures[layer_index] + rng.normal(0, 0.3, (n_spots, n_genes)))
    depth = rng.uniform(0.5, 1.5, n_spots)[:, None]
    counts = rng.poisson(rates * depth).astype(np.float32)

    tint = np.linspace(60, 200, len(LAYERS))[layer_index]
    base = rng.normal(0, 25, (n_spots, patch_size, patch_size, 3))
    base[..., 0] += tint[:, None, None]
    base[..., 1] += 255 - tint[:, None, None]
    base[..., 2] += 140
    patches = np.clip(base, 0, 255).astype(np.uint8)

    # A share of spots is left unannotated, as in the real sections, so the
    # self-supervised stage has unlabelled data to use.
    layer = np.array(LAYERS, dtype=object)[layer_index]
    layer[rng.random(n_spots) < 0.1] = None

    meta = pd.DataFrame({
        "barcode": [f"S{section_id}-{i:05d}" for i in range(n_spots)],
        "section": section_id,
        "donor": donor_of(section_id),
        "split": split_of(section_id),
        "array_row": rows * 2,
        "array_col": cols * 2 + (rows % 2),
        "px_row": rows * 150 + 500,
        "px_col": cols * 150 + 500,
        "layer": layer,
        "total_counts": counts.sum(axis=1),
        "n_genes": (counts > 0).sum(axis=1),
    })
    return meta, counts, patches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="processed")
    parser.add_argument("--spots", type=int, default=400, help="spots per section")
    parser.add_argument("--genes", type=int, default=400)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    names = MARKERS + [f"GENE{i:05d}" for i in range(args.genes - len(MARKERS))]
    pd.DataFrame({"name": names,
                  "ensembl_id": [f"ENSG{i:011d}" for i in range(args.genes)]}
                 ).to_csv(out / "genes.csv", index=False)

    signatures = rng.normal(0.5, 0.2, (len(LAYERS), args.genes))
    informative = rng.choice(args.genes, size=args.genes // 4, replace=False)
    signatures[:, informative] += rng.normal(0, 1.0, (len(LAYERS), len(informative)))

    frames = []
    for section_id in SECTIONS:
        meta, counts, patches = make_section(section_id, args.spots, args.genes,
                                             args.patch_size, signatures, rng)
        folder = out / section_id
        folder.mkdir(exist_ok=True)
        np.save(folder / "patches.npy", patches)
        sp.save_npz(folder / "counts.npz", sp.csr_matrix(counts))
        meta.to_csv(folder / "meta.csv", index=False)
        frames.append(meta)
        print(f"  {section_id}: {len(meta)} spots, {meta.layer.notna().sum()} labelled")

    pd.concat(frames, ignore_index=True).to_csv(out / "index.csv", index=False)


if __name__ == "__main__":
    main()

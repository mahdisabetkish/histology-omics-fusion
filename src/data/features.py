"""Build the model-ready expression matrix.

Everything that has to be estimated from data (which genes to keep, the mean
and scale used to standardise them) is estimated on the training sections
alone. The validation and test sections are then transformed with those fixed
statistics. Selecting variable genes on the pooled data first is an easy
mistake to make here and it leaks section-level structure into the split.

    python -m src.data.features --n-hvg 3000
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .samples import LAYER_TO_INDEX


def normalize_counts(counts, target=None):
    """Library-size normalise to a common depth, then log1p.

    `target` is the median library size of the training spots. Passing the
    training value when transforming val/test keeps the scaling identical
    across splits.
    """
    totals = np.asarray(counts.sum(axis=1)).ravel()
    totals[totals == 0] = 1.0
    if target is None:
        target = float(np.median(totals))
    scaled = sp.diags(target / totals) @ counts
    scaled = scaled.tocsr()
    scaled.data = np.log1p(scaled.data)
    return scaled, target


def select_variable_genes(matrix, n_top, min_spots, gene_names):
    """Rank genes by variance of the log-normalised values.

    Genes detected in very few spots are dropped first. Their variance is
    dominated by a handful of spots and they add noise rather than signal.
    """
    detected = np.asarray((matrix > 0).sum(axis=0)).ravel()
    eligible = np.flatnonzero(detected >= min_spots)

    subset = matrix[:, eligible]
    mean = np.asarray(subset.mean(axis=0)).ravel()
    mean_square = np.asarray(subset.multiply(subset).mean(axis=0)).ravel()
    variance = np.maximum(mean_square - mean ** 2, 0.0)

    # Mitochondrial and ribosomal genes track dissection and RNA quality more
    # than cortical identity, so they are excluded from the panel.
    names = gene_names[eligible]
    technical = np.array([
        n.startswith("MT-") or n.startswith("RPL") or n.startswith("RPS")
        for n in names
    ])
    variance[technical] = -np.inf

    order = np.argsort(-variance)[:n_top]
    return np.sort(eligible[order])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--n-hvg", type=int, default=3000)
    parser.add_argument("--min-spots", type=int, default=50)
    parser.add_argument("--clip", type=float, default=10.0,
                        help="clip standardised values to +/- this many SDs")
    args = parser.parse_args()

    root = Path(args.processed)
    index = pd.read_csv(root / "index.csv", dtype={"section": str})
    genes = pd.read_csv(root / "genes.csv")
    gene_names = genes["name"].to_numpy().astype(str)

    print("loading counts")
    # Follow index.csv rather than the full section list, so a partial build
    # still works and the row order is guaranteed to line up.
    section_order = list(dict.fromkeys(index["section"].astype(str)))
    blocks, order = [], []
    for section_id in section_order:
        block = sp.load_npz(root / section_id / "counts.npz").tocsr()
        blocks.append(block)
        order.extend([section_id] * block.shape[0])
    counts = sp.vstack(blocks).tocsr()

    if list(index["section"].astype(str)) != order:
        raise ValueError("index.csv row order does not match the count blocks")

    is_train = (index["split"] == "train").to_numpy()
    print(f"{counts.shape[0]} spots x {counts.shape[1]} genes "
          f"({is_train.sum()} training)")

    train_norm, depth = normalize_counts(counts[is_train])
    hvg = select_variable_genes(train_norm, args.n_hvg, args.min_spots, gene_names)
    print(f"kept {len(hvg)} variable genes, library depth {depth:.0f}")

    all_norm, _ = normalize_counts(counts, target=depth)
    panel = np.asarray(all_norm[:, hvg].todense(), dtype=np.float32)

    mean = panel[is_train].mean(axis=0)
    std = panel[is_train].std(axis=0)
    std[std < 1e-6] = 1.0
    panel = np.clip((panel - mean) / std, -args.clip, args.clip).astype(np.float32)

    labels = index["layer"].map(LAYER_TO_INDEX).fillna(-1).to_numpy().astype(np.int64)

    # The panel goes to a bare .npy so it can be memory-mapped. Inside a .npz
    # it would have to be decompressed into RAM by every process that opens it,
    # and DataLoader workers on Windows are separate processes that receive the
    # dataset by pickle, which a 500 MB array does not survive.
    np.save(root / "expression.npy", panel)
    np.savez_compressed(
        root / "expression_meta.npz",
        hvg_index=hvg,
        hvg_names=gene_names[hvg],
        gene_mean=mean,
        gene_std=std,
        labels=labels,
    )
    (root / "feature_config.json").write_text(json.dumps({
        "n_hvg": int(len(hvg)),
        "min_spots": args.min_spots,
        "library_depth": depth,
        "clip": args.clip,
        "n_spots": int(panel.shape[0]),
        "n_labelled": int((labels >= 0).sum()),
    }, indent=2))

    print(f"wrote expression.npy  {panel.shape}  "
          f"({(labels >= 0).sum()} labelled spots)")


if __name__ == "__main__":
    main()

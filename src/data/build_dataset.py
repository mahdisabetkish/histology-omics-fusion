"""Turn the raw Visium files into paired (expression, image patch) arrays.

For every spot under the tissue we keep three things: its UMI count vector, a
square H&E crop centred on the spot in full-resolution pixel space, and the
manual layer call where one exists. Spots the annotators left blank are kept
and marked as unlabelled, because the self-supervised stage can still use them.

The crop side is a multiple of `spot_diameter_fullres` rather than a fixed
pixel count. Spot diameter varies by a few percent between sections, so a fixed
crop would cover slightly different amounts of tissue in each one, and the
model could pick that up as a section cue.

    python -m src.data.build_dataset
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import tifffile
from PIL import Image

from .samples import SECTIONS, donor_of, split_of

# Full-resolution TIFFs are far above Pillow's decompression-bomb threshold.
Image.MAX_IMAGE_PIXELS = None


def read_counts(h5_path):
    """Read a CellRanger/SpaceRanger filtered matrix as (barcodes, genes) CSR."""
    with h5py.File(h5_path, "r") as handle:
        group = handle["matrix"] if "matrix" in handle else handle[list(handle)[0]]
        data = group["data"][:]
        indices = group["indices"][:]
        indptr = group["indptr"][:]
        n_genes, n_spots = group["shape"][:]
        features = group["features"] if "features" in group else group
        names = features["name"][:].astype(str)
        ids = features["id"][:].astype(str)
        barcodes = group["barcodes"][:].astype(str)

    # The file stores genes x spots in CSC, which is spots x genes in CSR.
    counts = sp.csr_matrix((data, indices, indptr), shape=(n_spots, n_genes))
    return counts, barcodes, names, ids


def read_positions(path):
    columns = ["barcode", "in_tissue", "array_row", "array_col", "px_row", "px_col"]
    frame = pd.read_csv(path, header=None, names=columns)
    return frame.set_index("barcode")


def read_layers(path):
    frame = pd.read_csv(path, sep="\t", header=None,
                        names=["barcode", "section", "layer"], dtype=str)
    frame["key"] = frame["section"] + "_" + frame["barcode"]
    return frame.set_index("key")["layer"]


def crop_patches(image, rows, cols, side, out_size):
    """Cut `side`-pixel squares centred on each (row, col) and resize them.

    Spots near the edge of the slide are clamped inward so the crop stays on
    the image. Fewer than a handful of spots per section are affected.
    """
    height, width = image.shape[:2]
    half = side // 2
    patches = np.empty((len(rows), out_size, out_size, 3), dtype=np.uint8)

    for i, (row, col) in enumerate(zip(rows, cols)):
        top = int(np.clip(row - half, 0, height - side))
        left = int(np.clip(col - half, 0, width - side))
        window = image[top:top + side, left:left + side]
        if window.shape[0] != side or window.shape[1] != side:
            pad_y = side - window.shape[0]
            pad_x = side - window.shape[1]
            window = np.pad(window, ((0, pad_y), (0, pad_x), (0, 0)), mode="edge")
        patches[i] = np.asarray(
            Image.fromarray(window).resize((out_size, out_size), Image.BILINEAR)
        )
    return patches


def load_image(path):
    # A download that stopped part-way leaves a file of the right length whose
    # missing ranges read as zeros, so check the magic number before trusting it.
    with open(path, "rb") as handle:
        magic = handle.read(4)
    if magic[:2] not in (b"II", b"MM"):
        raise ValueError(
            f"{path} is not a readable TIFF (header {magic!r}). The download is "
            f"probably incomplete; re-run `python -m src.data.download` to finish it.")

    image = tifffile.imread(str(path))
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.shape[0] in (3, 4) and image.ndim == 3 and image.shape[0] < image.shape[-1]:
        image = np.moveaxis(image, 0, -1)
    return np.ascontiguousarray(image[:, :, :3])


def build_section(section_id, raw_dir, out_dir, layers, context, out_size):
    raw = raw_dir / section_id
    out = out_dir / section_id
    out.mkdir(parents=True, exist_ok=True)

    counts, barcodes, gene_names, gene_ids = read_counts(
        raw / "filtered_feature_bc_matrix.h5")
    positions = read_positions(raw / "tissue_positions_list.txt")
    scale = json.loads((raw / "scalefactors_json.json").read_text())
    spot_diameter = scale["spot_diameter_fullres"]

    positions = positions.reindex(barcodes)
    missing = positions["px_row"].isna().sum()
    if missing:
        raise ValueError(f"{section_id}: {missing} barcodes absent from positions file")

    keys = pd.Index([f"{section_id}_{b}" for b in barcodes])
    layer = layers.reindex(keys).to_numpy()

    side = int(round(spot_diameter * context))
    image = load_image(raw / "full_image.tif")
    patches = crop_patches(image, positions["px_row"].to_numpy(),
                           positions["px_col"].to_numpy(), side, out_size)
    del image

    meta = pd.DataFrame({
        "barcode": barcodes,
        "section": section_id,
        "donor": donor_of(section_id),
        "split": split_of(section_id),
        "array_row": positions["array_row"].to_numpy(),
        "array_col": positions["array_col"].to_numpy(),
        "px_row": positions["px_row"].to_numpy(),
        "px_col": positions["px_col"].to_numpy(),
        "layer": layer,
        "total_counts": np.asarray(counts.sum(axis=1)).ravel(),
        "n_genes": np.diff(counts.indptr),
    })

    np.save(out / "patches.npy", patches)
    sp.save_npz(out / "counts.npz", counts.astype(np.float32))
    meta.to_csv(out / "meta.csv", index=False)

    labelled = meta["layer"].notna().sum()
    print(f"  {section_id}: {len(meta):5d} spots, {labelled:5d} labelled, "
          f"crop {side}px -> {out_size}px, median UMI {meta.total_counts.median():.0f}")
    return meta, gene_names, gene_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--sections", nargs="*", default=sorted(SECTIONS))
    parser.add_argument("--context", type=float, default=2.0,
                        help="crop side as a multiple of the spot diameter")
    parser.add_argument("--patch-size", type=int, default=112)
    args = parser.parse_args()

    raw_dir, out_dir = Path(args.raw), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = read_layers(raw_dir / "barcode_level_layer_map.tsv")

    print(f"building {len(args.sections)} sections")
    frames, reference_genes = [], None
    for section_id in args.sections:
        meta, names, ids = build_section(
            section_id, raw_dir, out_dir, layers, args.context, args.patch_size)
        if reference_genes is None:
            reference_genes = (names, ids)
            pd.DataFrame({"name": names, "ensembl_id": ids}).to_csv(
                out_dir / "genes.csv", index=False)
        elif not np.array_equal(reference_genes[1], ids):
            raise ValueError(f"{section_id}: gene list differs from the first section")
        frames.append(meta)

    index = pd.concat(frames, ignore_index=True)
    index.to_csv(out_dir / "index.csv", index=False)

    print(f"\n{len(index)} spots total, {index.layer.notna().sum()} labelled")
    print(index.groupby("split").size().to_string())
    print(index.layer.value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()

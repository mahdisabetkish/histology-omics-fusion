"""Collect every result into the static files the dashboard reads.

The published page is plain HTML and JavaScript with no server behind it, so
everything it needs has to be precomputed here: the metric tables, the training
curves, the per-spot predictions for the spatial maps, a sprite sheet of H&E
crops for the hover preview, and a 2-D projection of the learned embedding.

Numbers are rounded on the way out. Full precision would roughly double the
payload for digits nobody reads off a chart.

    python -m src.export_dashboard
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .data.samples import (LAYER_COLORS, LAYERS, SECTIONS, donor_of,
                           split_sections, split_of)
from .utils.common import read_json, write_json

Image.MAX_IMAGE_PIXELS = None

# Genes with well-described laminar patterns in human cortex. Used for the gene
# browser and for the measured-versus-predicted comparison.
MARKER_GENES = [
    "MBP", "PLP1", "MOBP", "GFAP", "AQP4", "SNAP25", "CALM1",
    "CCK", "ENC1", "NEFL", "NEFM", "SYT1", "CARTPT", "PCP4",
    "HPCAL1", "KRT17", "MOG", "CNP", "TF", "SLC17A7", "CUX2",
    "RORB", "FABP7", "NDRG2", "SPARCL1", "PVALB", "CALB1",
]

MODEL_LABELS = {
    "expression_only": ("Expression only", "unimodal"),
    "image_only": ("Histology only", "unimodal"),
    "image_only_scratch": ("Histology only, no ImageNet init", "unimodal"),
    "fusion_concat": ("Fusion, concatenation", "multimodal"),
    "fusion_gated": ("Fusion, gated", "multimodal"),
    "fusion_bilinear": ("Fusion, bilinear", "multimodal"),
    "fusion_gated_ssl": ("Fusion, gated, SSL init", "self-supervised"),
    "expression_only_ssl": ("Expression only, SSL init", "self-supervised"),
    "image_only_ssl": ("Histology only, SSL init", "self-supervised"),
    "probe_ssl": ("Frozen SSL encoders + linear head", "self-supervised"),
    "probe_ssl_no_imagenet": ("Frozen SSL, no ImageNet init", "ablation"),
    "probe_ssl_no_neighbour_mask": ("Frozen SSL, no neighbour masking", "ablation"),
}

METRIC_KEYS = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "kappa",
               "ari", "nmi", "adjacent_accuracy", "spatial_coherence", "ece",
               "macro_auroc"]


def rounded(value, digits=4):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return round(float(value), digits)


def collect_models(runs_dir):
    rows = []
    for tag, (label, family) in MODEL_LABELS.items():
        summary_path = runs_dir / tag / "summary.json"
        metrics_path = runs_dir / tag / "eval_test" / "metrics.json"
        if not summary_path.exists():
            continue
        summary = read_json(summary_path)
        metrics = read_json(metrics_path) if metrics_path.exists() else summary["test"]

        row = {
            "tag": tag,
            "label": label,
            "family": family,
            "modality": summary["args"]["modality"],
            "fusion": summary["args"]["fusion"] if summary["args"]["modality"] == "both" else None,
            "ssl_init": bool(summary["args"].get("init")),
            "frozen": bool(summary["args"].get("freeze_encoders")),
            "parameters": summary.get("parameters"),
            "minutes": rounded(summary.get("minutes"), 2),
            "best_epoch": summary.get("best_epoch"),
            "n_train": summary.get("n_train"),
            "val_macro_f1": rounded(summary.get("val", {}).get("macro_f1")),
        }
        for key in METRIC_KEYS:
            row[key] = rounded(metrics.get(key))
        row["per_class_f1"] = {
            name: rounded(values["f1"], 3)
            for name, values in metrics["per_class"].items()
        }
        row["support"] = {name: values["support"]
                          for name, values in metrics["per_class"].items()}
        row["confusion_matrix"] = metrics["confusion_matrix"]
        row["calibration"] = metrics.get("calibration")
        row["per_section"] = {
            section: {"accuracy": rounded(report["accuracy"], 4),
                      "macro_f1": rounded(report["macro_f1"], 4)}
            for section, report in metrics.get("per_section", {}).items()
        }
        for key in ("gate_mean_expression", "gate_mean_image"):
            if key in metrics:
                row[key] = rounded(metrics[key], 4)
        if "gate_by_layer" in metrics:
            row["gate_by_layer"] = {k: rounded(v, 4)
                                    for k, v in metrics["gate_by_layer"].items()}
        if "spatial_coherence_manual" in metrics:
            row["spatial_coherence_manual"] = rounded(metrics["spatial_coherence_manual"])
        rows.append(row)
    return rows


def collect_curves(runs_dir, tags):
    curves = {}
    for tag in tags:
        path = runs_dir / tag / "history.json"
        if not path.exists():
            continue
        history = read_json(path)
        curves[tag] = {
            key: [rounded(entry.get(key), 5) for entry in history]
            for key in history[0].keys()
        }
    return curves


def collect_ssl(runs_dir):
    out = {}
    for tag in ("ssl", "ssl_no_imagenet", "ssl_no_neighbour_mask"):
        summary_path = runs_dir / tag / "summary.json"
        if not summary_path.exists():
            continue
        summary = read_json(summary_path)
        history = read_json(runs_dir / tag / "history.json")
        probes = read_json(runs_dir / tag / "probes.json")
        out[tag] = {
            "n_spots": summary["n_spots"],
            "n_labelled": summary["n_labelled"],
            "n_unlabelled_used": summary["n_unlabelled_used"],
            "minutes": rounded(summary["minutes"], 2),
            "epochs": summary["args"]["epochs"],
            "batch_size": summary["args"]["batch_size"],
            "neighbour_radius": summary["args"]["neighbour_radius"],
            "history": {
                "epoch": [h["epoch"] for h in history],
                "loss": [rounded(h["loss"], 5) for h in history],
                "batch_accuracy": [rounded(h["batch_accuracy"], 5) for h in history],
                "temperature": [rounded(h["temperature"], 5) for h in history],
            },
            "probes": [{
                "epoch": p["epoch"],
                "recall@1": rounded(p["retrieval"]["recall@1"]),
                "recall@5": rounded(p["retrieval"]["recall@5"]),
                "recall@10": rounded(p["retrieval"]["recall@10"]),
                "recall@50": rounded(p["retrieval"]["recall@50"]),
                "median_rank": p["retrieval"]["median_rank"],
                "mrr": rounded(p["retrieval"]["mean_reciprocal_rank"]),
                "n_gallery": p["retrieval"]["n_gallery"],
                "knn_expression_f1": rounded((p.get("knn_expression") or {}).get("macro_f1")),
                "knn_image_f1": rounded((p.get("knn_image") or {}).get("macro_f1")),
                "knn_expression_acc": rounded((p.get("knn_expression") or {}).get("accuracy")),
                "knn_image_acc": rounded((p.get("knn_image") or {}).get("accuracy")),
            } for p in probes],
        }
    return out


def collect_efficiency(runs_dir):
    points = []
    for path in sorted(runs_dir.glob("eff_*/summary.json")):
        summary = read_json(path)
        tag = path.parent.name
        _, arm, percent = tag.split("_")
        points.append({
            "tag": tag,
            "arm": "ssl" if arm == "ssl" else "scratch",
            "fraction": rounded(summary["args"]["label_fraction"], 4),
            "percent": int(percent),
            "n_train": summary["n_train"],
            "macro_f1": rounded(summary["test"]["macro_f1"]),
            "accuracy": rounded(summary["test"]["accuracy"]),
            "balanced_accuracy": rounded(summary["test"]["balanced_accuracy"]),
            "minutes": rounded(summary["minutes"], 2),
        })
    return sorted(points, key=lambda p: (p["arm"], p["fraction"]))


def collect_regression(runs_dir):
    out = {}
    for tag in ("expr_from_image", "expr_from_image_ssl"):
        path = runs_dir / tag / "summary.json"
        if not path.exists():
            continue
        summary = read_json(path)
        scores = np.load(runs_dir / tag / "gene_scores.npz", allow_pickle=False)
        r = scores["r"]
        genes = scores["genes"].astype(str)

        # Histogram rather than 3000 raw values.
        counts, edges = np.histogram(r, bins=60, range=(-0.4, 1.0))
        out[tag] = {
            "mean_r": rounded(summary["test_mean_r"]),
            "median_r": rounded(summary["test_median_r"]),
            "mse": rounded(summary["test_mse"]),
            "above_03": rounded(summary["fraction_r_above_0.3"]),
            "above_05": rounded(summary["fraction_r_above_0.5"]),
            "minutes": rounded(summary["minutes"], 2),
            "top_genes": [{"gene": g["gene"], "r": rounded(g["r"], 3)}
                          for g in summary["top_genes"]],
            "bottom_genes": [{"gene": g["gene"], "r": rounded(g["r"], 3)}
                             for g in summary["bottom_genes"]],
            "histogram": {"counts": counts.tolist(),
                          "edges": [rounded(e, 3) for e in edges]},
            "marker_scores": {
                gene: rounded(float(r[genes == gene][0]), 3)
                for gene in MARKER_GENES if (genes == gene).any()
            },
        }
        scatter_path = runs_dir / tag / "test_predictions.npz"
        if tag == "expr_from_image" and scatter_path.exists():
            pack = np.load(scatter_path, allow_pickle=False)
            names = pack["genes"].astype(str)
            keep = list(range(min(6, len(names))))
            out[tag]["scatter"] = [{
                "gene": str(names[i]),
                "r": rounded(float(r[genes == names[i]][0]), 3),
                "measured": [rounded(v, 3) for v in pack["measured"][:1500, i]],
                "predicted": [rounded(v, 3) for v in pack["predicted"][:1500, i]],
            } for i in keep]
    return out


def build_section_payloads(processed, runs_dir, out_dir, model_tag, expression_tag):
    """Per-spot arrays for the spatial maps, one file per section."""
    index = pd.read_csv(processed / "index.csv", dtype={"section": str})
    panel = np.load(processed / "expression.npy", mmap_mode="r")
    store = np.load(processed / "expression_meta.npz", allow_pickle=False)
    hvg_names = store["hvg_names"].astype(str)
    gene_lookup = {name: i for i, name in enumerate(hvg_names)}

    predictions = {}
    for split in ("train", "val", "test"):
        path = runs_dir / model_tag / f"eval_{split}" / "predictions.csv"
        if path.exists():
            frame = pd.read_csv(path, dtype={"section": str})
            for _, part in frame.groupby("section"):
                predictions[(part.name if hasattr(part, "name") else None)] = part
            for section, part in frame.groupby("section"):
                predictions[section] = part

    predicted_expression = {}
    scatter_path = runs_dir / expression_tag / "test_predictions.npz"
    if scatter_path.exists():
        pack = np.load(scatter_path, allow_pickle=False)
        predicted_expression = {"genes": pack["genes"].astype(str),
                                "values": pack["predicted"]}

    available = sorted(MARKER_GENES, key=MARKER_GENES.index)
    available = [g for g in available if g in gene_lookup]

    section_dir = out_dir / "data" / "sections"
    section_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    for section in sorted(SECTIONS):
        take = (index["section"].astype(str) == section).to_numpy()
        if not take.any():
            continue
        meta = index[take].reset_index(drop=True)
        rows_global = np.flatnonzero(take)

        payload = {
            "section": section,
            "donor": donor_of(section),
            "position_um": SECTIONS[section]["position"],
            "split": split_of(section),
            "n_spots": int(take.sum()),
            "array_row": meta["array_row"].tolist(),
            "array_col": meta["array_col"].tolist(),
            "px_row": [int(v) for v in meta["px_row"]],
            "px_col": [int(v) for v in meta["px_col"]],
            "total_counts": [int(v) for v in meta["total_counts"]],
            "n_genes": [int(v) for v in meta["n_genes"]],
            "true": [LAYERS.index(v) if isinstance(v, str) and v in LAYERS else -1
                     for v in meta["layer"]],
        }

        frame = predictions.get(section)
        if frame is not None:
            keyed = frame.set_index("barcode")
            aligned = keyed.reindex(meta["barcode"])
            payload["pred"] = [int(v) if np.isfinite(v) else -1
                               for v in aligned["pred"].to_numpy(dtype=float)]
            payload["confidence"] = [rounded(v, 3) if np.isfinite(v) else None
                                     for v in aligned["confidence"].to_numpy(dtype=float)]
            payload["gate_expression"] = [
                rounded(v, 3) if np.isfinite(v) else None
                for v in aligned["gate_expression"].to_numpy(dtype=float)]
            payload["probabilities"] = [
                [rounded(v, 3) for v in aligned[f"p_{name}"].to_numpy(dtype=float)]
                for name in LAYERS]

        payload["genes"] = {
            gene: [rounded(v, 2) for v in panel[rows_global, gene_lookup[gene]]]
            for gene in available
        }
        if predicted_expression and split_of(section) == "test":
            names = list(predicted_expression["genes"])
            test_rows = index[(index["split"] == "test")].reset_index(drop=True)
            offset = int(np.flatnonzero(
                (test_rows["section"].astype(str) == section).to_numpy())[0])
            count = int(take.sum())
            payload["predicted_genes"] = {
                gene: [rounded(v, 2) for v in
                       predicted_expression["values"][offset:offset + count,
                                                      names.index(gene)]]
                for gene in available
                if gene in names and offset + count <= len(predicted_expression["values"])
            }

        write_json(section_dir / f"{section}.json", payload)
        summaries.append({
            "section": section,
            "donor": donor_of(section),
            "split": split_of(section),
            "position_um": SECTIONS[section]["position"],
            "n_spots": payload["n_spots"],
            "n_labelled": int(sum(v >= 0 for v in payload["true"])),
            "median_umi": int(np.median(meta["total_counts"])),
            "median_genes": int(np.median(meta["n_genes"])),
            "layer_counts": {name: int((meta["layer"] == name).sum()) for name in LAYERS},
        })
    return summaries, available


def build_sprites(processed, out_dir, tile=40, quality=80):
    """One JPEG per section holding every crop, laid out on a grid.

    The dashboard shows the tissue under a spot when the cursor is over it.
    Fetching four thousand small files would be hopeless over HTTP, so they go
    into a single image and the page offsets into it with CSS.
    """
    sprite_dir = out_dir / "assets" / "patches"
    sprite_dir.mkdir(parents=True, exist_ok=True)
    layout = {}

    for section in sorted(SECTIONS):
        source = processed / section / "patches.npy"
        if not source.exists():
            continue
        patches = np.load(source, mmap_mode="r")
        n = len(patches)
        columns = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / columns))

        sheet = Image.new("RGB", (columns * tile, rows * tile), "white")
        for i in range(n):
            crop = Image.fromarray(np.asarray(patches[i])).resize(
                (tile, tile), Image.BILINEAR)
            sheet.paste(crop, ((i % columns) * tile, (i // columns) * tile))
        sheet.save(sprite_dir / f"{section}.jpg", quality=quality, optimize=True)
        layout[section] = {"tile": tile, "columns": columns, "rows": rows, "n": n}
        print(f"  sprite {section}: {columns}x{rows} tiles, "
              f"{(sprite_dir / f'{section}.jpg').stat().st_size / 1e6:.2f} MB")
    return layout


def build_example_patch(processed, out_dir, section, index, size=224):
    """One crop at full stored resolution for the 'what a spot looks like' panel.

    The sprite tiles are deliberately small and look mushy when blown up, and
    this panel is the one place a single crop is shown large.
    """
    source = processed / section / "patches.npy"
    if not source.exists():
        return None
    patches = np.load(source, mmap_mode="r")
    index = int(np.clip(index, 0, len(patches) - 1))
    image = Image.fromarray(np.asarray(patches[index])).resize(
        (size, size), Image.LANCZOS)
    target = out_dir / "assets" / "example_spot.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, quality=92, optimize=True)
    return {"section": section, "index": index, "size": size}


def build_tissue_images(raw, out_dir, max_side=900, quality=82):
    """Downscaled H&E backdrops for the spatial maps."""
    tissue_dir = out_dir / "assets" / "tissue"
    tissue_dir.mkdir(parents=True, exist_ok=True)
    info = {}
    for section in sorted(SECTIONS):
        source = raw / section / "tissue_lowres_image.png"
        if not source.exists():
            continue
        image = Image.open(source).convert("RGB")
        scale = max_side / max(image.size)
        if scale < 1:
            image = image.resize((int(image.width * scale), int(image.height * scale)),
                                 Image.LANCZOS)
        image.save(tissue_dir / f"{section}.jpg", quality=quality, optimize=True)
        scalefactors = json.loads((raw / section / "scalefactors_json.json").read_text())
        info[section] = {
            "width": image.width,
            "height": image.height,
            # Full-resolution pixel coordinates times this land on the JPEG.
            "scale": scalefactors["tissue_lowres_scalef"] * (scale if scale < 1 else 1.0),
        }
    return info


def build_embedding(runs_dir, model_tag, out_dir, seed=0, max_points=12000):
    """Two-dimensional view of the fused embedding on the held-out donor."""
    path = runs_dir / model_tag / "eval_test" / "embeddings.npz"
    if not path.exists():
        return None
    pack = np.load(path, allow_pickle=False)
    embedding, labels = pack["embedding"], pack["label"]

    rng = np.random.default_rng(seed)
    if len(embedding) > max_points:
        pick = rng.choice(len(embedding), max_points, replace=False)
        embedding, labels = embedding[pick], labels[pick]

    import umap
    reducer = umap.UMAP(n_neighbors=25, min_dist=0.15, metric="cosine",
                        random_state=seed)
    coordinates = reducer.fit_transform(embedding)
    coordinates -= coordinates.mean(axis=0)
    coordinates /= np.abs(coordinates).max()

    return {
        "x": [rounded(v, 3) for v in coordinates[:, 0]],
        "y": [rounded(v, 3) for v in coordinates[:, 1]],
        "label": labels.tolist(),
        "n": int(len(labels)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--out", default="docs")
    parser.add_argument("--model-tag", default="fusion_gated_ssl")
    parser.add_argument("--expression-tag", default="expr_from_image")
    parser.add_argument("--skip-sprites", action="store_true")
    parser.add_argument("--skip-embedding", action="store_true")
    parser.add_argument("--tile", type=int, default=40)
    args = parser.parse_args()

    processed = Path(args.processed)
    runs_dir = Path(args.runs)
    out_dir = Path(args.out)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)

    print("models")
    models = collect_models(runs_dir)
    print(f"  {len(models)} runs with test metrics")

    print("curves")
    curves = collect_curves(runs_dir, [m["tag"] for m in models])

    print("self-supervised stage")
    ssl_block = collect_ssl(runs_dir)

    print("label efficiency")
    efficiency = collect_efficiency(runs_dir)

    print("cross-modal regression")
    regression = collect_regression(runs_dir)

    print("sections")
    sections, marker_genes = build_section_payloads(
        processed, runs_dir, out_dir, args.model_tag, args.expression_tag)

    print("tissue backdrops")
    tissue = build_tissue_images(Path(args.raw), out_dir)

    sprites = {}
    if not args.skip_sprites:
        print("patch sprites")
        sprites = build_sprites(processed, out_dir, tile=args.tile)

    # A single annotated spot, used for the panel that shows what one sample is.
    example = None
    index_frame = pd.read_csv(processed / "index.csv", dtype={"section": str})
    for candidate in [s["section"] for s in sections if s["split"] == "test"] + \
            [s["section"] for s in sections]:
        rows = index_frame[index_frame["section"].astype(str) == candidate]
        annotated = np.flatnonzero(rows["layer"].notna().to_numpy())
        if len(annotated):
            example = build_example_patch(processed, out_dir, candidate,
                                          annotated[len(annotated) // 2])
            break

    embedding = None
    if not args.skip_embedding:
        print("embedding projection")
        embedding = build_embedding(runs_dir, args.model_tag, out_dir)

    feature_config = read_json(processed / "feature_config.json")
    splits = split_sections()

    manifest = {
        "dataset": {
            "name": "spatialLIBD human DLPFC",
            "platform": "10x Genomics Visium",
            "n_sections": len(sections),
            "n_donors": 3,
            "n_spots": sum(s["n_spots"] for s in sections),
            "n_labelled": sum(s["n_labelled"] for s in sections),
            "n_genes_measured": 33538,
            "n_genes_modelled": feature_config["n_hvg"],
            "patch_size": 112,
            "splits": splits,
            "sections": sections,
        },
        "layers": LAYERS,
        "layer_colors": LAYER_COLORS,
        "marker_genes": marker_genes,
        "models": models,
        "curves": curves,
        "ssl": ssl_block,
        "efficiency": efficiency,
        "regression": regression,
        "tissue": tissue,
        "sprites": sprites,
        "example_spot": example,
        "embedding": embedding,
        "headline_model": args.model_tag,
        "feature_config": feature_config,
    }
    write_json(out_dir / "data" / "manifest.json", manifest)

    size = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
    print(f"\nwrote {out_dir}/  ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

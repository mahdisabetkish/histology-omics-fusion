"""Score a trained checkpoint on the held-out donor and dump per-spot output.

Beyond the summary metrics this writes one row per test spot: the predicted
layer, the full probability vector, the fusion gate weights and the joint
embedding. The spatial maps and the embedding plots in the dashboard are all
built from that file, and having it on disk means the figures can be redrawn
without touching the GPU again.

    python -m src.evaluate --run runs/fusion_ssl --split test
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data.dataset import SpotDataset
from .data.samples import LAYERS
from .models.multimodal import SpotClassifier
from .utils.common import get_device, read_json, write_json
from .utils.metrics import (calibration_curve, classification_report,
                            spatial_coherence)


def load_model(run_dir, n_genes, device):
    state = torch.load(Path(run_dir) / "checkpoint.pt", map_location="cpu",
                       weights_only=False)
    saved = state["args"]
    model = SpotClassifier(
        n_genes=n_genes, n_classes=len(LAYERS), modality=saved["modality"],
        fusion=saved["fusion"], embed_dim=saved["embed_dim"],
        hidden_dim=saved["hidden_dim"], dropout=saved["dropout"],
        pretrained_image=not saved["no_imagenet"],
    )
    model.load_state_dict(state["model"])
    return model.to(device).eval(), saved


@torch.no_grad()
def predict(model, loader, device):
    logits, embeddings, gates, labels, rows = [], [], [], [], []
    for batch in loader:
        expression = batch["expression"].to(device) \
            if model.expression_encoder is not None else None
        image = batch["image"].to(device) if model.image_encoder is not None else None

        batch_logits, joint, _, _ = model(expression=expression, image=image,
                                          return_embedding=True)
        logits.append(batch_logits.float().cpu().numpy())
        embeddings.append(joint.float().cpu().numpy())
        labels.append(batch["label"].numpy())
        rows.append(batch["row"].numpy())

        gate = model.modality_gates(expression, image) if model.modality == "both" else None
        gates.append(gate.float().cpu().numpy() if gate is not None
                     else np.full((len(batch["label"]), 2), np.nan, np.float32))

    return (np.concatenate(logits), np.concatenate(embeddings),
            np.concatenate(gates), np.concatenate(labels), np.concatenate(rows))


def softmax(x):
    shifted = x - x.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def evaluate_run(run_dir, processed="data/processed", split="test",
                 batch_size=256, workers=6, device=None):
    run_dir = Path(run_dir)
    device = device or get_device()

    dataset = SpotDataset(processed, split, labelled_only=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)
    model, saved = load_model(run_dir, dataset.n_genes, device)

    logits, embeddings, gates, labels, rows = predict(model, loader, device)
    probabilities = softmax(logits)
    predictions = probabilities.argmax(axis=1)

    report = classification_report(labels, predictions, probabilities=probabilities,
                                   n_classes=len(LAYERS), class_names=LAYERS)
    meta = dataset.meta

    report["spatial_coherence"] = spatial_coherence(
        predictions, meta["array_row"], meta["array_col"], meta["section"])
    # The same statistic on the manual calls is the ceiling: annotators drew
    # smooth bands, but the lattice still puts some spots on a boundary.
    report["spatial_coherence_manual"] = spatial_coherence(
        labels, meta["array_row"], meta["array_col"], meta["section"])
    report["calibration"] = calibration_curve(labels, probabilities)

    per_section = {}
    for section in sorted(meta["section"].astype(str).unique()):
        take = (meta["section"].astype(str) == section).to_numpy()
        per_section[section] = classification_report(
            labels[take], predictions[take], probabilities=probabilities[take],
            n_classes=len(LAYERS), class_names=LAYERS)
    report["per_section"] = per_section

    if model.modality == "both" and not np.isnan(gates).all():
        report["gate_mean_expression"] = float(np.nanmean(gates[:, 0]))
        report["gate_mean_image"] = float(np.nanmean(gates[:, 1]))
        report["gate_by_layer"] = {
            LAYERS[c]: float(np.nanmean(gates[labels == c, 0]))
            for c in range(len(LAYERS)) if (labels == c).any()
        }

    predictions_frame = pd.DataFrame({
        "section": meta["section"].astype(str).to_numpy(),
        "barcode": meta["barcode"].to_numpy(),
        "array_row": meta["array_row"].to_numpy(),
        "array_col": meta["array_col"].to_numpy(),
        "px_row": meta["px_row"].to_numpy(),
        "px_col": meta["px_col"].to_numpy(),
        "total_counts": meta["total_counts"].to_numpy(),
        "true": labels,
        "pred": predictions,
        "confidence": probabilities.max(axis=1),
        "gate_expression": gates[:, 0],
        "gate_image": gates[:, 1],
    })
    for i, name in enumerate(LAYERS):
        predictions_frame[f"p_{name}"] = probabilities[:, i]

    out = run_dir / f"eval_{split}"
    out.mkdir(parents=True, exist_ok=True)
    predictions_frame.to_csv(out / "predictions.csv", index=False)
    np.savez_compressed(out / "embeddings.npz", embedding=embeddings.astype(np.float32),
                        label=labels, row=rows)
    write_json(out / "metrics.json", report)
    return report, predictions_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    report, _ = evaluate_run(args.run, args.processed, args.split,
                             args.batch_size, args.workers)
    print(f"[{Path(args.run).name}] {args.split}")
    print(f"  accuracy           {report['accuracy']:.4f}")
    print(f"  balanced accuracy  {report['balanced_accuracy']:.4f}")
    print(f"  macro F1           {report['macro_f1']:.4f}")
    print(f"  kappa              {report['kappa']:.4f}")
    print(f"  ARI                {report['ari']:.4f}")
    print(f"  adjacent-layer acc {report['adjacent_accuracy']:.4f}")
    print(f"  spatial coherence  {report['spatial_coherence']:.4f} "
          f"(manual {report['spatial_coherence_manual']:.4f})")
    print("  per-class F1: " + "  ".join(
        f"{k} {v['f1']:.3f}" for k, v in report["per_class"].items()))


if __name__ == "__main__":
    main()

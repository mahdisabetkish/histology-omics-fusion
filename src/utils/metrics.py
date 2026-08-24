"""Evaluation metrics.

Layer frequencies are uneven in this tissue: layer 3 covers a wide band and
takes about a quarter of the annotated spots, while layer 4 is a thin stripe
worth a few percent. Plain accuracy rewards a model for getting the wide layers
right and hides what it does on the thin ones, so macro-F1 and balanced
accuracy are the numbers to read first.
"""

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def classification_report(y_true, y_pred, probabilities=None, n_classes=7,
                          class_names=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(n_classes))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)

    report = {
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_class": {},
        "n": int(len(y_true)),
    }

    names = class_names or [str(i) for i in labels]
    for i, name in enumerate(names):
        report["per_class"][name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    if probabilities is not None:
        probabilities = np.asarray(probabilities)
        present = np.unique(y_true)
        if len(present) > 1:
            try:
                report["macro_auroc"] = float(roc_auc_score(
                    y_true, probabilities, multi_class="ovr", average="macro",
                    labels=labels))
            except ValueError:
                report["macro_auroc"] = None
        report["ece"] = expected_calibration_error(y_true, probabilities)
        report["mean_confidence"] = float(probabilities.max(axis=1).mean())

    # Layers are ordered through the cortex, so confusing L2 with L3 is a
    # smaller error than confusing L2 with white matter. Counting how often the
    # prediction lands on an adjacent layer captures that.
    report["adjacent_accuracy"] = float((np.abs(y_true - y_pred) <= 1).mean())
    return report


def expected_calibration_error(y_true, probabilities, n_bins=15):
    """Gap between confidence and accuracy, averaged over confidence bins."""
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == np.asarray(y_true)).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > low) & (confidence <= high)
        if not in_bin.any():
            continue
        error += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(error)


def calibration_curve(y_true, probabilities, n_bins=15):
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == np.asarray(y_true)).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > low) & (confidence <= high)
        bins.append({
            "lower": float(low),
            "upper": float(high),
            "count": int(in_bin.sum()),
            "confidence": float(confidence[in_bin].mean()) if in_bin.any() else None,
            "accuracy": float(correct[in_bin].mean()) if in_bin.any() else None,
        })
    return bins


def spatial_coherence(predictions, array_row, array_col, sections, radius=1):
    """Share of spots whose prediction matches the majority call of their neighbours.

    Cortical layers are contiguous bands. A model can reach a decent accuracy
    while producing speckled, anatomically implausible maps, and this number
    separates that case from one where the predicted regions hold together.
    The same quantity computed on the manual annotation gives the ceiling.
    """
    predictions = np.asarray(predictions)
    array_row = np.asarray(array_row)
    array_col = np.asarray(array_col)
    sections = np.asarray(sections)

    agree = []
    for section in np.unique(sections):
        take = sections == section
        rows, cols, preds = array_row[take], array_col[take], predictions[take]
        lookup = {(r, c): p for r, c, p in zip(rows, cols, preds)}
        # Six neighbours on the Visium lattice.
        steps = [(0, -2), (0, 2), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        if radius > 1:
            steps += [(0, -4), (0, 4), (-2, -2), (-2, 0), (-2, 2),
                      (2, -2), (2, 0), (2, 2)]
        for (r, c), p in lookup.items():
            neighbours = [lookup[(r + dr, c + dc)] for dr, dc in steps
                          if (r + dr, c + dc) in lookup]
            if not neighbours:
                continue
            counts = np.bincount(neighbours, minlength=int(predictions.max()) + 1)
            agree.append(float(p == counts.argmax()))
    return float(np.mean(agree)) if agree else float("nan")


def retrieval_metrics(query, gallery, ks=(1, 5, 10, 50), exclude=None):
    """Cross-modal retrieval scores for L2-normalised embeddings.

    Row i of `query` and row i of `gallery` are the two modalities of the same
    spot. `exclude` masks pairs that should not count as errors, which is used
    to ignore a spot's immediate spatial neighbours.
    """
    query = query / np.linalg.norm(query, axis=1, keepdims=True)
    gallery = gallery / np.linalg.norm(gallery, axis=1, keepdims=True)
    similarity = query @ gallery.T

    n = similarity.shape[0]
    correct = similarity[np.arange(n), np.arange(n)].copy()
    if exclude is not None:
        similarity = np.where(exclude, -np.inf, similarity)
        similarity[np.arange(n), np.arange(n)] = correct

    ranks = (similarity > correct[:, None]).sum(axis=1) + 1
    out = {f"recall@{k}": float((ranks <= k).mean()) for k in ks}
    out["median_rank"] = float(np.median(ranks))
    out["mean_reciprocal_rank"] = float((1.0 / ranks).mean())
    out["n_gallery"] = int(n)
    return out


def gene_correlations(y_true, y_pred):
    """Per-gene Pearson correlation between measured and predicted expression."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    true_centred = y_true - y_true.mean(axis=0, keepdims=True)
    pred_centred = y_pred - y_pred.mean(axis=0, keepdims=True)
    denominator = (np.linalg.norm(true_centred, axis=0)
                   * np.linalg.norm(pred_centred, axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (true_centred * pred_centred).sum(axis=0) / denominator
    return np.nan_to_num(r, nan=0.0)

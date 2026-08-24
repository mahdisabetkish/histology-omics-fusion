"""Print the results tables as markdown, straight from the run artefacts.

Keeps the numbers in the README tied to what is actually on disk instead of
being retyped by hand.

    python scripts/make_report.py
    python scripts/make_report.py --sort macro_f1
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.samples import LAYERS  # noqa: E402
from src.export_dashboard import MODEL_LABELS  # noqa: E402

RUNS = Path("runs")


def load(tag):
    summary_path = RUNS / tag / "summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    metrics_path = RUNS / tag / "eval_test" / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() \
        else summary["test"]
    return summary, metrics


def fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sort", default="macro_f1")
    args = parser.parse_args()

    rows = []
    for tag, (label, family) in MODEL_LABELS.items():
        loaded = load(tag)
        if loaded:
            rows.append((tag, label, family, *loaded))
    if not rows:
        raise SystemExit("no completed runs found under runs/")
    rows.sort(key=lambda r: r[4].get(args.sort) or -1, reverse=True)

    best = {}
    for key in ("macro_f1", "balanced_accuracy", "accuracy", "kappa", "ari",
                "spatial_coherence"):
        values = [r[4].get(key) for r in rows if r[4].get(key) is not None]
        if values:
            best[key] = max(values)

    print("### Held-out donor (Br8100)\n")
    print("| Model | Macro F1 | Balanced acc | Accuracy | κ | ARI | ±1 layer | Coherence |")
    print("|---|---|---|---|---|---|---|---|")
    for tag, label, family, summary, metrics in rows:
        cells = []
        for key in ("macro_f1", "balanced_accuracy", "accuracy", "kappa", "ari",
                    "adjacent_accuracy", "spatial_coherence"):
            value = metrics.get(key)
            text = fmt(value)
            if key in best and value is not None and abs(value - best[key]) < 1e-9:
                text = f"**{text}**"
            cells.append(text)
        name = f"{label}" + (" ⁺" if summary["args"].get("init") else "")
        print(f"| {name} | " + " | ".join(cells) + " |")
    print("\n⁺ initialised from the self-supervised checkpoint.")

    print("\n### F1 by layer\n")
    print("| Model | " + " | ".join(LAYERS) + " |")
    print("|---" * (len(LAYERS) + 1) + "|")
    for tag, label, family, summary, metrics in rows:
        per_class = metrics["per_class"]
        print(f"| {label} | " + " | ".join(
            fmt(per_class[layer]["f1"], 2) for layer in LAYERS) + " |")

    manual = next((m.get("spatial_coherence_manual") for _, _, _, _, m in rows
                   if m.get("spatial_coherence_manual")), None)
    if manual:
        print(f"\nSpatial coherence of the manual annotation itself: {manual:.3f}.")

    efficiency = sorted(RUNS.glob("eff_*/summary.json"))
    if efficiency:
        print("\n### Label efficiency\n")
        points = {}
        for path in efficiency:
            summary = json.loads(path.read_text())
            _, arm, percent = path.parent.name.split("_")
            points.setdefault(int(percent), {})[arm] = summary
        print("| Labels | Spots | From scratch | SSL init | Δ |")
        print("|---|---|---|---|---|")
        for percent in sorted(points):
            pair = points[percent]
            scratch = pair.get("scratch")
            ssl = pair.get("ssl")
            delta = (ssl["test"]["macro_f1"] - scratch["test"]["macro_f1"]) \
                if scratch and ssl else None
            print(f"| {percent}% | {scratch['n_train'] if scratch else '—'} | "
                  f"{fmt(scratch['test']['macro_f1']) if scratch else '—'} | "
                  f"{fmt(ssl['test']['macro_f1']) if ssl else '—'} | "
                  f"{('+' if delta and delta > 0 else '') + fmt(delta) if delta is not None else '—'} |")

    total = sum(r[3].get("minutes", 0) for r in rows)
    for tag in ("ssl", "ssl_no_imagenet", "ssl_no_neighbour_mask",
                "expr_from_image", "expr_from_image_ssl"):
        path = RUNS / tag / "summary.json"
        if path.exists():
            total += json.loads(path.read_text()).get("minutes", 0)
    for path in efficiency:
        total += json.loads(path.read_text()).get("minutes", 0)
    print(f"\nTotal training time across every run: {total / 60:.1f} hours.")


if __name__ == "__main__":
    main()

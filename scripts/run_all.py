"""Run the whole study end to end.

Stages are independent and can be run separately. Each one skips work that has
already produced a summary.json unless --force is given, so an interrupted run
picks up where it stopped.

    python scripts/run_all.py                       # everything
    python scripts/run_all.py --stages ssl supervised
    python scripts/run_all.py --stages efficiency --force
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"

SSL_CHECKPOINT = "runs/ssl/checkpoint.pt"

# Fractions of the annotated training spots used for the label-efficiency
# curve. The point of the curve is the small end: if self-supervised
# pretraining is doing anything useful, it shows up where labels are scarce and
# largely washes out once the full training set is available.
FRACTIONS = [0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.00]


def stage_ssl(common):
    return [
        ("ssl", ["-m", "src.pretrain_ssl", "--tag", "ssl",
                 "--epochs", "60", "--batch-size", "256"] + common),
        # Can the contrastive objective build a usable image encoder on its own,
        # without ImageNet weights underneath it?
        ("ssl_no_imagenet", ["-m", "src.pretrain_ssl", "--tag", "ssl_no_imagenet",
                             "--epochs", "60", "--batch-size", "256",
                             "--no-imagenet"] + common),
        ("ssl_no_neighbour_mask", ["-m", "src.pretrain_ssl",
                                   "--tag", "ssl_no_neighbour_mask",
                                   "--epochs", "60", "--batch-size", "256",
                                   "--neighbour-radius", "0"] + common),
    ]


def stage_supervised(common, epochs):
    base = ["-m", "src.train", "--epochs", str(epochs)]
    jobs = [
        ("expression_only", base + ["--tag", "expression_only",
                                    "--modality", "expression"]),
        ("image_only", base + ["--tag", "image_only", "--modality", "image"]),
        ("image_only_scratch", base + ["--tag", "image_only_scratch",
                                       "--modality", "image", "--no-imagenet"]),
        ("fusion_concat", base + ["--tag", "fusion_concat", "--modality", "both",
                                  "--fusion", "concat"]),
        ("fusion_gated", base + ["--tag", "fusion_gated", "--modality", "both",
                                 "--fusion", "gated"]),
        ("fusion_bilinear", base + ["--tag", "fusion_bilinear", "--modality", "both",
                                    "--fusion", "bilinear"]),
    ]
    return [(tag, cmd + common) for tag, cmd in jobs]


def stage_ssl_transfer(common, epochs):
    base = ["-m", "src.train", "--epochs", str(epochs), "--init", SSL_CHECKPOINT]
    jobs = [
        ("fusion_gated_ssl", base + ["--tag", "fusion_gated_ssl",
                                     "--modality", "both", "--fusion", "gated"]),
        ("expression_only_ssl", base + ["--tag", "expression_only_ssl",
                                        "--modality", "expression"]),
        ("image_only_ssl", base + ["--tag", "image_only_ssl", "--modality", "image"]),
        # Frozen encoders. Measures the representation itself rather than what
        # fine-tuning can recover from it.
        ("probe_ssl", ["-m", "src.train", "--tag", "probe_ssl", "--modality", "both",
                       "--fusion", "gated", "--init", SSL_CHECKPOINT,
                       "--freeze-encoders", "--epochs", "30", "--lr", "1e-3"]),
        ("probe_ssl_no_imagenet",
         ["-m", "src.train", "--tag", "probe_ssl_no_imagenet", "--modality", "both",
          "--fusion", "gated", "--init", "runs/ssl_no_imagenet/checkpoint.pt",
          "--freeze-encoders", "--epochs", "30", "--lr", "1e-3", "--no-imagenet"]),
        ("probe_ssl_no_neighbour_mask",
         ["-m", "src.train", "--tag", "probe_ssl_no_neighbour_mask",
          "--modality", "both", "--fusion", "gated",
          "--init", "runs/ssl_no_neighbour_mask/checkpoint.pt",
          "--freeze-encoders", "--epochs", "30", "--lr", "1e-3"]),
    ]
    return [(tag, cmd + common) for tag, cmd in jobs]


def stage_efficiency(common, epochs):
    jobs = []
    for fraction in FRACTIONS:
        name = f"{int(fraction * 100):03d}"
        shared = ["-m", "src.train", "--modality", "both", "--fusion", "gated",
                  "--label-fraction", str(fraction), "--epochs", str(epochs)]
        jobs.append((f"eff_scratch_{name}",
                     shared + ["--tag", f"eff_scratch_{name}"]))
        jobs.append((f"eff_ssl_{name}",
                     shared + ["--tag", f"eff_ssl_{name}", "--init", SSL_CHECKPOINT]))
    return [(tag, cmd + common) for tag, cmd in jobs]


def stage_regression(common, epochs):
    return [
        ("expr_from_image", ["-m", "src.train_expression", "--tag", "expr_from_image",
                             "--epochs", str(epochs)] + common),
        ("expr_from_image_ssl", ["-m", "src.train_expression",
                                 "--tag", "expr_from_image_ssl",
                                 "--epochs", str(epochs),
                                 "--init", SSL_CHECKPOINT] + common),
    ]


def stage_evaluate(_common, _epochs):
    tags = ["expression_only", "image_only", "image_only_scratch", "fusion_concat",
            "fusion_gated", "fusion_bilinear", "fusion_gated_ssl",
            "expression_only_ssl", "image_only_ssl", "probe_ssl",
            "probe_ssl_no_imagenet", "probe_ssl_no_neighbour_mask"]
    jobs = []
    for tag in tags:
        jobs.append((f"eval_{tag}",
                     ["-m", "src.evaluate", "--run", f"runs/{tag}", "--split", "test"]))
    return jobs


STAGES = {
    "ssl": stage_ssl,
    "supervised": stage_supervised,
    "transfer": stage_ssl_transfer,
    "efficiency": stage_efficiency,
    "regression": stage_regression,
    "evaluate": stage_evaluate,
}

ORDER = ["ssl", "supervised", "transfer", "efficiency", "regression", "evaluate"]


def already_done(tag, stage):
    if stage == "evaluate":
        return (RUNS / tag.replace("eval_", "") / "eval_test" / "metrics.json").exists()
    return (RUNS / tag / "summary.json").exists()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", nargs="*", default=ORDER, choices=ORDER)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    common = ["--workers", str(args.workers), "--seed", str(args.seed)]
    started = time.time()
    failures = []

    for stage in [s for s in ORDER if s in args.stages]:
        builder = STAGES[stage]
        jobs = builder(common) if stage == "ssl" else builder(common, args.epochs)
        print(f"\n{'=' * 72}\n{stage.upper()}  ({len(jobs)} jobs)\n{'=' * 72}")

        for tag, command in jobs:
            if not args.force and already_done(tag, stage):
                print(f"-- {tag}: already done, skipping")
                continue
            printable = " ".join(command)
            if args.dry_run:
                print(f"-- {tag}: python {printable}")
                continue

            print(f"\n>> {tag}")
            job_started = time.time()
            result = subprocess.run([sys.executable] + command, cwd=ROOT)
            elapsed = (time.time() - job_started) / 60
            if result.returncode != 0:
                print(f"!! {tag} failed with code {result.returncode}")
                failures.append(tag)
            else:
                print(f"<< {tag} finished in {elapsed:.1f} min")

    total = (time.time() - started) / 60
    print(f"\n{'=' * 72}")
    print(f"total {total:.1f} min" + (f", {len(failures)} failed: "
                                      f"{', '.join(failures)}" if failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

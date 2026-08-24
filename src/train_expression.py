"""Predict the expression panel from histology alone.

This is the cross-modal direction of the problem and it doubles as a check on
how much the two modalities actually overlap. Genes whose spatial pattern
follows the cortical bands should be partly recoverable from the crop, because
laminar position is visible in the tissue. Genes that vary for reasons with no
morphological signature should not be, and those are exactly the ones that
justify keeping the expression arm in the fused classifier.

Reported per gene as the Pearson correlation between measured and predicted
values across the held-out donor's spots, which is the convention used by the
histology-to-expression literature and keeps the numbers comparable.

    python -m src.train_expression --tag expr_from_image --epochs 40
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data.dataset import ImageAugment, SpotDataset
from .models.multimodal import ExpressionFromImage
from .utils.common import (RunningMean, count_parameters, describe_device,
                           get_device, set_seed, write_json)
from .utils.metrics import gene_correlations


@torch.no_grad()
def collect(model, loader, device):
    model.eval()
    predicted, measured = [], []
    for batch in loader:
        output = model(batch["image"].to(device, non_blocking=True))
        predicted.append(output.float().cpu().numpy())
        measured.append(batch["expression"].numpy())
    return np.concatenate(predicted), np.concatenate(measured)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--out", default="runs")
    parser.add_argument("--tag", default="expr_from_image")
    parser.add_argument("--init", default=None,
                        help="self-supervised checkpoint to initialise the image encoder")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--color-jitter", type=float, default=0.25)
    parser.add_argument("--top-genes", type=int, default=50,
                        help="how many best-predicted genes to record individually")
    parser.add_argument("--no-imagenet", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(not args.cpu)
    out_dir = Path(args.out) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Every spot counts here, annotated or not, since no label is involved.
    train = SpotDataset(args.processed, "train", labelled_only=False,
                        image_augment=ImageAugment(jitter=args.color_jitter),
                        seed=args.seed)
    val = SpotDataset(args.processed, "val", labelled_only=False)
    test = SpotDataset(args.processed, "test", labelled_only=False)

    common = dict(batch_size=args.batch_size, num_workers=args.workers,
                  pin_memory=True, persistent_workers=args.workers > 0)
    train_loader = DataLoader(train, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val, shuffle=False, **common)
    test_loader = DataLoader(test, shuffle=False, **common)

    print(f"[{args.tag}] device {describe_device(device)}")
    print(f"  train {len(train)}  val {len(val)}  test {len(test)}  "
          f"targets {train.n_genes} genes")

    model = ExpressionFromImage(n_genes=train.n_genes, embed_dim=args.embed_dim,
                                hidden_dim=args.hidden_dim,
                                pretrained_image=not args.no_imagenet).to(device)
    if args.init:
        state = torch.load(args.init, map_location="cpu", weights_only=False)
        weights = state.get("model", state)
        trunk = {k[len("image_encoder."):]: v for k, v in weights.items()
                 if k.startswith("image_encoder.")}
        missing = model.image_encoder.load_state_dict(trunk, strict=False)
        print(f"  initialised image encoder from {args.init} "
              f"({len(trunk) - len(missing.missing_keys)} tensors)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    def schedule(epoch):
        if epoch < args.warmup:
            return (epoch + 1) / max(args.warmup, 1)
        progress = (epoch - args.warmup) / max(args.epochs - args.warmup, 1)
        return 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    history, best = [], {"score": -np.inf, "epoch": -1}
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        loss_meter = RunningMean()
        for batch in train_loader:
            target = batch["expression"].to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=scaler is not None):
                loss = F.mse_loss(model(batch["image"].to(device, non_blocking=True)),
                                  target)
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            loss_meter.update(loss.item())
        scheduler.step()

        predicted, measured = collect(model, val_loader, device)
        correlations = gene_correlations(measured, predicted)
        score = float(np.mean(correlations))
        history.append({"epoch": epoch, "train_loss": loss_meter.value,
                        "val_mean_r": score,
                        "val_mse": float(np.mean((predicted - measured) ** 2)),
                        "lr": optimizer.param_groups[0]["lr"]})

        marker = ""
        if score > best["score"]:
            best = {"score": score, "epoch": epoch}
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "epoch": epoch}, out_dir / "checkpoint.pt")
            marker = "  *"
        print(f"  {epoch:3d}  train MSE {loss_meter.value:.4f}   "
              f"val mean r {score:.4f}{marker}")

        if epoch - best["epoch"] >= args.patience:
            print(f"  no improvement for {args.patience} epochs, stopping")
            break

    state = torch.load(out_dir / "checkpoint.pt", map_location=device,
                       weights_only=False)
    model.load_state_dict(state["model"])
    predicted, measured = collect(model, test_loader, device)
    correlations = gene_correlations(measured, predicted)

    order = np.argsort(-correlations)
    genes = train.hvg_names
    summary = {
        "tag": args.tag,
        "args": vars(args),
        "best_epoch": best["epoch"],
        "minutes": (time.time() - started) / 60.0,
        "parameters": count_parameters(model),
        "test_mean_r": float(correlations.mean()),
        "test_median_r": float(np.median(correlations)),
        "test_mse": float(np.mean((predicted - measured) ** 2)),
        "fraction_r_above_0.3": float((correlations > 0.3).mean()),
        "fraction_r_above_0.5": float((correlations > 0.5).mean()),
        "top_genes": [{"gene": str(genes[i]), "r": float(correlations[i])}
                      for i in order[:args.top_genes]],
        "bottom_genes": [{"gene": str(genes[i]), "r": float(correlations[i])}
                         for i in order[-20:]],
    }
    write_json(out_dir / "history.json", history)
    write_json(out_dir / "summary.json", summary)
    np.savez_compressed(out_dir / "gene_scores.npz", r=correlations,
                        genes=genes.astype(str))
    # Kept for the scatter panels in the dashboard.
    np.savez_compressed(out_dir / "test_predictions.npz",
                        predicted=predicted[:, order[:200]].astype(np.float32),
                        measured=measured[:, order[:200]].astype(np.float32),
                        genes=genes[order[:200]].astype(str))

    print(f"\n[{args.tag}] held-out donor: mean r {correlations.mean():.4f}  "
          f"median r {np.median(correlations):.4f}  "
          f"{(correlations > 0.3).mean() * 100:.1f}% of genes above r=0.3")
    print("  best: " + ", ".join(f"{genes[i]} {correlations[i]:.2f}"
                                 for i in order[:8]))


if __name__ == "__main__":
    main()

"""Supervised training for the layer classifier.

The same entry point covers the two unimodal baselines and the fused model, and
optionally starts from a self-supervised checkpoint or freezes the encoders for
a linear probe.

    python -m src.train --modality both --fusion gated --tag fusion_gated
    python -m src.train --modality both --init runs/ssl/checkpoint.pt --tag fusion_ssl
    python -m src.train --modality both --init runs/ssl/checkpoint.pt --freeze-encoders \
        --tag probe_ssl --epochs 30
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data.dataset import ExpressionAugment, ImageAugment, SpotDataset
from .data.samples import LAYERS
from .models.multimodal import SpotClassifier
from .utils.common import (RunningMean, count_parameters, describe_device,
                           get_device, set_seed, write_json)
from .utils.metrics import classification_report


def stratified_subset(labels, fraction, seed, minimum_per_class=5):
    """Pick a class-balanced fraction of the training indices.

    Sampling within each class keeps the rare layers present even at very small
    fractions. Without it a 1% draw can miss layer 4 entirely and the resulting
    curve says more about the draw than about label efficiency.
    """
    if fraction >= 1.0:
        return None
    rng = np.random.default_rng(seed)
    chosen = []
    for value in np.unique(labels):
        pool = np.flatnonzero(labels == value)
        take = max(minimum_per_class, int(round(len(pool) * fraction)))
        take = min(take, len(pool))
        chosen.append(rng.choice(pool, size=take, replace=False))
    return np.sort(np.concatenate(chosen))


def build_loaders(args):
    image_augment = ImageAugment(jitter=args.color_jitter, gray=args.grayscale_prob)
    expression_augment = ExpressionAugment(dropout=args.gene_dropout,
                                           noise=args.gene_noise)

    train = SpotDataset(args.processed, "train", labelled_only=True,
                        image_augment=image_augment,
                        expression_augment=expression_augment, seed=args.seed)
    if args.label_fraction < 1.0:
        subset = stratified_subset(train.labels, args.label_fraction, args.seed)
        train = SpotDataset(args.processed, "train", labelled_only=True,
                            image_augment=image_augment,
                            expression_augment=expression_augment,
                            subset=subset, seed=args.seed)

    val = SpotDataset(args.processed, "val", labelled_only=True)
    test = SpotDataset(args.processed, "test", labelled_only=True)

    common = dict(batch_size=args.batch_size, num_workers=args.workers,
                  pin_memory=True, persistent_workers=args.workers > 0)
    return (
        train,
        DataLoader(train, shuffle=True, drop_last=len(train) > args.batch_size, **common),
        DataLoader(val, shuffle=False, **common),
        DataLoader(test, shuffle=False, **common),
    )


def forward_batch(model, batch, device):
    expression = batch["expression"].to(device, non_blocking=True) \
        if model.expression_encoder is not None else None
    image = batch["image"].to(device, non_blocking=True) \
        if model.image_encoder is not None else None
    return model(expression=expression, image=image)


def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    loss_meter = RunningMean()
    logits_all, labels_all = [], []

    for batch in loader:
        labels = batch["label"].to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with torch.autocast("cuda", enabled=scaler is not None):
                logits = forward_batch(model, batch, device)
                loss = criterion(logits, labels)
        if training:
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
        loss_meter.update(loss.item(), labels.numel())
        logits_all.append(logits.detach().float().cpu())
        labels_all.append(labels.cpu())

    logits_all = torch.cat(logits_all)
    labels_all = torch.cat(labels_all)
    probabilities = torch.softmax(logits_all, dim=1).numpy()
    report = classification_report(labels_all.numpy(), probabilities.argmax(axis=1),
                                   probabilities=probabilities,
                                   n_classes=len(LAYERS), class_names=LAYERS)
    report["loss"] = loss_meter.value
    return report


def class_weights(counts, device, power=0.5):
    """Mildly inverse-frequency weights.

    Full inverse frequency over-corrects here and drags precision on the wide
    layers down further than the gain on the thin ones justifies, so the
    weights are softened with a square root.
    """
    counts = np.asarray(counts, dtype=np.float64)
    counts[counts == 0] = 1.0
    weights = (counts.sum() / counts) ** power
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--out", default="runs")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--modality", default="both",
                        choices=["expression", "image", "both"])
    parser.add_argument("--fusion", default="gated",
                        choices=["concat", "gated", "bilinear"])
    parser.add_argument("--init", default=None,
                        help="self-supervised checkpoint to initialise encoders from")
    parser.add_argument("--freeze-encoders", action="store_true",
                        help="train only the fusion and head (linear-probe setting)")
    parser.add_argument("--no-imagenet", action="store_true",
                        help="random image-encoder init instead of ImageNet weights")
    parser.add_argument("--label-fraction", type=float, default=1.0)
    # Defaults are the settings that came out of tuning on the validation
    # sections. Six thousand image parameters per labelled spot is a lot, and
    # the lighter regularisation this started with reached a training macro-F1
    # of 0.99 while validation stalled around 0.73. Pushing dropout, weight
    # decay and both augmentations up costs nothing in wall time and moved the
    # held-out donor from 0.576 to 0.599 macro-F1.
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--encoder-lr-scale", type=float, default=0.1,
                        help="lower learning rate for pretrained encoders")
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.45)
    parser.add_argument("--color-jitter", type=float, default=0.4)
    parser.add_argument("--grayscale-prob", type=float, default=0.0)
    parser.add_argument("--gene-dropout", type=float, default=0.3)
    parser.add_argument("--gene-noise", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(not args.cpu)
    out_dir = Path(args.out) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    train_set, train_loader, val_loader, test_loader = build_loaders(args)
    print(f"[{args.tag}] device {describe_device(device)}")
    print(f"  train {len(train_set)}  val {len(val_loader.dataset)}  "
          f"test {len(test_loader.dataset)}  genes {train_set.n_genes}")

    model = SpotClassifier(
        n_genes=train_set.n_genes, n_classes=len(LAYERS), modality=args.modality,
        fusion=args.fusion, embed_dim=args.embed_dim, hidden_dim=args.hidden_dim,
        dropout=args.dropout, pretrained_image=not args.no_imagenet,
    ).to(device)

    if args.init:
        state = torch.load(args.init, map_location="cpu", weights_only=False)
        model.load_pretrained(state.get("model", state))

    encoder_parameters, head_parameters = [], []
    for name, parameter in model.named_parameters():
        if name.startswith(("expression_encoder.", "image_encoder.")):
            if args.freeze_encoders:
                parameter.requires_grad_(False)
                continue
            encoder_parameters.append(parameter)
        else:
            head_parameters.append(parameter)

    # Encoders that arrive pretrained are nudged rather than retrained; a head
    # starting from scratch needs the full rate.
    encoder_lr = args.lr * (args.encoder_lr_scale if args.init else 1.0)
    groups = [{"params": head_parameters, "lr": args.lr}]
    if encoder_parameters:
        groups.append({"params": encoder_parameters, "lr": encoder_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)

    def schedule(epoch):
        if epoch < args.warmup:
            return (epoch + 1) / max(args.warmup, 1)
        progress = (epoch - args.warmup) / max(args.epochs - args.warmup, 1)
        return 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(train_set.class_counts(len(LAYERS)), device),
        label_smoothing=args.label_smoothing)

    print(f"  {count_parameters(model) / 1e6:.2f}M trainable parameters")

    history, best = [], {"macro_f1": -1.0, "epoch": -1}
    started = time.time()
    for epoch in range(args.epochs):
        train_report = run_epoch(model, train_loader, device, criterion,
                                 optimizer, scaler)
        with torch.no_grad():
            val_report = run_epoch(model, val_loader, device, criterion)
        scheduler.step()

        history.append({
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_report["loss"],
            "train_accuracy": train_report["accuracy"],
            "train_macro_f1": train_report["macro_f1"],
            "val_loss": val_report["loss"],
            "val_accuracy": val_report["accuracy"],
            "val_macro_f1": val_report["macro_f1"],
        })

        marker = ""
        if val_report["macro_f1"] > best["macro_f1"]:
            best = {"macro_f1": val_report["macro_f1"], "epoch": epoch}
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "epoch": epoch, "val": val_report},
                       out_dir / "checkpoint.pt")
            marker = "  *"
        print(f"  {epoch:3d}  train {train_report['loss']:.3f}/"
              f"{train_report['macro_f1']:.3f}   val {val_report['loss']:.3f}/"
              f"{val_report['macro_f1']:.3f}{marker}")

        if epoch - best["epoch"] >= args.patience:
            print(f"  no improvement for {args.patience} epochs, stopping")
            break

    state = torch.load(out_dir / "checkpoint.pt", map_location=device,
                       weights_only=False)
    model.load_state_dict(state["model"])
    with torch.no_grad():
        test_report = run_epoch(model, test_loader, device, criterion)

    elapsed = time.time() - started
    summary = {
        "tag": args.tag,
        "args": vars(args),
        "best_epoch": best["epoch"],
        "epochs_run": len(history),
        "minutes": elapsed / 60.0,
        "n_train": len(train_set),
        "parameters": count_parameters(model),
        "val": state["val"],
        "test": test_report,
    }
    write_json(out_dir / "history.json", history)
    write_json(out_dir / "summary.json", summary)

    print(f"\n[{args.tag}] held-out donor: acc {test_report['accuracy']:.4f}  "
          f"macro-F1 {test_report['macro_f1']:.4f}  "
          f"balanced acc {test_report['balanced_accuracy']:.4f}  "
          f"ARI {test_report['ari']:.4f}  ({elapsed / 60:.1f} min)")
    print(json.dumps({k: round(v["f1"], 3)
                      for k, v in test_report["per_class"].items()}))


if __name__ == "__main__":
    main()

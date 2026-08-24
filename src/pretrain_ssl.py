"""Self-supervised cross-modal pretraining.

No layer annotation is read at any point in this script. The only supervision
is the pairing that the assay provides for free: the transcriptome measured at
a spot and the piece of tissue photographed at that same spot describe the same
few cells, and the encoders are trained to recognise which expression vector
goes with which crop.

A caveat on what this can and cannot buy here. Self-supervised pretraining is
often justified by unlabelled data going to waste, and that argument does not
apply to this dataset: the spatialLIBD annotators covered essentially every
spot, so there is almost no unlabelled pool to exploit. The question worth
asking instead is whether the pairing alone teaches the encoders enough that
they need fewer labels afterwards, which is what the label-efficiency sweep in
scripts/run_all.py measures, and whether the representation it produces is any
good on its own, which the frozen probes and the retrieval scores measure.

The reason to expect anything at all is what the task forces the model to
encode. To match a crop to its transcriptome the image encoder has to represent
cell density, nuclear size and where the spot sits in the cortical ribbon,
which is close to what a layer classifier needs. That makes it a sensible
starting point rather than an arbitrary one.

    python -m src.pretrain_ssl --tag ssl --epochs 80
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.dataset import ExpressionAugment, ImageAugment, PairedSSLDataset
from .data.samples import LAYERS
from .models.multimodal import CrossModalContrastive
from .utils.common import (RunningMean, count_parameters, describe_device,
                           get_device, set_seed, write_json)
from .utils.metrics import retrieval_metrics


def collate_device(batch, device):
    return {
        "expression": batch["expression"].to(device, non_blocking=True),
        "image": batch["image"].to(device, non_blocking=True),
        "array_row": batch["array_row"].to(device, non_blocking=True),
        "array_col": batch["array_col"].to(device, non_blocking=True),
        "section_index": batch["section_index"].to(device, non_blocking=True),
    }


@torch.no_grad()
def embed_split(model, loader, device):
    model.eval()
    expression, image, labels, rows, cols, sections = [], [], [], [], [], []
    for batch in loader:
        z_expression, z_image = model.encode(
            batch["expression"].to(device), batch["image"].to(device))
        expression.append(z_expression.float().cpu().numpy())
        image.append(z_image.float().cpu().numpy())
        labels.append(batch["label"].numpy())
        rows.append(batch["array_row"].numpy())
        cols.append(batch["array_col"].numpy())
        sections.append(batch["section_index"].numpy())
    return (np.concatenate(expression), np.concatenate(image),
            np.concatenate(labels), np.concatenate(rows),
            np.concatenate(cols), np.concatenate(sections))


def neighbour_mask(rows, cols, sections, radius=2):
    """True where two spots are close enough to be near-duplicates."""
    d_row = np.abs(rows[:, None] - rows[None, :])
    d_col = np.abs(cols[:, None] - cols[None, :])
    hexagonal = np.maximum(d_row, (d_row + d_col) / 2)
    same = sections[:, None] == sections[None, :]
    mask = same & (hexagonal <= radius)
    np.fill_diagonal(mask, False)
    return mask


def knn_probe(train_embeddings, train_labels, test_embeddings, test_labels, k=25,
              max_fit=10000, max_query=6000, seed=0):
    """How well a frozen embedding separates the layers, with nothing fitted.

    A linear probe still trains weights; a nearest-neighbour vote does not, so
    this reads the geometry of the representation directly. Both sides are
    capped because this runs several times during pretraining and a brute-force
    cosine search over the full 24k x 14k pair is minutes of CPU for a number
    that is stable well before then.
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(seed)
    fit_rows = np.flatnonzero(train_labels >= 0)
    query_rows = np.flatnonzero(test_labels >= 0)
    if len(fit_rows) < k or not len(query_rows):
        return None
    if len(fit_rows) > max_fit:
        fit_rows = rng.choice(fit_rows, max_fit, replace=False)
    if len(query_rows) > max_query:
        query_rows = rng.choice(query_rows, max_query, replace=False)

    model = KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="distance")
    model.fit(train_embeddings[fit_rows], train_labels[fit_rows])
    predicted = model.predict(test_embeddings[query_rows])
    truth = test_labels[query_rows]
    return {
        "accuracy": float((predicted == truth).mean()),
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "k": k,
        "n_fit": int(len(fit_rows)),
        "n_query": int(len(query_rows)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--out", default="runs")
    parser.add_argument("--tag", default="ssl")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256,
                        help="also the number of negatives per positive")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--neighbour-radius", type=int, default=2,
                        help="mask spots within this many lattice steps from the negatives")
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--color-jitter", type=float, default=0.4)
    parser.add_argument("--grayscale-prob", type=float, default=0.1)
    parser.add_argument("--gene-dropout", type=float, default=0.25)
    parser.add_argument("--gene-noise", type=float, default=0.15)
    parser.add_argument("--no-imagenet", action="store_true")
    parser.add_argument("--labelled-only", action="store_true",
                        help="ablation: drop the unannotated spots the SSL stage can use")
    parser.add_argument("--retrieval-gallery", type=int, default=4096,
                        help="held-out spots to rank against for the recall scores")
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(not args.cpu)
    out_dir = Path(args.out) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    image_augment = ImageAugment(jitter=args.color_jitter, gray=args.grayscale_prob)
    expression_augment = ExpressionAugment(dropout=args.gene_dropout,
                                           noise=args.gene_noise)

    train = PairedSSLDataset(args.processed, "train",
                             labelled_only=args.labelled_only,
                             image_augment=image_augment,
                             expression_augment=expression_augment, seed=args.seed)
    # Probing sets are read without augmentation.
    train_clean = PairedSSLDataset(args.processed, "train", labelled_only=True)
    test_clean = PairedSSLDataset(args.processed, "test", labelled_only=True)

    common = dict(batch_size=args.batch_size, num_workers=args.workers,
                  pin_memory=True, persistent_workers=args.workers > 0)
    train_loader = DataLoader(train, shuffle=True, drop_last=True, **common)
    train_eval_loader = DataLoader(train_clean, shuffle=False, **common)
    test_eval_loader = DataLoader(test_clean, shuffle=False, **common)

    labelled = int((train.labels >= 0).sum())
    print(f"[{args.tag}] device {describe_device(device)}")
    print(f"  {len(train)} training spots, no annotation read "
          f"({labelled} of them happen to be annotated)")

    model = CrossModalContrastive(
        n_genes=train.n_genes, embed_dim=args.embed_dim,
        projection_dim=args.projection_dim, temperature=args.temperature,
        neighbour_radius=args.neighbour_radius,
        pretrained_image=not args.no_imagenet, dropout=args.dropout,
    ).to(device)
    print(f"  {count_parameters(model) / 1e6:.2f}M parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    def schedule(epoch):
        if epoch < args.warmup:
            return (epoch + 1) / max(args.warmup, 1)
        progress = (epoch - args.warmup) / max(args.epochs - args.warmup, 1)
        return 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    history, probes = [], []
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        loss_meter, accuracy_meter = RunningMean(), RunningMean()
        for batch in train_loader:
            inputs = collate_device(batch, device)
            with torch.autocast("cuda", enabled=scaler is not None):
                loss, stats = model.loss(**inputs)
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
            loss_meter.update(stats["loss"].item())
            accuracy_meter.update(stats["batch_accuracy"].item())
        scheduler.step()

        temperature = float(1.0 / model.logit_scale.exp().clamp(max=100.0))
        history.append({"epoch": epoch, "loss": loss_meter.value,
                        "batch_accuracy": accuracy_meter.value,
                        "temperature": temperature,
                        "lr": optimizer.param_groups[0]["lr"]})
        print(f"  {epoch:3d}  loss {loss_meter.value:.4f}  "
              f"in-batch match {accuracy_meter.value:.3f}  T {temperature:.4f}")

        last = epoch == args.epochs - 1
        if args.eval_every and (epoch % args.eval_every == 0 or last):
            train_expression, train_image, train_labels, *_ = embed_split(
                model, train_eval_loader, device)
            (test_expression, test_image, test_labels,
             rows, cols, sections) = embed_split(model, test_eval_loader, device)

            # Retrieval runs against a fixed-size gallery. The full held-out
            # donor would need a 14k x 14k similarity matrix and several more
            # of the same shape for the neighbour mask, and recall against a
            # gallery that keeps growing with the dataset is harder to compare
            # against published numbers anyway.
            gallery = np.random.default_rng(args.seed).permutation(
                len(test_expression))[:args.retrieval_gallery]
            mask = neighbour_mask(rows[gallery], cols[gallery], sections[gallery],
                                  args.neighbour_radius)
            retrieval = retrieval_metrics(test_expression[gallery],
                                          test_image[gallery], exclude=mask)
            probe = knn_probe(train_expression, train_labels,
                              test_expression, test_labels, seed=args.seed)
            image_probe = knn_probe(train_image, train_labels,
                                    test_image, test_labels, seed=args.seed)
            probes.append({"epoch": epoch, "retrieval": retrieval,
                           "knn_expression": probe, "knn_image": image_probe})
            print(f"       retrieval R@1 {retrieval['recall@1']:.3f} "
                  f"R@10 {retrieval['recall@10']:.3f} "
                  f"median rank {retrieval['median_rank']:.0f}  |  "
                  f"kNN layer F1  expr {probe['macro_f1']:.3f} "
                  f"image {image_probe['macro_f1']:.3f}")

    torch.save({"model": model.state_dict(), "args": vars(args),
                "epochs": args.epochs}, out_dir / "checkpoint.pt")
    write_json(out_dir / "history.json", history)
    write_json(out_dir / "probes.json", probes)
    write_json(out_dir / "summary.json", {
        "tag": args.tag,
        "args": vars(args),
        "minutes": (time.time() - started) / 60.0,
        "n_spots": len(train),
        "n_labelled": labelled,
        "n_unlabelled_used": len(train) - labelled,
        "final": probes[-1] if probes else None,
        "classes": LAYERS,
    })
    print(f"\n[{args.tag}] done in {(time.time() - started) / 60:.1f} min "
          f"-> {out_dir / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()

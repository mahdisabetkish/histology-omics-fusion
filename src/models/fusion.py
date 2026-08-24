"""Ways of combining the two embeddings before the classifier."""

import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    """Concatenate and project. The baseline every other variant is measured against."""

    def __init__(self, embed_dim, out_dim=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_dim = out_dim

    def forward(self, expression, image):
        return self.net(torch.cat([expression, image], dim=-1))

    def gates(self, expression, image):
        return None


class GatedFusion(nn.Module):
    """Per-sample, per-modality gating.

    The two modalities are not equally informative everywhere on a section.
    Expression separates the layers well in the middle of the cortex; near the
    white matter boundary the histology is the clearer signal, and over spots
    with low RNA capture the expression vector is close to noise. A gate
    computed from both embeddings lets the model down-weight whichever modality
    is unreliable for a given spot instead of committing to one fixed ratio.

    The gate values are also worth looking at directly, so `gates` exposes them
    for the spatial maps in the dashboard.
    """

    def __init__(self, embed_dim, out_dim=256, dropout=0.3, hidden=128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )
        self.project = nn.Sequential(
            nn.Linear(embed_dim * 2, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_dim = out_dim

    def _weights(self, expression, image):
        return torch.softmax(self.gate(torch.cat([expression, image], dim=-1)), dim=-1)

    def forward(self, expression, image):
        weights = self._weights(expression, image)
        scaled = torch.cat([
            expression * weights[:, 0:1],
            image * weights[:, 1:2],
        ], dim=-1)
        return self.project(scaled)

    @torch.no_grad()
    def gates(self, expression, image):
        return self._weights(expression, image)


class BilinearFusion(nn.Module):
    """Low-rank bilinear pooling.

    Captures interactions between the two embeddings that an additive scheme
    cannot. It is the most expressive of the three and the most eager to
    overfit, which is why it is reported as an ablation rather than used as the
    main model.
    """

    def __init__(self, embed_dim, out_dim=256, rank=256, dropout=0.3):
        super().__init__()
        self.left = nn.Linear(embed_dim, rank)
        self.right = nn.Linear(embed_dim, rank)
        self.net = nn.Sequential(
            nn.Linear(rank + embed_dim * 2, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_dim = out_dim

    def forward(self, expression, image):
        interaction = self.left(expression) * self.right(image)
        return self.net(torch.cat([interaction, expression, image], dim=-1))

    def gates(self, expression, image):
        return None


FUSIONS = {"concat": ConcatFusion, "gated": GatedFusion, "bilinear": BilinearFusion}


def build_fusion(name, embed_dim, out_dim=256, dropout=0.3):
    if name not in FUSIONS:
        raise KeyError(f"unknown fusion '{name}', expected one of {sorted(FUSIONS)}")
    return FUSIONS[name](embed_dim, out_dim=out_dim, dropout=dropout)

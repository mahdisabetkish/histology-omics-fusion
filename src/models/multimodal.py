"""Classifier and self-supervised wrappers around the two encoders."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import ExpressionEncoder, ImageEncoder, ProjectionHead
from .fusion import build_fusion


class SpotClassifier(nn.Module):
    """Predicts the cortical layer of a spot from one or both modalities.

    `modality` selects between the two unimodal baselines and the fused model.
    Keeping all three behind one class means the training loop, the
    regularisation and the evaluation code are shared, so a difference in the
    reported numbers comes from the input and not from the surrounding setup.
    """

    def __init__(self, n_genes, n_classes=7, modality="both", fusion="gated",
                 embed_dim=256, hidden_dim=256, dropout=0.3,
                 pretrained_image=True, small_stem=True):
        super().__init__()
        if modality not in ("expression", "image", "both"):
            raise ValueError(f"unknown modality '{modality}'")
        self.modality = modality

        self.expression_encoder = None
        self.image_encoder = None
        if modality in ("expression", "both"):
            self.expression_encoder = ExpressionEncoder(n_genes, embed_dim=embed_dim,
                                                        dropout=dropout + 0.1)
        if modality in ("image", "both"):
            self.image_encoder = ImageEncoder(embed_dim=embed_dim,
                                              pretrained=pretrained_image,
                                              small_stem=small_stem)

        if modality == "both":
            self.fusion = build_fusion(fusion, embed_dim, out_dim=hidden_dim,
                                       dropout=dropout)
            head_dim = self.fusion.out_dim
        else:
            self.fusion = None
            head_dim = embed_dim

        self.head = nn.Sequential(
            nn.Linear(head_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def embed(self, expression=None, image=None):
        z_expression = self.expression_encoder(expression) \
            if self.expression_encoder is not None else None
        z_image = self.image_encoder(image) if self.image_encoder is not None else None
        return z_expression, z_image

    def forward(self, expression=None, image=None, return_embedding=False):
        z_expression, z_image = self.embed(expression, image)
        if self.modality == "expression":
            joint = z_expression
        elif self.modality == "image":
            joint = z_image
        else:
            joint = self.fusion(z_expression, z_image)
        logits = self.head(joint)
        if return_embedding:
            return logits, joint, z_expression, z_image
        return logits

    @torch.no_grad()
    def modality_gates(self, expression, image):
        """Softmax weight the gated fusion assigns to each modality, or None."""
        if self.modality != "both" or not hasattr(self.fusion, "gates"):
            return None
        z_expression, z_image = self.embed(expression, image)
        return self.fusion.gates(z_expression, z_image)

    def load_pretrained(self, state_dict, verbose=True):
        """Copy encoder weights from a self-supervised checkpoint.

        Only the encoder trunks are transferred. Projection heads belong to the
        contrastive objective and the classifier head does not exist yet.
        """
        own = self.state_dict()
        transferred = {k: v for k, v in state_dict.items()
                       if k in own and own[k].shape == v.shape
                       and k.startswith(("expression_encoder.", "image_encoder."))}
        own.update(transferred)
        self.load_state_dict(own)
        if verbose:
            total = sum(1 for k in own if k.startswith(("expression_encoder.",
                                                        "image_encoder.")))
            print(f"  loaded {len(transferred)}/{total} encoder tensors from SSL checkpoint")
        return len(transferred)


def hex_distance(rows, cols):
    """Pairwise distance on the Visium hexagonal lattice, in spot steps.

    Array coordinates advance by two columns between horizontal neighbours and
    by one row and one column diagonally, so plain Euclidean distance on
    (row, col) does not give integer steps.
    """
    d_row = (rows[:, None] - rows[None, :]).abs()
    d_col = (cols[:, None] - cols[None, :]).abs()
    return torch.maximum(d_row, (d_row + d_col) / 2)


class CrossModalContrastive(nn.Module):
    """Aligns expression and histology without using any layer label.

    Each spot is a positive pair: its transcriptome and the H&E crop taken at
    the same physical location. Every other spot in the batch is a negative.
    The encoders have to discover what the two views share, which for cortex is
    largely laminar identity, and they do it from the pairing alone.

    One correction matters. Visium spots are 100 um apart, close enough that a
    spot and its immediate neighbours frequently contain the same cells. Left
    alone, the loss pushes those apart as hard negatives and works directly
    against the structure worth learning. Pairs within `neighbour_radius` steps
    on the same section are therefore dropped from the denominator.
    """

    def __init__(self, n_genes, embed_dim=256, projection_dim=128,
                 temperature=0.07, neighbour_radius=2, pretrained_image=True,
                 small_stem=True, dropout=0.3):
        super().__init__()
        self.expression_encoder = ExpressionEncoder(n_genes, embed_dim=embed_dim,
                                                    dropout=dropout)
        self.image_encoder = ImageEncoder(embed_dim=embed_dim,
                                          pretrained=pretrained_image,
                                          small_stem=small_stem)
        self.expression_projection = ProjectionHead(embed_dim, out_dim=projection_dim)
        self.image_projection = ProjectionHead(embed_dim, out_dim=projection_dim)
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / temperature).log())
        self.neighbour_radius = neighbour_radius

    def forward(self, expression, image):
        z_expression = self.expression_encoder(expression)
        z_image = self.image_encoder(image)
        return (self.expression_projection(z_expression),
                self.image_projection(z_image))

    def loss(self, expression, image, array_row=None, array_col=None,
             section_index=None):
        p_expression, p_image = self.forward(expression, image)
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = scale * p_expression @ p_image.t()

        n = logits.shape[0]
        target = torch.arange(n, device=logits.device)

        if self.neighbour_radius > 0 and array_row is not None:
            same_section = section_index[:, None] == section_index[None, :]
            close = hex_distance(array_row.float(), array_col.float()) <= self.neighbour_radius
            mask = same_section & close
            mask.fill_diagonal_(False)
            logits = logits.masked_fill(mask, float("-inf"))

        loss = 0.5 * (F.cross_entropy(logits, target)
                      + F.cross_entropy(logits.t(), target))

        with torch.no_grad():
            accuracy = (logits.argmax(dim=1) == target).float().mean()
        return loss, {"loss": loss.detach(), "batch_accuracy": accuracy,
                      "temperature": 1.0 / scale.detach()}

    @torch.no_grad()
    def encode(self, expression, image):
        """Embeddings used for retrieval and probing, before the projection heads."""
        return self.expression_encoder(expression), self.image_encoder(image)


class ExpressionFromImage(nn.Module):
    """Regresses the variable-gene panel from the H&E crop alone.

    A direct test of how much transcriptional signal the histology carries: if
    a gene's spatial pattern can be recovered from morphology, the two
    modalities are genuinely redundant for it, and if it cannot, the expression
    arm of the fused model is contributing something the image does not have.
    """

    def __init__(self, n_genes, embed_dim=256, hidden_dim=512, dropout=0.2,
                 pretrained_image=True, small_stem=True):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim=embed_dim,
                                          pretrained=pretrained_image,
                                          small_stem=small_stem)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_genes),
        )

    def forward(self, image):
        return self.head(self.image_encoder(image))

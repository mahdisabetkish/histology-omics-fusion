"""Per-modality encoders.

Both encoders map their input to a shared embedding width so the fusion module
and the contrastive objective can treat them symmetrically.
"""

import torch
import torch.nn as nn
import torchvision


class ExpressionEncoder(nn.Module):
    """MLP over the standardised variable-gene panel.

    The first layer carries almost all of the parameters, so it gets the
    heaviest dropout. Wider or deeper variants overfit quickly: with roughly
    ten thousand labelled training spots there is not much to be gained past
    two hidden layers.
    """

    def __init__(self, n_genes, hidden=(1024, 512), embed_dim=256,
                 dropout=0.4, input_dropout=0.1):
        super().__init__()
        layers = [nn.Dropout(input_dropout)]
        width = n_genes
        for i, size in enumerate(hidden):
            layers += [
                nn.Linear(width, size),
                nn.BatchNorm1d(size),
                nn.GELU(),
                nn.Dropout(dropout if i == 0 else dropout * 0.5),
            ]
            width = size
        layers.append(nn.Linear(width, embed_dim))
        self.net = nn.Sequential(*layers)
        self.embed_dim = embed_dim

    def forward(self, x):
        return self.net(x)


class ImageEncoder(nn.Module):
    """ResNet-18 trunk over the H&E crop.

    The crops are 112 px, half the resolution the stock network expects. Its
    stem downsamples fourfold before the first residual stage, which at this
    input size leaves nuclei smaller than a pixel by the time any features are
    computed, and nuclear density and size are most of what separates one
    cortical layer from the next.

    Dropping the max pool while keeping the 7x7 stride-2 convolution puts the
    first stage at 56x56 instead of 28x28. Replacing the stem outright with a
    stride-1 convolution preserves more still, but measured on the GTX 1080 Ti
    used here it ran about ten times slower than stock for the whole study,
    while this version costs under three times stock and keeps the pretrained
    stem weights that a custom convolution would throw away.
    """

    def __init__(self, embed_dim=256, pretrained=True, small_stem=True,
                 dropout=0.2, freeze_bn=False):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        trunk = torchvision.models.resnet18(weights=weights)

        if small_stem:
            trunk.maxpool = nn.Identity()

        n_features = trunk.fc.in_features
        trunk.fc = nn.Identity()
        self.trunk = trunk
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_features, embed_dim))
        self.embed_dim = embed_dim
        self.freeze_bn = freeze_bn

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_bn:
            for module in self.trunk.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def forward(self, x):
        return self.head(self.trunk(x))


class ProjectionHead(nn.Module):
    """Two-layer projection used only by the contrastive objective.

    Kept separate from the encoder so it can be thrown away after pretraining.
    Representations one layer before the contrastive loss transfer better than
    the projected ones, which is the standard finding from SimCLR and holds
    here too.
    """

    def __init__(self, in_dim, hidden=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return torch.nn.functional.normalize(self.net(x), dim=-1)

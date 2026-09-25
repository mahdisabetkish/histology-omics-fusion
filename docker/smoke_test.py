"""Check that a built image can run the models, without any data.

Imports every module, loads the baked-in ImageNet weights with the network
disabled, and pushes a random batch through every model the pipeline trains,
including one backward pass. Takes a few seconds on CPU.

    docker run --rm --network none \
        -v "$PWD/docker/smoke_test.py:/tmp/smoke_test.py:ro" \
        IMAGE python /tmp/smoke_test.py
"""

import importlib
import pkgutil
import sys

import torch

sys.path.insert(0, "/app")

import src  # noqa: E402
from src.data.samples import LAYERS  # noqa: E402
from src.models.multimodal import (CrossModalContrastive,  # noqa: E402
                                   ExpressionFromImage, SpotClassifier)

N_GENES, BATCH = 64, 4


def main():
    for module in pkgutil.walk_packages(src.__path__, "src."):
        importlib.import_module(module.name)
    print("imports      ok")

    torch.manual_seed(0)
    expression = torch.randn(BATCH, N_GENES)
    image = torch.randn(BATCH, 3, 112, 112)

    # pretrained_image=True reads the weights from TORCH_HOME; run with
    # --network none and this fails if they were not baked into the image.
    for modality, fusion in [("expression", "gated"), ("image", "gated"),
                             ("both", "concat"), ("both", "gated"),
                             ("both", "bilinear")]:
        model = SpotClassifier(n_genes=N_GENES, n_classes=len(LAYERS),
                               modality=modality, fusion=fusion).eval()
        with torch.no_grad():
            logits = model(expression=expression, image=image)
        assert logits.shape == (BATCH, len(LAYERS)), logits.shape
        print(f"classifier   {modality}/{fusion} ok")

    # Coordinates and section ids exercise the neighbour masking as well.
    contrastive = CrossModalContrastive(n_genes=N_GENES).train()
    loss, _ = contrastive.loss(expression, image,
                               array_row=torch.tensor([0, 0, 1, 9]),
                               array_col=torch.tensor([0, 2, 1, 9]),
                               section_index=torch.zeros(BATCH, dtype=torch.long))
    loss.backward()
    assert torch.isfinite(loss), loss
    print(f"contrastive  ok (loss {loss.item():.3f}, backward ok)")

    regression = ExpressionFromImage(n_genes=N_GENES).eval()
    with torch.no_grad():
        predicted = regression(image)
    assert predicted.shape == (BATCH, N_GENES), predicted.shape
    print("regression   ok")

    print("smoke test passed")


if __name__ == "__main__":
    main()

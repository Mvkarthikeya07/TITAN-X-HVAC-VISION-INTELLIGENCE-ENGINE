"""
ResNet18 Embedding Model Definition for InTakeoff Pipeline.

Defines the headless ResNet18 architecture used for generating 512-dimensional
feature embeddings of detected HVAC symbols.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class ResNet18Embedder(nn.Module):
    """
    Headless ResNet18 that outputs a 512-dim L2-normalized embedding vector.
    The final classification layer is removed and replaced with an identity layer.
    """

    def __init__(self, pretrained: bool = True, embedding_dim: int = 512):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Load pretrained ResNet18
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = models.resnet18(weights=weights)

        # Remove classification head (fc layer) — keep everything up to avgpool
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])

        # Optional projection head if embedding_dim != 512
        if embedding_dim != 512:
            self.projection = nn.Linear(512, embedding_dim)
        else:
            self.projection = nn.Identity()

        logger.info(f"Initialized ResNet18Embedder (pretrained={pretrained}, dim={embedding_dim})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 3, 224, 224).

        Returns:
            Tensor of shape (B, embedding_dim), L2-normalized.
        """
        features = self.backbone(x)          # (B, 512, 1, 1)
        features = features.squeeze(-1).squeeze(-1)  # (B, 512)
        embeddings = self.projection(features)       # (B, embedding_dim)

        # L2 normalize
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings


def create_embedding_model(pretrained: bool = True, embedding_dim: int = 512) -> ResNet18Embedder:
    """Factory function to create and return the embedding model."""
    model = ResNet18Embedder(pretrained=pretrained, embedding_dim=embedding_dim)
    model.eval()
    return model

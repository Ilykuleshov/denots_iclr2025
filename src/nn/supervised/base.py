"""File with the base class for the supervised backbones."""

from abc import ABC, abstractmethod
from typing import Literal

from torch import Tensor, nn


class SupervisedBackbone(nn.Module, ABC):
    """The base class for the supervised backbones."""

    hidden_dim: int
    evs_nan: Literal["keep", "fill", "zero"]
    cts_nan: Literal["keep", "fill", "zero"]
    requires_idx: bool = False

    @abstractmethod
    def forward(
        self, stat_emb: Tensor | None, embedding: Tensor, time: Tensor, mask: Tensor
    ):
        """Pass the embeddings through the backbone."""

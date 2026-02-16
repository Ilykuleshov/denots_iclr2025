"""File with the Mamba2 model.

Taken from examples from the Mamba repository.
https://github.com/state-spaces/mamba
"""

import torch
from mamba_ssm.modules.mamba2 import Mamba2
from torch import Tensor, nn

from ...mask_utils import masklast
from ..layers import TimePositionalEncoding
from .base import SupervisedBackbone


class MambaModel(SupervisedBackbone):
    """The Mamba2 model."""

    evs_nan = "fill"
    cts_nan = "fill"

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layer: int,
    ) -> None:
        """Initialize the Mamba2 model."""
        super().__init__()

        self.mamba = nn.Sequential()

        for i in range(n_layer):
            layer = Mamba2(
                d_model=hidden_dim,
                headdim=8,
                d_state=16,
                d_conv=4,
                expand=2,
            )

            self.mamba.add_module(f"layer_{i}", layer)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.pos_enc = TimePositionalEncoding(hidden_dim, max_len=5000)
        self.proj = nn.Linear(input_dim, self.hidden_dim)

    def forward(
        self, stat_emb: Tensor | None, embedding: Tensor, time: Tensor, mask: Tensor
    ):
        """Forward pass of the Mamba2 model."""
        embedding = self.proj(embedding)
        embedding = embedding + self.pos_enc(time)
        embedding = self.mamba(embedding.contiguous()).contiguous()
        embedding = masklast(embedding, mask, 1)
        if stat_emb is None:
            stat_emb = torch.zeros_like(embedding)

        return embedding + stat_emb

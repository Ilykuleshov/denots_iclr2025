"""Module with the Tempformer class."""

import torch
from torch import Tensor, nn

from ..encoder import Encoder
from ..layers import TimePositionalEncoding
from .base import SupervisedBackbone


class TempFormer(SupervisedBackbone):
    """Supervised Transformer classifier with temporal embeddings.

    Returns the embedding corresponding to the CLS token.
    """

    evs_nan = "fill"
    cts_nan = "fill"

    def __init__(
        self,
        input_dim: Encoder,
        hidden_dim: int,
        nhead: int,
        num_layers: int,
    ):
        """Initialize the Tempformer.

        Args:
        ----
            input_dim (Encoder): The input encoder.
            hidden_dim (int): The hidden dimension.
            nhead (int): The number of attention heads.
            num_layers (int): The number of layers.

        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.num_layers = num_layers

        self.projector = nn.Linear(input_dim, self.hidden_dim)
        self.hdim = self.hidden_dim // self.nhead
        self.pos_enc = TimePositionalEncoding(hidden_dim, max_len=1000)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.nhead,
            batch_first=True,
            activation=nn.GELU(),
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, self.num_layers)

        self.transformer.compile()
        self.projector = nn.Linear(input_dim, self.hidden_dim)
        self.cls_learnable = nn.Parameter(torch.randn(hidden_dim))

    def forward(
        self, stat_emb: Tensor | None, embedding: Tensor, time: Tensor, mask: Tensor
    ):
        """Pass the embeddings through the transformer, prepending CLS token.

        Takes into account `mask` as attention mask for **non-padding** tokens.
        """
        B = embedding.size(0)

        embedding = self.projector(embedding) + self.pos_enc(time)
        if stat_emb is None:
            cls_token = self.cls_learnable.expand(B, self.hidden_dim)
        else:
            cls_token = stat_emb

        tokens = torch.cat(
            [cls_token.unsqueeze(1).expand(B, 1, -1), embedding], dim=1
        )  # B x L + 1 x C

        # Add cls token mask
        mask = torch.cat([mask.new_zeros(B, 1), mask], dim=1)

        embedding = self.transformer(tokens, src_key_padding_mask=~mask)
        return embedding[:, 0]

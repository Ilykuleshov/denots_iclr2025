"""Module with the Forecasting Roformer class."""

import torch
from torch import Tensor, nn
from torchtune.modules import RotaryPositionalEmbeddings

from .base import ForecastingBackbone


class RoFormer(ForecastingBackbone):
    """Forecasting transformer with Rotary Positional Embeddings."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        nhead: int,
        num_layers: int,
    ):
        """Initialize the RoFormer.

        Args:
        ----
            input_dim (int): Input dimension.
            hidden_dim (int): Hidden dimension.
            nhead (int): Number of attention heads.
            num_layers (int): Number of layers.

        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.num_layers = num_layers

        self.projector = nn.Linear(input_dim, hidden_dim)
        self.hdim = self.hidden_dim // self.nhead
        self.pos_enc = RotaryPositionalEmbeddings(self.hdim, 1000)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=self.nhead,
            batch_first=True,
            activation=nn.GELU(),
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, self.num_layers)

        self.transformer.compile()
        self.mask_token = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, t: Tensor, x: Tensor, t_eval: Tensor):
        """Predict the values at given t_eval.

        This sorts t and t_eval, masks the missing values and predicts them
        using the Transformer.
        """
        x = self.projector(x)

        B, S, H = x.shape
        T = t_eval.shape[1]

        t_all = torch.cat([t, t_eval], dim=1)
        x_eval_dummy = self.mask_token.expand(B, T, H)
        x_all = torch.cat([x, x_eval_dummy], dim=1)

        sorter = torch.argsort(t_all, dim=1)[0]
        unsorter = torch.argsort(sorter)
        xsort = x_all.index_select(1, sorter)

        x_per_head = torch.stack(torch.split(xsort, self.hdim, dim=2), dim=2)
        encoding = self.pos_enc(x_per_head)
        encoding = encoding.reshape(B, T + S, H)

        zsort = self.transformer(encoding)
        z_all = zsort.index_select(1, unsorter)
        return z_all[:, -T:]

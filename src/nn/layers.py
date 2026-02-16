"""Module with our nn layers."""

import math

import torch
from torch import Tensor, nn


class NaNBatchNorm(nn.BatchNorm1d):
    """Modified version of nn.BatchNorm1d, able to handle NaNs.

    Expects 3D input, (B, L, 1) .
    """

    def forward(self, x: Tensor):
        """Forward pass of the modified batch norm."""
        # Handle static features
        mask = x.isfinite().any(-1)
        # Not enough non-nan values to normalize
        if mask.sum() <= 1:
            return x

        res = torch.full_like(x, fill_value=torch.nan)
        res[mask] = super().forward(x[mask])
        return res


class NaNEmbedding(nn.Embedding):
    """Modified version of nn.Embedding, able to handle NaNs (indicated by `-1`-s)."""

    def forward(self, x: Tensor):
        """Forward pass of the modified embedding."""
        mask = x != -1
        res = torch.full(
            (*x.shape, self.embedding_dim),
            torch.nan,
            dtype=torch.float,
            device=x.device,
        )
        res[mask] = super().forward(x[mask].int())
        return res


class NaNLinear(nn.Linear):
    """Modified version of nn.Linear, which propagates NaNs without messing up gradients."""

    def forward(self, x: Tensor):
        """Forward pass of the modified linear layer."""
        mask = x.isfinite()
        x = torch.where(mask, x, 0)
        x = super().forward(x)
        return torch.where(mask, x, torch.nan)


class ZeroAllNaN(nn.Module):
    """Zero out channels where all values are NaNs."""

    def __init__(self, dim: int):
        """Initialize the ZeroAllNaN layer.

        Args:
            dim (int): Dimension to zero out if all-NaN.

        """
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor):
        """Forward pass of the ZeroAllNaN layer."""
        return torch.where(x.isnan().all(dim=self.dim, keepdim=True), 0, x)


class Diff(nn.Module):
    """torch.diff as a layer."""

    def forward(self, x: Tensor):
        """Forward pass of the diff layer."""
        intensity = torch.zeros_like(x)
        intensity[:, 1:] = x.diff(dim=1)
        return intensity


class Unsqueeze(nn.Module):
    """Unsqueeze last dimension as a layer."""

    def forward(self, x: Tensor):
        """Unsqueeze last dimension."""
        return x.unsqueeze(-1)


class TimePositionalEncoding(nn.Module):
    """Temporal encoding in THP, ICML 2020. Taken from EasyTPP.

    https://github.com/Anonymous0006/EasyTPP/blob/31a83d2056bc95aa978e961078c3a4386df6ea4c/easy_tpp/model/tf_model/tf_baselayer.py#L258
    """

    def __init__(self, d_model, max_len=5000):
        """Initialize the TimePositionalEncoding layer.

        Args:
        ----
            d_model (int): dimension of the model
            max_len (int): maximum length of the sequence

        """
        super().__init__()

        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        """Compute time positional encoding defined in Equation (2) in THP model.

        Args:
            x (tensor): time_seqs, [batch_size, seq_len]

        Returns:
            temporal encoding vector, [batch_size, seq_len, model_dim]

        """
        length = x.size(1)

        return self.pe[:, :length]

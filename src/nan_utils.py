"""Module with some nan utilities."""

import torch
from torch import Tensor

from .mask_utils import maskbfill, maskffill


def fill_nan(t: Tensor, dim: int = 1):
    """Fill nan values using forward, then backward filling."""
    t = maskffill(t, ~t.isnan(), dim)
    t = maskbfill(t, ~t.isnan(), dim)
    return t


def zero_all_nan(t: Tensor, mask: Tensor, dim: int = 1):
    """Zero-out constantly-NaN/infinite components."""
    if mask.ndim < t.ndim:
        mask = mask.unsqueeze(-1)

    return torch.where((t.isfinite() & mask).any(dim, keepdim=True), t, 0)

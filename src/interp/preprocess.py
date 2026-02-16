"""File with utilities to handle missing values in interpolated sequences."""

from typing import Literal

import torch
from torch import Tensor

from ..mask_utils import maskbfill, maskffill, maskroll


def ffill_bfill_ends(x: Tensor, dim: int, strategy: Literal["zero", "last"]):
    """Fill the nans in the beginning/end with zeros or first/last values respectively."""
    first_nan_mask = torch.cummax(~x.isnan(), dim).values
    last_nan_mask = torch.cummax(~x.isnan().flip(dim), dim).values.flip(dim)

    match strategy:
        case "last":
            x = maskbfill(x, first_nan_mask, dim)
            x = maskffill(x, last_nan_mask, dim)
        case "zero":
            x = torch.where(first_nan_mask | last_nan_mask, x, 0)

    return x


def expand_roll_nans(
    t: Tensor,
    x: Tensor,
    mask: Tensor,
):
    """Handle missing values.

    This does the following:
    1. If there are NaNs, we zero-out the corresponding mask elements. This requires
    us to expand time and mask, adding a channel dimension, since NaNs are channel-independent.
    2. All elements of t,x,mask where the mask is zero are moved to the end for ease of use.
    3. All elements of t where mask is zero are set to +inf to prevent interpolating/integrating
    over them.

    Returns the modified times, xs and the resulting mask.
    """
    # Handle NaNs
    mask = mask.unsqueeze(-1).expand_as(x)
    t = t.unsqueeze(-1).expand_as(x)
    mask = mask & x.isfinite()

    # Move the values we wish to ignore to the end.
    mask, t, x = mask.mT, t.mT, x.mT
    mask, t, x = maskroll(mask, t, x)
    mask, t, x = mask.mT, t.mT, x.mT

    return t, x, mask

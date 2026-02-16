"""Module with a wrapper to define BaseInterpolator outside of interpolation bounds."""

from typing import Literal

import torch
from torch import Tensor

from .base import BaseInterpolator


class ForecastingWrapper(BaseInterpolator):
    """Wrapper to handle out-of-bounds values during interpolation."""

    def __init__(
        self, interp: BaseInterpolator, forecast_method: Literal["zero", "last"]
    ):
        """Initialize internal state."""
        super().__init__()
        self.interp = interp
        self.forecast_method = forecast_method

    def fit(self, t: Tensor, x: Tensor):
        """Fit underlying interp and save max times."""
        self.interp.fit(t, x)
        self.tmax = t.amax(dim=1)

    def forward(self, t: Tensor):
        """Interpolate, handling out-of-bounds values."""
        out_of_bounds = t > self.tmax
        t = torch.clip(t, max=self.tmax)
        x = self.interp(t)
        if out_of_bounds.ndim < x.ndim:
            out_of_bounds = out_of_bounds.unsqueeze(-1)

        match self.forecast_method:
            case "zero":
                return torch.where(out_of_bounds, 0, x)
            case "last":
                return x
            case _:
                raise ValueError()

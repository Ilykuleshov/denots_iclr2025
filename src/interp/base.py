"""The module with the base class for interpolators."""

from torch import Tensor, nn


class BaseInterpolator(nn.Module):
    """Base class for interpolators."""

    def fit(self, t: Tensor, x: Tensor):
        """Fit an interpolation to a given sequence of observations.

        Arguments:
        ---------
        t (Tensor):
            time tensor, shape (B, L, C) or (B, L).
        x (Tensor):
            observations tensor, shape (B, L, C).

        """
        raise NotImplementedError()

    def forward(self, t: Tensor) -> Tensor:
        """Interpolate observations for given time values.

        Arguments:
        ---------
        t (Tensor):
            time tensor, shape (B, C) or (B).

        """
        raise NotImplementedError()

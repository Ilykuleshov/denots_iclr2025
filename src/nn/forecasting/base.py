"""The file with the base class for forecasting backbones."""

from abc import ABC, abstractmethod
from typing import Literal

from torch import Tensor, nn


class ForecastingBackbone(nn.Module, ABC):
    """The base class for forecasting backbones."""

    nan_behaviour: Literal["keep", "fill", "zero"]
    input_dim: int
    hidden_dim: int

    @abstractmethod
    def forward(self, t: Tensor, x: Tensor, t_eval: Tensor):
        """Interpolate/extrapolate the given trajectory (t, x) to t_eval."""

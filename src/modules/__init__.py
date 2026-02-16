"""The module with all LightningModule-s."""

from .forecasting import ForecastingModule
from .supervised import SupervisedModule

__all__ = ["SupervisedModule", "ForecastingModule"]

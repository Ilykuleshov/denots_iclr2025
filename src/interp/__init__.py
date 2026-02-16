"""The module with all interpolation-related utilities.

Specifically, this includes the natural cubic spline utils and some preprocessing functions.
"""

from .natural_cubic import NaturalCubicSpline
from .preprocess import expand_roll_nans

__all__ = ["NaturalCubicSpline", "expand_roll_nans"]

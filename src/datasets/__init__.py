"""The module with all dataset-connected tools.

This includes Subclasses of the torch Dataset, as well as our attack augmentations.
"""

from .base import Dataset, IterableDataset

__all__ = ["Dataset", "IterableDataset"]

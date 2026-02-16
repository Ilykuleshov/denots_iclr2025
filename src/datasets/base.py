"""The module with base classes for datasets.

Each dataset returns a dictionary. This is done to separate polars and DataLoader preprocessing.
"""

from dataclasses import dataclass
from itertools import chain
from operator import itemgetter

import numpy as np
from torch.utils.data import Dataset as Dataset_
from torch.utils.data import IterableDataset as IterableDataset_


@dataclass(eq=False, kw_only=True)
class Dataset(Dataset_):
    """The map-style dataset base class.

    This implements three features, not present
    in torch.utils.data.Dataset:
        1. Ability to specify transforms, which are applied to
        each returned tensordict. This is used to specify attacks.
        2. A function to conveniently retrieve the class balance
        for use in WeightedRandomSampler.

    Utility attributes used:
        - transforms (list[Callable] | None, optional):
            List of transforms to apply.

    """

    data: list[dict]

    def __post_init__(self):
        """Handle default values."""
        self.transforms = getattr(self, "transforms", [])

    def __getitem__(self, index):
        """Return the item located at index, applying transforms."""
        row = self.data[index]
        for transform in self.transforms:
            row = transform(row)
        return row

    def __len__(self):
        """Return the length of the stored data."""
        return len(self.data)

    @property
    def all_transforms(self):
        """Retrieve all transforms used in dataset."""
        return self.transforms

    def balance(self) -> np.ndarray:
        """Calculate the counts of each sample's class, based on 'balance_col'."""
        col = np.fromiter(
            map(itemgetter("balance_col"), self.data),
            np.int32,
            count=len(self.data),
        )

        _, counts = np.unique(col, return_counts=True)
        balance_arr = counts[col.astype(np.int32)]
        return balance_arr


@dataclass(eq=False, kw_only=True)
class IterableDataset(IterableDataset_):
    """The iterable-style dataset base class.

    This implements one feature, not present
    in torch.utils.data.IterableDataset:
        1. Ability to specify transforms, which are applied to
        each yielded tensordict.

    Attributes
    ----------
        transforms (list[Callable], optional):
            List of function-style transforms to apply.
            Defaults to None for no transforms.
        iter_transforms (list[Callable], optional):
            List of generator-style transforms to apply to
            each element of data before yielding.
            Defaults to None for no iter_transforms.

    """

    data: list[dict]

    def __post_init__(self):
        """Handle default values."""
        self.transforms = getattr(self, "transforms", [])
        self.iter_transforms = getattr(self, "iter_transforms", [])

    def __iter__(self):
        """Iterate through self.data.

        For each self.data td, apply all self.iter_transforms
        consequtively (yielding from each of them into the next),
        after which apply the normal transforms.
        """
        generator = (d for d in self.data)
        for iter_transform in self.iter_transforms:
            nested_iter = map(iter_transform, generator)
            generator = chain.from_iterable(nested_iter)

        for transform in self.transforms:
            generator = map(transform, generator)
        yield from generator

    @property
    def all_transforms(self):
        """Retrieve all transforms used in dataset."""
        return self.transforms + self.iter_transforms

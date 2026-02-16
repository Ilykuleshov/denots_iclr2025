"""The module with all our attacks."""

import numpy as np
from funcy import first, is_list, omit, select_values


class BOSDropAttack:
    """The drop attack. Replaces a fraction of the sequence with NaNs."""

    def __init__(self, frac: float, size: float, offset: float):
        """Initialize the attack.

        Args:
            frac (float): The fraction to drop.
            size (float): The size of the candidate window.
            offset (float): The offset of the candidate window.

        """
        self.frac = frac
        self.offset = offset
        self.size = size

    def __call__(self, row: dict):
        """Apply the attack to the row.

        Args:
            row (dict): The row to attack.

        Returns:
            dict: The attacked row.

        """
        seq: dict = select_values(is_list, row)
        nonseq: dict = omit(row, seq.keys())
        L = len(first(seq.values()))
        start = int(L * self.offset)
        size = int(L * self.size)
        assert start + size <= L

        change = int(size * self.frac)
        for k, v in seq.items():
            if k == "time":
                continue
            attacked = np.random.choice(size, change, replace=False)
            for i in attacked:
                v[start + i] = float("nan")

        return seq | nonseq


class BOSChangeAttack:
    """The change attack. Replaces a fraction of the sequence with  Gaussian noise."""

    def __init__(self, frac: float, size: float, offset: float):
        """Initialize the attack.

        Args:
            frac (float): The fraction to change.
            size (float): The size of the candidate window.
            offset (float): The offset of the candidate window.

        """
        self.frac = frac
        self.size = size
        self.offset = offset

    def __call__(self, row: dict):
        """Apply the attack to the row.

        Args:
            row (dict): The row to attack.

        Returns:
            dict: The attacked row.

        """
        seq: dict = select_values(is_list, row)
        nonseq: dict = omit(row, seq.keys())
        L = len(first(seq.values()))
        start = int(L * self.offset)
        size = int(L * self.size)
        assert start + size <= L

        change = int(size * self.frac)
        for k, v in seq.items():
            if k == "time":
                continue
            attacked = np.random.choice(size, change, replace=False)
            for i in attacked:
                v[start + i] = np.random.randn()

        return seq | nonseq

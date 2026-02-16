"""The module with a DataModule class, compatible with our project."""

from functools import partial

import polars as pl
from funcy import first, walk_keys
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler

from src.datasets.base import Dataset, IterableDataset
from src.tensordict_utils import collate_pad_sequence

from .datasource import DataSource
from .typevars import Stage


class DataModule(LightningDataModule):
    """The DataModule class with some useful methods."""

    def __init__(
        self,
        datasource: DataSource,
        dataset_factories: dict[Stage, partial[Dataset | IterableDataset]],
        batch_size: int,
        batches_per_epoch: int | None = None,
        balance: bool = False,
        num_workers: int = 1,
    ):
        """Initialize the internal state.

        Args:
        ----
            datasource (DataSource): The datasource to use.
            dataset_factories (dict): The dataset factories to use.
            batch_size (int): The batch size to use.
            batches_per_epoch (int, optional): The number of batches per epoch. Defaults to None.
            balance (bool, optional): Whether to balance the dataset. Defaults to False.
            num_workers (int, optional): The number of workers to use. Defaults to 1.

        """
        super().__init__()
        self.datasource = datasource
        self.dataset_factories = dataset_factories
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.balance = balance
        self.num_workers = num_workers

        self.augmentations = {}
        self.datasets: dict[Stage, Dataset | IterableDataset] = {}

        if self.batches_per_epoch:
            self.samples_per_epoch = self.batches_per_epoch * self.batch_size
        else:
            self.samples_per_epoch = None

    def setup(self, stage: str) -> None:
        """Set up the datasets for different stages (fitting, validating, testing, or predicting).

        Reads a parquet file and filters it based on the "split" column.
        The resulting datasets are stored in the self.datasets dictionary.
        """
        df = self.datasource().collect()
        split_dict = df.partition_by("split", include_key=False, as_dict=True)
        split_dict: dict[str, pl.DataFrame] = walk_keys(first, split_dict)
        for k in self.dataset_factories:
            data = split_dict[k].rows(named=True)
            self.datasets[k] = self.dataset_factories[k](data=data)
            if k in self.augmentations:
                self.datasets[k].transforms.extend(self.augmentations[k])

    def _sample_dataloader(self, dataset: Dataset):
        if self.balance:
            if "balance_col" not in DataSource.labels(self.datasource.get_schema()):
                raise ValueError(
                    f"Balance col not present in {self.datasource} with balance=True."
                )

            if not self.samples_per_epoch:
                raise ValueError(
                    "You must specify the number of samples per epoch if balance = True!"
                )

            sampler = WeightedRandomSampler(
                1 / dataset.balance(),
                num_samples=self.samples_per_epoch,
                replacement=True,
            )
        elif self.samples_per_epoch:
            sampler = RandomSampler(
                dataset,
                replacement=self.samples_per_epoch > len(dataset),
                num_samples=self.samples_per_epoch,
            )
        else:
            sampler = RandomSampler(dataset)

        return DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            collate_fn=collate_pad_sequence,
            num_workers=self.num_workers,
        )

    def _iter_dataloader(self, dataset):
        if isinstance(dataset, IterableDataset):
            num_workers = 0  # see https://discuss.pytorch.org/t/dataloader-yields-copied-batches/200529/4
        else:
            num_workers = self.num_workers

        return DataLoader(
            dataset,
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=collate_pad_sequence,
            num_workers=num_workers,
        )

    def train_dataloader(self, predict=False):
        """Return the train dataloader."""
        if isinstance(self.datasets["train"], IterableDataset) or predict:
            return self._iter_dataloader(self.datasets["train"])

        return self._sample_dataloader(self.datasets["train"])

    def val_dataloader(self):
        """Return the val dataloader."""
        return self._iter_dataloader(self.datasets["val"])

    def test_dataloader(self):
        """Return the test dataloader."""
        return self._iter_dataloader(self.datasets["test"])

    def predict_dataloader(self):
        """Return a tuple of all dataloaders."""
        return (
            self.train_dataloader(),
            self.val_dataloader(),
            self.test_dataloader(),
        )

"""The experiment script for forecasting."""

import sys

import hydra
from hydra.utils import instantiate
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

sys.path.append("/app/")

from src.datamodule import DataModule
from src.datasets import Dataset
from src.datasource import DataSource
from src.experiment_management import init_experiment, train_val_test
from src.modules import ForecastingModule


def forecasting(config):
    """Run the forecasting pipeline for the specified config."""
    datasource = DataSource(f"/mnt/data/preprocessed/{config['datasource']}.parquet")
    datamodule = DataModule(
        datasource=datasource,
        dataset_factories={"train": Dataset, "val": Dataset, "test": Dataset},
        batch_size=config["batch_size"],
        batches_per_epoch=config["batches_per_epoch"],
        balance=False,
        num_workers=16,
    )

    schema = datasource.get_schema()
    features = datasource.features(schema)
    backbone = instantiate(config["backbone"], input_dim=len(features))

    module = ForecastingModule(backbone, features, lr=config["lr"])

    callbacks = [
        ModelCheckpoint(monitor="val_metric", mode="max"),
        EarlyStopping(
            monitor="val_metric",
            mode="max",
            patience=5,
            min_delta=1e-3,
            stopping_threshold=config.get("stopping_threshold"),
        ),
    ]

    with init_experiment(config, callbacks=callbacks) as trainer:
        val_metrics = train_val_test(
            cfg=config, module=module, datamodule=datamodule, trainer=trainer
        )
    if val_metrics:
        return val_metrics[0]["val_metric"]


if __name__ == "__main__":
    hydra.main(
        config_path="/app/config/forecasting",
        config_name="main.yaml",
        version_base=None,
    )(forecasting)()

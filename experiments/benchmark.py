"""The script for retrieving the number of parameters of the models."""

import sys
from pathlib import Path

import hydra
from omegaconf import read_write
import torch
from hydra.utils import instantiate
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

sys.path.append("/app/")

from src.datamodule import DataModule
from src.datasets import Dataset
from src.datasource import DataSource
from src.experiment_management import (
    init_experiment,
    log_torchinfo,
)
from src.modules import SupervisedModule
from src.nn.encoder import Encoder
from src.nn.supervised.base import SupervisedBackbone
from src.nn.supervised.ncde import TorchCDEBackbone


def init_supervised_modules(cfg):
    """Initialize module and datamodule for an experiment."""
    # All classes which require further arguments
    # are instantiated partially: passing arguments to instantiate directly will either
    # just convert the arguments to OmegaConf objects (producing difficult bugs)
    # or then re-instantiate them, copying and breaking gradients (even harder to catch).
    # Specific behavior depends on the _convert_ argument, and is suboptimal in any case.

    ds_path = Path(f"/mnt/data/preprocessed/{cfg['datasource']}.parquet")

    datasource = DataSource(ds_path)
    datamodule = DataModule(
        datasource=datasource,
        dataset_factories={"train": Dataset, "val": Dataset, "test": Dataset},
        batch_size=cfg["batch_size"],
        batches_per_epoch=cfg["batches_per_epoch"],
        balance=cfg["balance"],
        num_workers=32,
    )

    schema = datasource.get_schema()
    encoder = Encoder(
        schema=schema,
        emb_dim=cfg["encoder_dim"],
        time_emb=cfg.get("time_emb"),
        num_norm="TorchCDEBackbone" not in cfg["backbone"]["_target_"],
    )

    backbone: SupervisedBackbone = instantiate(cfg["backbone"])(
        input_dim=encoder.hidden_dim
    )

    module = SupervisedModule(
        schema=schema,
        encoder=encoder,
        backbone=backbone,
        label_type=datasource.label_type(schema, "target"),
        sequence_type=cfg["sequence_type"],
        lr=cfg["lr"],
    )
    return module, datamodule


def benchmark(config):
    """Get the number of parameters in the supervised pipeline for the specified config."""
    module, datamodule = init_supervised_modules(config)

    if not config.get("disable_early_stopping"):
        callbacks = [
            EarlyStopping(monitor="val_metric", mode="max"),
            ModelCheckpoint(monitor="val_metric", mode="max"),
        ]
    else:
        callbacks = []


    with read_write(config):
        config["experiment"] = config["experiment"] + "_benchmark"

    with init_experiment(config, callbacks=callbacks) as trainer:
        log_torchinfo(module, trainer.logger)

if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")

    hydra.main(
        config_path="/app/config/supervised",
        config_name="main.yaml",
        version_base=None,
    )(benchmark)()

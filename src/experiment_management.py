"""Small import module with miscellaneous code, useful for launching experiments."""

import logging
import os
import random
from ast import literal_eval
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import git
import mlflow
import pytorch_lightning as pl
import torch
import torchinfo
import yaml
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from .datamodule import DataModule
from .omegaconf_resolvers import register_path_resolvers

DEBUG: Final = literal_eval(os.environ.get("DEBUG", "False"))
register_path_resolvers()


def assert_commited(*paths):
    """Raise error if any of the indicated paths contain uncommited changes."""
    repo = git.Repo()
    modified_paths = repo.untracked_files
    modified_paths.extend([p.a_path or p.b_path for p in repo.index.diff(None)])
    paths = [Path(p) for p in paths]

    for p in modified_paths:
        p = Path(p)
        if any(parent in p.parents for parent in paths):
            raise RuntimeError("Commit or ignore changes before running!")


@contextmanager
def init_experiment(cfg, **trainer_kw_overrides):
    """Initialize an experiment, returning a Trainer instance."""
    if not DEBUG:
        random.seed()  # to have different mlflow run names
        pl_logger = MLFlowLogger(
            experiment_name=cfg["experiment"], log_model=cfg.get("log_model", True)
        )
        pl_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
        pl_logger.experiment.log_text(
            pl_logger.run_id, yaml.safe_dump(OmegaConf.to_container(cfg)), "config.yaml"
        )
        logger.info(f"Run ID: {pl_logger.run_id}")
    else:
        pl_logger = False
        # configure logging at the root level of Lightning
        logging.getLogger("pytorch_lightning").setLevel(logging.DEBUG)

    pl.seed_everything(cfg["seed"])
    trainer_kwargs: dict = instantiate(cfg.get("trainer_args", {}))
    trainer_kwargs.update(trainer_kw_overrides)
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        logger=pl_logger,
        log_every_n_steps=1,
        fast_dev_run=DEBUG,
        **trainer_kwargs,
    )

    if DEBUG:
        yield trainer
        return

    status = ""
    try:
        yield trainer
    except Exception as e:
        if isinstance(e, KeyboardInterrupt):
            status = "KILLED"
        else:
            status = "FAILED"

        pl_logger.experiment.log_text(pl_logger.run_id, repr(e), "error.txt")
        raise e
    else:
        status = "FINISHED"
    finally:
        mlflow.end_run(status)


def train_val_test(
    cfg, module: LightningModule, datamodule: DataModule, trainer: Trainer
):
    """Fit the module, load and save the best checkpoint and evaluate."""
    trainer.fit(module, datamodule)
    if not DEBUG and isinstance(trainer.checkpoint_callback, ModelCheckpoint):
        module.load_state_dict(
            torch.load(trainer.checkpoint_callback.best_model_path)["state_dict"]
        )

    if not DEBUG:
        val_metrics = trainer.validate(module, datamodule)
        if not cfg.get("skip_test"):
            trainer.test(module, datamodule)
        return val_metrics


def log_torchinfo(module: LightningModule, logger: MLFlowLogger | None):
    """Log model information."""
    summary = torchinfo.summary(module.backbone)
    if logger is not None:
        logger.experiment.log_text(logger.run_id, repr(summary), "model_summary.txt")
    else:
        print(summary)

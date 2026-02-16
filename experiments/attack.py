"""The experiment script for testing attacks."""

import sys
from pathlib import Path

import hydra
import mlflow
import polars as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, read_write
from pytorch_lightning import seed_everything

sys.path.append("/app/")

from experiments.supervised import init_supervised_modules
from src.experiment_management import init_experiment
from src.nn.supervised.ncde import TorchCDEBackbone

torch.set_float32_matmul_precision("medium")


def attack_supervised(config: DictConfig):
    """Run the supervised pipeline for the specified config."""
    attack = instantiate(config["attack"])

    run = mlflow.get_run(config["run_id"]).to_dictionary()
    artifact_uri = Path(run["info"]["artifact_uri"])
    cfg_path = artifact_uri / "config.yaml"
    ckpt_path = next(artifact_uri.rglob("*.ckpt"))
    cfg_loaded = OmegaConf.create(cfg_path.read_text())

    module, datamodule = init_supervised_modules(cfg_loaded)
    datamodule.augmentations["test"] = [attack]

    with read_write(config):
        config["experiment"] = f"attacks_{cfg_loaded['datasource']}"

    with init_experiment(config) as trainer:
        if isinstance(module.backbone, TorchCDEBackbone):
            data = datamodule.datasource().filter(split="test").collect()

            seed_everything(config["seed"])  # seed for attacks
            data = pl.DataFrame(map(attack, data.to_dicts()))

            module.backbone.populate_cache(
                data, [f for f in module.encoder.seq_enc.keys() if f != "time"]
            )
        trainer.test(module, datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    hydra.main(
        config_path="/app/config/attacks",
        config_name="main.yaml",
        version_base=None,
    )(attack_supervised)()

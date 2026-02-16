"""The supervised module."""

from typing import Literal

import torch
from pytorch_lightning import LightningModule
from tensordict import TensorDict
from torch import nn
from torchmetrics.classification import (
    BinaryAUROC,
    MulticlassAccuracy,
)
from torchmetrics.regression import R2Score

from ..nan_utils import fill_nan, zero_all_nan
from ..nn.encoder import Encoder
from ..nn.losses.supervised import RelaxedBCELogitLoss, RelaxedCrossEntropyLoss
from ..nn.supervised.base import SupervisedBackbone
from ..nn.supervised.ncde import TorchCDEBackbone


class SupervisedModule(LightningModule):
    """The supervised module."""

    def __init__(
        self,
        schema: dict,
        encoder: Encoder,
        backbone: SupervisedBackbone,
        label_type: Literal["binary", "numeric", "category"],
        sequence_type: Literal["evs", "cts"],
        lr: float = 1e-3,
    ):
        """Initialize the neural networks."""
        super().__init__()
        self.encoder = encoder
        self.backbone = backbone
        self.sequence_type = sequence_type
        self.lr = lr
        match label_type:
            case "binary":
                metric = BinaryAUROC()
                self.loss = RelaxedBCELogitLoss()
                self.head_module = nn.Sequential(
                    nn.Linear(self.backbone.hidden_dim, 1), nn.Flatten(0, -1)
                )
                self.act = nn.Sigmoid()

            case "category":
                self.num_types = schema["label"]["category"]["target"]
                metric = MulticlassAccuracy(num_classes=self.num_types, average="micro")
                self.loss = RelaxedCrossEntropyLoss()
                self.head_module = nn.Sequential(
                    nn.Linear(self.backbone.hidden_dim, self.num_types)
                )
                self.act = nn.Softmax(dim=1)

            case "numeric":
                metric = R2Score()
                self.loss = nn.MSELoss()

                self.head_module = nn.Sequential(
                    nn.Linear(self.backbone.hidden_dim, 1), nn.Flatten(0, -1)
                )
                self.act = nn.Identity()

        if encoder.stat_dim > 0:
            self.stat_proj = nn.Linear(encoder.stat_dim, backbone.hidden_dim)

        self.train_metric = metric.clone()
        self.val_metric = metric.clone()
        self.test_metric = metric.clone()

    def shared_step(self, stage, batch: TensorDict, *args, **kwargs):
        """Shared step for train/val/test."""
        mask = batch["mask"]
        if isinstance(self.backbone, TorchCDEBackbone):
            embedding = self.backbone(batch["index"].to(torch.int64))
        else:
            stat_emb, seq_emb = self.encoder(batch)

            if stat_emb is not None:
                stat_emb = self.stat_proj(stat_emb)

            seq_emb = zero_all_nan(seq_emb, mask, dim=1)
            seq_emb[~mask] = 0

            nan_behaviour = getattr(self.backbone, f"{self.sequence_type}_nan")
            match nan_behaviour:
                case "keep":
                    pass
                case "fill":
                    seq_emb = fill_nan(seq_emb, dim=1)
                case "zero":
                    seq_emb = seq_emb.nan_to_num(0)
            embedding = self.backbone(stat_emb, seq_emb, batch["time"], mask)

        pred = self.head_module(embedding)
        target = batch["target"]
        loss: torch.Tensor = self.loss(pred, target)

        pred = self.act(pred)
        metric = getattr(self, f"{stage}_metric")
        metric(pred, target)

        self.log(f"{stage}_loss", loss, prog_bar=stage == "train")
        self.log(f"{stage}_metric", metric, prog_bar=stage == "val")

        return loss

    def training_step(self, *args, **kwargs):
        """Run the training step."""
        return self.shared_step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs):
        """Run the validation step."""
        return self.shared_step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs):
        """Run the test step."""
        return self.shared_step("test", *args, **kwargs)

    def configure_optimizers(self):
        """Configure the optimizer."""
        return torch.optim.Adam(self.parameters(), self.lr)

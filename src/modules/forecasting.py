"""The module with the ForecastingModule."""

import torch
from pytorch_lightning import LightningModule
from tensordict import TensorDict
from torch import Tensor, nn
from torchmetrics import R2Score

from ..nn.forecasting.base import ForecastingBackbone


class ForecastingModule(LightningModule):
    """The module for training and testing forecasting backbones."""

    def __init__(self, backbone: ForecastingBackbone, features: list[str], lr: float):
        """Initialize the module.

        Args:
        ----
            backbone (ForecastingBackbone): the backbone to use.
            features (list[str]): the features to forecast.
            lr (float): the learning rate.

        """
        super().__init__()
        self.backbone = backbone
        self.features = features
        self.head = nn.Linear(backbone.hidden_dim, backbone.input_dim)
        self.loss = nn.MSELoss(reduction="none")
        metric = R2Score()
        self.train_metric = metric.clone()
        self.val_metric = metric.clone()
        self.test_metric = metric.clone()
        self.lr = lr

    def shared_step(self, stage, batch: TensorDict, *args, **kwargs):
        """Shared step for train/val/test."""
        x = torch.stack([batch.get(f) for f in self.features], dim=-1)
        t = batch["time"]
        S = x.shape[1] // 2
        metric = getattr(self, f"{stage}_metric")

        sample_t, sample_x, target_t, target_x = self.split_sample_target(x, t)

        pred_x = self(sample_t, sample_x, target_t)
        loss = self.loss(pred_x, target_x)
        metric(pred_x.flatten(), target_x.flatten())

        interp_loss = loss[:, : S // 2].mean()
        forecast_loss = loss[:, S // 2 :].mean()
        loss = (interp_loss + forecast_loss) / 2

        self.log(f"{stage}_metric", metric, prog_bar=True)

        return loss

    def forward(self, sample_t, sample_x, target_t):
        """Predict targets for given samples."""
        pred_z = self.backbone(sample_t, sample_x, target_t)
        pred_x = self.head(pred_z)
        return pred_x

    def split_sample_target(self, x: Tensor, t: Tensor):
        """Split observations into samples & targets."""
        S = x.shape[1] // 2

        sample_x = x[:, 0:S:2]
        sample_t = t[:, 0:S:2]

        target_x = torch.cat((x[:, 1:S:2], x[:, S:]), dim=1)
        target_t = torch.cat((t[:, 1:S:2], t[:, S:]), dim=1)
        return sample_t, sample_x, target_t, target_x

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
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "max", patience=3
        )

        return {"optimizer": optimizer, "scheduler": scheduler, "monitor": "val_metric"}

"""File with the Neural CDE implementation.

https://proceedings.neurips.cc/paper/2020/hash/4a5876b450b45371f6cfe5047ac8cd45-Abstract.html.
"""

import signatory
import torch
import torchcde
from torch import Tensor, nn

from ..supervised.ncde import CDEFunc
from .base import ForecastingBackbone
from .latentode import LatentODEDecoder


class TorchCDEBackbone(ForecastingBackbone):
    """Neural CDE implementation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        logsig: bool = False,
        logsig_depth: int = 3,
        logsig_chunks: int = 16,
    ):
        """Initialize the Neural CDE backbone."""
        super().__init__()

        self.logsig_depth = logsig_depth
        self.logsig_chunks = logsig_chunks
        self.logsig = logsig

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Time is added as a channel separately
        data_dim = input_dim + 1
        if logsig:
            data_dim = signatory.logsignature_channels(data_dim, logsig_depth)

        self.nn = CDEFunc(data_dim, hidden_dim)
        self.initial = nn.Linear(data_dim, hidden_dim)
        self.decoder = LatentODEDecoder(hidden_dim)

    def forward(self, t: Tensor, x: Tensor, t_eval: Tensor):
        """Forward pass of the Neural CDE backbone."""
        x = torch.cat([x, t.unsqueeze(-1)], dim=-1)
        coefs = torchcde.natural_cubic_coeffs(x)
        X = torchcde.CubicSpline(coefs)

        z0 = self.initial(X.evaluate(X.interval[0]))
        S = t.shape[1]
        I = (t_eval < t.amax(1, True)).sum(1)[0].item()  # noqa: E741

        interp_ranks = torch.cat([t[0], t_eval[0, :I]]).argsort().argsort()

        t_int = torch.zeros(I + 1, device=t.device, dtype=t.dtype)
        t_int[1:] = interp_ranks[-I:] * S / (I + S)

        z_interp = torchcde.cdeint(
            X=X,
            func=self.nn,
            z0=z0,
            t=t_int,
            adjoint=False,
        )[:, 1:]

        t_extrap = t_eval[:, I:] - t_eval[:, I - 1, None]
        z_extrap = self.decoder(z_interp[:, -1], t_extrap)

        return torch.cat([z_interp, z_extrap], dim=1)

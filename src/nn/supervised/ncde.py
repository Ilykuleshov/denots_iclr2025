"""File with the Neural CDE implementation.

https://proceedings.neurips.cc/paper/2020/hash/4a5876b450b45371f6cfe5047ac8cd45-Abstract.html.
"""

from math import ceil

import joblib
import polars as pl
import signatory
import torch
import torchcde
from loguru import logger
from polars import selectors as cs
from tensordict import make_tensordict
from torch import Tensor, nn
from tqdm import tqdm

from ...polars_utils import FLOATLIST, INTLIST
from ...tensordict_utils import cast
from .base import SupervisedBackbone


class CDEFunc(nn.Module):
    """Controlled Differential Equation's vector field."""

    def __init__(self, input_channels, hidden_channels):
        """Initialize the vector field."""
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.nn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, input_channels * hidden_channels),
            nn.Tanh(),
        )

    def forward(self, t, z: Tensor):
        """Forward pass of the vector field."""
        z = self.nn(z)
        z = z.reshape(z.size(0), self.hidden_channels, self.input_channels)
        return z


@joblib.Memory("/tmp/cache").cache
def _fit_splines(
    data: dict[list],
    seq_feats: list[str],
    logsig: bool,
    logsig_chunks: int,
    logsig_depth: int,
):
    logger.info("Populating cubic spline coefficients cache:")
    logger.info("Collating...")

    td = make_tensordict(data)
    td = cast(td, {torch.float64: torch.float, torch.int64: torch.int})

    seq_emb = torch.stack([td[f] for f in seq_feats], dim=-1)
    # Normalize to 0-1
    nanmean = seq_emb.nanmean((0, 1), True)
    nanstd = ((nanmean - seq_emb) ** 2).nanmean((0, 1), True).sqrt()
    seq_emb = (seq_emb - nanmean) / nanstd

    time = td["time"]
    B, L = time.shape

    # We do not add "observational intensity":
    # for some reason, it causes the model to diverge.
    seq_emb = torch.cat([seq_emb, time.unsqueeze(-1)], dim=-1)

    if logsig:
        w = int(ceil(L / logsig_chunks))
        seq_emb = torch.cat(
            [
                torchcde.logsig_windows(s, logsig_depth, w)
                for s in tqdm(seq_emb.split(256), desc="Calculating log signatures")
            ]
        )

    # Multiprocessing hangs sometimes, so I just added a pbar
    coefs_l = list(
        tqdm(
            map(torchcde.natural_cubic_coeffs, seq_emb.unbind(0)),
            total=B,
            desc="Fitting splines",
        )
    )

    coefs = torch.stack(coefs_l)
    logger.info("Caching done!")
    return coefs


class TorchCDEBackbone(SupervisedBackbone):
    """Neural CDE implementation."""

    evs_nan = "keep"
    cts_nan = "keep"

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
        data_dim = input_dim
        if logsig:
            data_dim = signatory.logsignature_channels(data_dim, logsig_depth)

        self.nn = CDEFunc(data_dim, hidden_dim)
        self.initial = nn.Linear(data_dim, hidden_dim)

    def populate_cache(self, data: pl.DataFrame, seq_feats: list[str]):
        """Populate the cache with the spline coefficients."""
        scols = data.select(cs.by_dtype(FLOATLIST, INTLIST)).columns

        # Pad sequential values with nans
        data = data.with_columns(
            pl.col(scols)
            .list.gather(pl.int_range(pl.col(scols).list.len().max()), null_on_oob=True)
            .list.eval(pl.element().fill_null(float("nan")))
        )

        coefs = _fit_splines(
            data.to_dict(as_series=False),
            seq_feats,
            self.logsig,
            self.logsig_chunks,
            self.logsig_depth,
        )

        all_coefs = torch.empty(data["index"].max() + 1, *coefs.shape[1:])
        all_coefs[data["index"].to_list()] = coefs
        self.register_buffer("coefs", all_coefs, persistent=False)

    def forward(self, idx: Tensor):
        """Forward pass of the Neural CDE backbone."""
        coefs = self.coefs[idx]
        X = torchcde.CubicSpline(coefs)

        z0 = self.initial(X.evaluate(X.interval[0]))

        zT = torchcde.cdeint(
            X=X,
            func=self.nn,
            z0=z0,
            t=X.interval,
            adjoint=False,
            atol=1e-3,
            rtol=1e-3,
        )[:, 1]

        return zT

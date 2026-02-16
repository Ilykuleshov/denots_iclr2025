"""Module with the Encoder class."""

import copy
from typing import Literal

import torch
from loguru import logger
from tensordict import TensorDict
from torch import nn

from .layers import Diff, NaNBatchNorm, NaNEmbedding, NaNLinear, Unsqueeze


class Encoder(nn.Module):
    """The time series Encoder.

    This applies batch norm, concatenates time, inflates the dimension if necessary
    """

    def __init__(
        self,
        schema: dict,
        emb_dim: int,
        time_emb: Literal["diff", "cat"] | None = "diff",
        num_norm: bool = True,
    ):
        """Initialize the Encoder.

        Args:
        ----
            schema (dict): The schema of the data.
            emb_dim (int): The inflation embedding dimension.
            time_emb (str): The time embedding type.
            num_norm (bool): Whether to apply batch norm.

        """
        super().__init__()
        self.schema = schema
        self.emb_dim = emb_dim
        self.time_emb = time_emb
        self.num_norm = num_norm
        features: dict[str, dict | list] = copy.deepcopy(schema)
        labels: dict[str, list | dict] = features.pop("label")

        # Remove non-feature labels
        for coll in labels.values():
            for v in ["target", "balance_col"]:
                if v in coll:
                    if isinstance(coll, list):
                        coll.remove(v)
                    else:
                        coll.pop(v)

        self.seq_enc = nn.ModuleDict()
        self.stat_enc = nn.ModuleDict()

        # Categoric features
        for k, v in features["category"].items():
            self.seq_enc[k] = NaNEmbedding(v, emb_dim, 0)

        for k, v in labels["category"].items():
            self.stat_enc[k] = NaNEmbedding(v, emb_dim)

        for k in labels["binary"]:
            self.stat_enc[k] = NaNEmbedding(2, emb_dim)

        # Numeric features
        for k in features["numeric"]:
            self.seq_enc[k] = nn.Sequential(Unsqueeze())
            if num_norm:
                self.seq_enc[k].append(NaNBatchNorm(1))
            if emb_dim > 1:
                self.seq_enc[k].append(NaNLinear(1, emb_dim))

        for k in labels["numeric"]:
            self.stat_enc[k] = nn.Sequential(Unsqueeze())
            if num_norm:
                self.stat_enc[k].append(nn.BatchNorm1d(1))
            if emb_dim > 1:
                self.stat_enc[k].append(NaNLinear(1, emb_dim))

        # Handle time embeddings
        if time_emb is not None:
            self.seq_enc["time"] = nn.Sequential(Unsqueeze())
            if time_emb == "diff":
                self.seq_enc["time"].append(Diff())
            if num_norm:
                self.seq_enc["time"].append(NaNBatchNorm(1))
            if emb_dim > 1:
                self.seq_enc["time"].append(NaNLinear(1, emb_dim))

        self.hidden_dim = len(self.seq_enc) * emb_dim
        self.stat_dim = len(self.stat_enc) * emb_dim
        assert "target" not in self.stat_enc

        logger.debug(f"Sequential columns to encode: {', '.join(self.seq_enc.keys())}.")
        logger.debug(f"Total sequential encoding dim: {self.hidden_dim}.")
        logger.debug(f"Static columns to encode: {', '.join(self.stat_enc.keys())}.")
        logger.debug(f"Total static encoding dim: {self.stat_dim}.")

    def forward(self, td: TensorDict):
        """Embed the input sequence."""
        # Static embedding
        stat_emb = []
        for k, v in self.stat_enc.items():
            stat_emb.append(v(td[k].nan_to_num(0)))

        # Sequential embeddings
        seq_emb = []
        for k, v in self.seq_enc.items():
            seq_emb.append(v(td[k]))

        seq_emb = torch.concatenate(seq_emb, dim=-1)
        if not stat_emb:
            return None, seq_emb

        return torch.concatenate(stat_emb, dim=-1), seq_emb

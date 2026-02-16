"""The file with the RNN backbone class."""

from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ...mask_utils import masklast
from .base import SupervisedBackbone


class RNNBackbone(SupervisedBackbone):
    """The class with a simple RNN-based backbone."""

    evs_nan = "fill"
    cts_nan = "fill"

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        rnn_type: Literal["lstm", "gru"],
        num_layers: int = 1,
        bidirectional: bool = True,
    ):
        """Initialize internal state.

        Args:
        ----
            input_dim (int): The input dimension.
            hidden_dim (int): The hidden dimension.
            rnn_type (str): The type of RNN.
            num_layers (int): The number of layers.
            bidirectional (bool): Whether to use bidirectional RNN.

        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        if self.rnn_type == "lstm":
            rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                batch_first=True,
                num_layers=self.num_layers,
                bidirectional=self.bidirectional,
            )
        elif self.rnn_type == "gru":
            rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                batch_first=True,
                num_layers=self.num_layers,
                bidirectional=self.bidirectional,
            )
        elif self.rnn_type == "elman":
            rnn = nn.RNN(
                input_size=input_dim,
                hidden_size=hidden_dim,
                batch_first=True,
                num_layers=self.num_layers,
                bidirectional=self.bidirectional,
            )
        else:
            raise ValueError

        self.rnn = torch.compile(rnn)

    def forward(
        self, stat_emb: Tensor | None, embedding: Tensor, time: Tensor, mask: Tensor
    ):
        """Run encoder embeddings through the rnn."""
        if stat_emb is None:
            stat_emb = torch.zeros(
                embedding.shape[0],
                self.hidden_dim,
                device=embedding.device,
                dtype=embedding.dtype,
            )

        packed_emb = pack_padded_sequence(
            embedding,
            mask.sum(1).detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_backbone_embedding = self.rnn(packed_emb, stat_emb.unsqueeze(0))[0]
        embedding = pad_packed_sequence(packed_backbone_embedding, batch_first=True)[0]

        return masklast(embedding, mask, dim=1)

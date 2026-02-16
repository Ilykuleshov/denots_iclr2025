"""Module with all the considered dynamics vector fields."""

from abc import ABC, abstractmethod

from torch import Tensor, nn

from ...interp.base import BaseInterpolator


class InterpVFBase(nn.Module, ABC):
    """Base class for interpolator-based VFs."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        interp: BaseInterpolator,
        nf: bool,
    ):
        """Initialize the interpolator-based VF.

        Args:
        ----
            input_size (int): Input size.
            hidden_size (int): Hidden size.
            interp (BaseInterpolator): Interpolator.
            nf (bool): Whether to use negative feedback.

        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.interp = interp
        self.nf = nf
        self._setup_network()

    @abstractmethod
    def _setup_network(self):
        """Set up the dynamics network."""

    @abstractmethod
    def forward(self, t: Tensor, h: Tensor):
        """Forward pass of the interpolator-based VF."""


class StrictGRUInterpVF(InterpVFBase):
    """Strict GRU interpolator-based VF."""

    def _setup_network(self):
        """Initialize GRU."""
        self.net = nn.GRUCell(self.input_size, self.hidden_size)

    def forward(self, t: Tensor, h: Tensor):
        """Forward pass of the interpolator-based VF."""
        x = self.interp(t)
        if self.nf:
            return self.net(x, h) - h
        else:
            return self.net(x, h)


class AdaptiveGRUInterpVF(InterpVFBase):
    """GRU interpolator-based VF (adaptive version)."""

    def _setup_network(self):
        """Initialize GRU."""
        self.net = nn.GRUCell(self.input_size, self.hidden_size)

    def forward(self, t: Tensor, h: Tensor):
        """Forward pass of the interpolator-based VF."""
        x = self.interp(t)
        if self.nf:
            return self.net(x, -h)
        else:
            return self.net(x, h)


class ReLUInterpVF(InterpVFBase):
    """ReLU-based unlimited VF without NF."""

    def _setup_network(self):
        """Initialize ReLU MLP."""
        self.xin = nn.Linear(self.input_size, self.hidden_size)
        self.hin = nn.Linear(self.hidden_size, self.hidden_size)
        self.out = nn.Sequential(
            nn.ReLU(), nn.Linear(self.hidden_size, self.hidden_size)
        )

    def forward(self, t: Tensor, h: Tensor):
        """Forward pass of the interpolator-based VF."""
        x = self.interp(t)
        return self.out(self.xin(x) + self.hin(h))


class TanhInterpVF(InterpVFBase):
    """Tanh-based unlimited VF without NF."""

    def _setup_network(self):
        """Initialize Tanh MLP."""
        self.xin = nn.Linear(self.input_size, self.hidden_size)
        self.hin = nn.Linear(self.hidden_size, self.hidden_size)
        self.out = nn.Sequential(
            nn.Tanh(), nn.Linear(self.hidden_size, self.hidden_size), nn.Tanh()
        )

    def forward(self, t: Tensor, h: Tensor):
        """Forward pass of the interpolator-based VF."""
        x = self.interp(t)
        return self.out(self.xin(x) + self.hin(h))

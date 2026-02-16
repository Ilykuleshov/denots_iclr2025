"""The file with the LatentODE forecasting model and its encoder/decoder.

Inspired by https://arxiv.org/abs/1907.03907.
"""

import torch
import torchode as to
from torch import Tensor, nn

from .base import ForecastingBackbone


class ODEFunc(nn.Module):
    """The ODE vector function.

    The underlying NN is a Perceptron with 1 hidden layer and ReLU activation.
    """

    def __init__(self, dim: int):
        """Initialize the ODE vector function.

        Args:
        ----
            dim (int): The dimension of the input and output.

        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t, x):
        """Run the forward pass."""
        return self.net(x)


class LatentODEEncoder(nn.Module):
    """The LatentODE encoder.

    This module runs an RNN-ODE through
    """

    def __init__(self, input_dim: int, hidden_dim: int, tol: float = 1e-3):
        """Initialize the LatentODE encoder.

        Args:
        ----
            input_dim (int): The dimension of the input.
            hidden_dim (int): The dimension of the hidden state.
            tol (float): The tolerance for the ODE solver.

        """
        super().__init__()
        self.rnn = nn.GRUCell(input_dim, hidden_dim)

        self.vf = ODEFunc(hidden_dim)
        self.term = to.ODETerm(self.vf, with_stats=False)
        self.stepper = to.Dopri5(term=self.term)
        self.controller = to.IntegralController(atol=tol, rtol=tol, term=self.term)
        self.solver = to.AutoDiffAdjoint(
            self.stepper, self.controller, backprop_through_step_size_control=False
        )

    def forward(self, t: Tensor, x: Tensor):
        """Run the forward pass, compressing a sequence into a vector."""
        x = x.flip(1)
        t = t.flip(1)

        # Traverse backwards
        h = self.rnn(x[:, 0])
        for i in range(1, x.size(1)):
            ivp = to.InitialValueProblem(h, t[:, i - 1], t[:, i])
            sol = self.solver.solve(ivp, term=self.term)
            h = sol.ys[:, -1]
            h = self.rnn(x[:, i], h)

        return h


class LatentODEDecoder(nn.Module):
    """The LatentODE decoder.

    This module launches Neural ODE integration from the given embedding,
    evaluating it at given timepoints.
    """

    def __init__(self, hidden_dim: int, tol: float = 1e-3):
        """Initialize the decoder.

        Args:
        ----
            hidden_dim (int): The dimension of the hidden state.
            tol (float): The tolerance for the ODE solver.

        """
        super().__init__()
        self.vf = ODEFunc(hidden_dim)
        self.term = to.ODETerm(self.vf, with_stats=False)
        self.stepper = to.Dopri5(term=self.term)
        self.controller = to.IntegralController(atol=tol, rtol=tol, term=self.term)
        self.solver = to.AutoDiffAdjoint(
            self.stepper, self.controller, backprop_through_step_size_control=False
        )

    def forward(self, h: Tensor, t_eval: Tensor):
        """Run the forward pass, decoding a vector into a sequence."""
        B = h.size(0)
        t_start = torch.zeros(B, device=h.device, dtype=h.dtype)
        ivp = to.InitialValueProblem(h, t_start=t_start, t_eval=t_eval)
        sol = self.solver.solve(ivp, term=self.term)
        return sol.ys


class LatentODE(ForecastingBackbone):
    """The LatentODE forecasting model.

    This model is a combination of the LatentODEEncoder and LatentODEDecoder.
    """

    def __init__(self, input_dim: int, hidden_dim: int, tol: float = 1e-3):
        """Initialize the model.

        Args:
        ----
            input_dim (int): The dimension of the input.
            hidden_dim (int): The dimension of the hidden state.
            tol (float): The tolerance for the ODE solver.

        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.tol = tol

        self.encoder = LatentODEEncoder(input_dim, hidden_dim, tol)
        self.decoder = LatentODEDecoder(hidden_dim, tol)

    def forward(self, t, x, t_eval):
        """Run the forward pass."""
        h = self.encoder(t, x)
        z_eval = self.decoder(h, t_eval)
        return z_eval

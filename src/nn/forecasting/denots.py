"""File with the DeNOTS backbone class."""

from typing import Literal

import torchode as to
from torch import Tensor, nn

from ...interp.base import BaseInterpolator
from ...interp.forecasting import ForecastingWrapper
from ..vfs import INTERP_VFS
from ..vfs.interprnn import InterpVFBase
from .base import ForecastingBackbone


class DeNOTS(ForecastingBackbone):
    """The DeNOTS backbone."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        interp: BaseInterpolator,
        vf_type: Literal["gru", "strict"],
        forecast_method: Literal["zero", "last"],
        nf: bool = True,
        depth: float = 10,
        tol: float = 1e-3,
    ):
        """Initialize the DeNOTS backbone.

        Args:
        ----
            input_dim (int): The input dimension.
            hidden_dim (int): The hidden dimension.
            interp (BaseInterpolator): The interpolator.
            vf_type (str): The type of vector field (gru or strict).
            forecast_method (str): The forecasting method.
            nf (bool): Whether to use negative feedback.
            depth (float): The "depth" of the ODE.
            tol (float): The tolerance of the ODE solver.

        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.interp = ForecastingWrapper(interp, forecast_method)
        self.vf_type = vf_type
        self.nf = nf
        self.depth = depth
        self.tol = tol

        self.start_proj = nn.Linear(input_dim, hidden_dim)

        vf_cls: type[InterpVFBase] = INTERP_VFS[self.vf_type]
        self.f = vf_cls(input_dim, hidden_dim, interp=self.interp, nf=self.nf)
        self.term = to.ODETerm(self.f)
        self.stepper = to.Dopri5(self.term)
        self.controller = to.IntegralController(self.tol, self.tol, term=self.term)
        self.solver = to.AutoDiffAdjoint(
            self.stepper,
            self.controller,
            backprop_through_step_size_control=False,
        )

    def forward(self, t: Tensor, x: Tensor, t_eval: Tensor):
        """Integrate the underlying ODE & apply post-processing."""
        # scale T to accomodate for depth
        t = t * self.depth
        t_eval = t_eval * self.depth

        self.interp.fit(t, x)
        y0 = self.start_proj(x[:, 0])
        ivp = to.InitialValueProblem(y0=y0, t_start=t[:, 0], t_eval=t_eval)  # type: ignore
        solution: to.Solution = self.solver.solve(ivp, term=self.term)
        return solution.ys

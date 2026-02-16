"""File with the DeNOTS backbone class."""

from typing import Literal

import torch
import torchode as to
from torch import Tensor

from src.mask_utils import maskmax

from ...interp.base import BaseInterpolator
from ...interp.preprocess import expand_roll_nans, ffill_bfill_ends
from ..vfs import INTERP_VFS
from ..vfs.interprnn import InterpVFBase
from .base import SupervisedBackbone


class DeNOTS(SupervisedBackbone):
    """The DeNOTS backbone."""

    evs_nan = "zero"
    cts_nan = "keep"

    def __init__(
        self,
        input_dim,
        hidden_dim: int,
        interp: BaseInterpolator,
        vf_type: Literal["adaptive", "strict"],
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
            vf_type (str): The type of vector field (adaptive or strict).
            nf (bool): Whether to use negative feedback.
            depth (float): The "depth" of the ODE.
            tol (float): The tolerance of the ODE solver.

        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.interp = interp
        self.vf_type = vf_type
        self.nf = nf
        self.depth = depth
        self.tol = tol

        vf_cls: type[InterpVFBase] = INTERP_VFS[self.vf_type]
        self.f = vf_cls(input_dim, self.hidden_dim, interp=self.interp, nf=self.nf)
        self.term = to.ODETerm(self.f)
        self.stepper = to.Dopri5(self.term)
        self.controller = to.IntegralController(self.tol, self.tol, term=self.term)
        self.solver = to.AutoDiffAdjoint(
            self.stepper,
            self.controller,
            backprop_through_step_size_control=False,
        )

    def forward(
        self, stat_emb: Tensor | None, embedding: Tensor, time: Tensor, mask: Tensor
    ):
        """Integrate the underlying ODE & apply post-processing."""
        B = embedding.shape[0]

        if stat_emb is None:
            stat_emb = torch.zeros(
                B, self.hidden_dim, device=embedding.device, dtype=embedding.dtype
            )

        # scale T to accomodate for depth
        time = time * self.depth

        # Find max time BEFORE NaN handling
        tmax = maskmax(time, mask, 1)

        # Handle nans
        if embedding[mask].isnan().any():
            # Fill ends to escape out-of-interval instability
            embedding = ffill_bfill_ends(embedding, 1, strategy="last")
            # Add channel dim to time, roll missing values to the end and mark as padding.
            time, embedding, mask = expand_roll_nans(time, embedding, mask)

        # make time infinite at padding
        # to not interpolate it.
        time[~mask] = torch.inf
        self.interp.fit(time, embedding)

        t_start = torch.zeros(B, dtype=time.dtype, device=time.device)
        ivp = to.InitialValueProblem(stat_emb, t_start=t_start, t_end=tmax)  # type: ignore
        solution: to.Solution = self.solver.solve(ivp, term=self.term)

        representation = solution.ys[:, -1]

        return representation

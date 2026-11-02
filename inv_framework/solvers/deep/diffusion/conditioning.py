"""solvers deep diffusion conditioning implementation."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch
from inv_framework.core.operator import ForwardOperator
from inv_framework.ct.noise import NoiseModel


class ConditioningMethod(ABC):
    """Abstract measurement-conditioning hook.

    Generic conditioning (DPS, RED, etc.) only needs the forward model and
    autograd, so we type against ForwardOperator. Methods that additionally
    require an adjoint (PiGDM, MCG with projection) should narrow the type
    in their own __init__ and assert ``isinstance(operator, LinearOperator)``.
    """

    def __init__(self,
                 operator: ForwardOperator,
                 noiser: Optional[NoiseModel] = None,
                 scale: float = 1.0):
        self.operator = operator
        self.noiser = noiser
        self.scale = scale

    @abstractmethod
    def apply(self,
              x_t: torch.Tensor,
              x_0_hat: torch.Tensor,
              measurement: torch.Tensor,
              x_prev: torch.Tensor,
              **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (x_t_conditioned, residual_norm).

        Parameters
        ----------
        x_t : the DDIM-stepped current iterate (will be modified).
        x_0_hat : Tweedie estimate of x_0 from the model.
        measurement : y.
        x_prev : the variable carrying autograd graph back to model inputs.
        """


class PosteriorSampling(ConditioningMethod):
    """DPS (Chung et al., 2023): subtract scaled gradient of ||y - A(x0_hat)||."""

    def apply(self, x_t, x_0_hat, measurement, x_prev, **kwargs):
        Ax = self.operator.forward(x_0_hat)
        diff = measurement - Ax
        residual = torch.linalg.norm(diff)
        grad, = torch.autograd.grad(residual, x_prev, retain_graph=False)
        return x_t - self.scale * grad, residual.detach()

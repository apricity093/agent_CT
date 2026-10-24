"""Quadratic Tikhonov regularization."""

from typing import Optional

import torch

from ..operators.base import LinearOperator
from .base import Regularizer, ProximalOperator


class TikhonovRegularizer(Regularizer, ProximalOperator):
    """The quadratic penalty ``0.5 * ||Lx||_2^2``.

    ``L`` defaults to the identity. Supplying a ``LinearOperator`` supports
    generalized Tikhonov regularization without changing the reconstruction
    operator interface.
    """

    def __init__(self, operator: Optional[LinearOperator] = None):
        if operator is not None and not isinstance(operator, LinearOperator):
            raise TypeError(
                "TikhonovRegularizer operator must be a LinearOperator or None; "
                f"got {type(operator).__name__}."
            )
        self.operator = operator

    def split_transform(self, shape):
        from ..operators.gradient import GradientOperator
        if self.operator is not None and not isinstance(self.operator, GradientOperator):
            raise ValueError('Only identity or GradientOperator Tikhonov is supported')
        return self.operator, 1. if self.operator is None else 8.

    def coefficient_value(self, z):
        return .5*z.square()

    def coefficient_proximal(self, z, t):
        return z/(1+t)

    @property
    def supports_box_proximal(self):
        return self.operator is None

    @property
    def has_full_proximal(self):
        return self.operator is None

    def proximal(self, x: torch.Tensor, step_size: float) -> torch.Tensor:
        """Identity L only; strength belongs to the composite objective."""
        import math

        if self.operator is not None:
            raise ValueError("Generalized Tikhonov has no implemented full proximal.")
        if not math.isfinite(float(step_size)) or step_size < 0:
            raise ValueError("proximal step_size must be finite and nonnegative")
        return x / (1.0 + step_size)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        transformed = x if self.operator is None else self.operator.forward(x)
        return 0.5 * transformed.reshape(transformed.shape[0], -1).square().sum(dim=1)

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``L^T L x`` (or ``x`` when ``L`` is the identity)."""
        if self.operator is None:
            return x
        return self.operator.adjoint(self.operator.forward(x))

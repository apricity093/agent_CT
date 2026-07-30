"""Quadratic Tikhonov regularization."""

from typing import Optional

import torch

from ..operators.base import LinearOperator
from .base import Regularizer


class TikhonovRegularizer(Regularizer):
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

    def value(self, x: torch.Tensor) -> torch.Tensor:
        transformed = x if self.operator is None else self.operator.forward(x)
        return 0.5 * transformed.reshape(transformed.shape[0], -1).square().sum(dim=1)

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``L^T L x`` (or ``x`` when ``L`` is the identity)."""
        if self.operator is None:
            return x
        return self.operator.adjoint(self.operator.forward(x))

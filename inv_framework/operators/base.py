"""Abstract forward operators for inverse problems y = A(x) + n.

Two layers of abstraction:

* `ForwardOperator` — the most general case. A may be non-linear (phase
  retrieval, blind deblurring, saturated sensing, MRI with non-Cartesian
  trajectories, etc.). The only requirement is that `forward` is
  autograd-traceable, so that gradient-based solvers (DIP, INR, DPS, RED,
  variational methods) can backprop through L = loss(A(x), y).

* `LinearOperator` — A is linear. Adds `adjoint` (transpose-vector product)
  and a default `pseudo_inverse`. Solvers that require an explicit adjoint
  (FBP, SIRT, Landweber, PiGDM, ...) take this subtype.

CT is a linear inverse problem, so the operators in `operators/ct/` inherit
from `LinearOperator`. Non-linear operators inherit from `ForwardOperator`
directly.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import torch


class ForwardOperator(ABC):
    """Abstract forward model A: x -> y, linear or non-linear.

    Attributes
    ----------
    domain_shape : tuple
        Per-sample shape of x (no batch dim), e.g. ``(C, H, W)``.
    range_shape : tuple
        Per-sample shape of y.
    """

    domain_shape: Tuple[int, ...]
    range_shape: Tuple[int, ...]

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply A to a batched x. Must support autograd through x."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class LinearOperator(ForwardOperator):
    """Linear forward operator with an explicit adjoint.

    The adjoint is the transpose of the implemented discrete forward map.
    Agreement with a custom backward alone does not establish this identity.
    Implementations must wire `forward` so that its autograd backward equals
    `adjoint` — this lets gradient-based solvers and explicit-adjoint solvers
    agree on what A^T is. For CT operators we achieve this with a custom
    `torch.autograd.Function`; see `operators/ct/radon_torch.py`.
    """

    @abstractmethod
    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Apply A^T to y. Must support batched input."""

    nonnegative_matrix = False

    def squared_norm_upper_bound(self, x):
        """Schur squared-norm bound, using nonnegative coefficients or small columns."""
        import math
        if self.nonnegative_matrix:
            rows = self.forward(torch.ones_like(x))
            cols = self.adjoint(torch.ones_like(rows))
        else:
            # Exact absolute sums, deliberately restricted to small generic test operators.
            n = math.prod(self.domain_shape)
            if n > 4096:
                raise ValueError('generic operator bound requires at most 4096 domain elements')
            rows = None
            cols = torch.zeros_like(x)
            for j in range(n):
                e = torch.zeros_like(x); e.flatten(1)[:,j] = 1
                column = self.forward(e).abs()
                rows = column if rows is None else rows+column
                cols.flatten(1)[:,j] = column.flatten(1).sum(1)
        return float(rows.abs().max())*float(cols.abs().max())

    def pseudo_inverse(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        """Default pseudo-inverse via Landweber iteration.

        Subclasses may override (e.g. CT operators expose FBP/SIRT).
        """
        from inv_framework.solvers.classical import landweber
        return landweber(self, y, **kwargs)

    def project(self, x: torch.Tensor, y: torch.Tensor,
                step_size: Optional[float] = None) -> torch.Tensor:
        """One Landweber-style data-consistency step toward Ax = y."""
        residual = self.forward(x) - y
        if step_size is None:
            step_size = 1.0
        return x - step_size * self.adjoint(residual)

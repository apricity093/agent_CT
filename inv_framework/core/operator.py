"""Linear operators, discrete derivatives, Haar transforms and norm estimation."""
from __future__ import annotations
from typing import Tuple, Optional
from abc import ABC, abstractmethod
import torch
import math


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
        from inv_framework.solvers.algebraic import landweber
        return landweber(self, y, **kwargs)

    def project(self, x: torch.Tensor, y: torch.Tensor,
                step_size: Optional[float] = None) -> torch.Tensor:
        """One Landweber-style data-consistency step toward Ax = y."""
        residual = self.forward(x) - y
        if step_size is None:
            step_size = 1.0
        return x - step_size * self.adjoint(residual)


class GradientOperator(LinearOperator):
    """D with stacked vertical/horizontal components at axis 1 (after batch)."""

    transform_kind = 'D'

    def __init__(self, domain_shape):
        self.domain_shape = tuple(domain_shape)
        if len(self.domain_shape) < 2 or any(d <= 0 for d in self.domain_shape):
            raise ValueError("GradientOperator requires positive spatial dimensions.")
        self.range_shape = (2, *self.domain_shape)

    def nullspace_basis(self,reference):
        leading=reference[0].numel()//(reference.shape[-2]*reference.shape[-1])
        basis=[]
        for index in range(leading):
            vector=torch.zeros_like(reference)
            vector.reshape(reference.shape[0],leading,*reference.shape[-2:])[:,index]=1
            basis.append(vector)
        return basis

    def forward(self, x):
        return torch.stack(gradient(x), dim=1)

    def adjoint(self, y):
        return gradient_adjoint(y[:, 0], y[:, 1])

    def pseudo_inverse(self, y, **kwargs):
        raise NotImplementedError("GradientOperator has no configured pseudo-inverse.")


def gradient(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return vertical and horizontal differences with zero terminal entries."""
    vertical = torch.zeros_like(x)
    horizontal = torch.zeros_like(x)
    vertical[..., :-1, :] = x[..., 1:, :] - x[..., :-1, :]
    horizontal[..., :, :-1] = x[..., :, 1:] - x[..., :, :-1]
    return vertical, horizontal


def gradient_adjoint(vertical: torch.Tensor, horizontal: torch.Tensor) -> torch.Tensor:
    """Apply D* to an arbitrary pair of equally shaped dual tensors."""
    if vertical.shape != horizontal.shape:
        raise ValueError("Gradient dual components must have the same shape.")
    out = torch.zeros_like(vertical)
    out[..., :-1, :] -= vertical[..., :-1, :]
    out[..., 1:, :] += vertical[..., :-1, :]
    out[..., :, :-1] -= horizontal[..., :, :-1]
    out[..., :, 1:] += horizontal[..., :, :-1]
    return out


class AugmentedLeastSquaresOperator(LinearOperator):
    def __init__(self, operator, fidelity, strength, regularizer):
        pass
        self.operator, self.fidelity = operator, fidelity
        self.domain_shape = operator.domain_shape
        self.data_size = math.prod(operator.range_shape)
        self.transform = GradientOperator(self.domain_shape) if regularizer == 'h1' else None
        self.strength = strength
        self.extra_size = (math.prod(self.transform.range_shape) if self.transform else math.prod(self.domain_shape)) if strength else 0
        self.range_shape = (self.data_size+self.extra_size,)

    def forward(self, x):
        data = self.operator.forward(x)*self.fidelity.weights.sqrt()
        if not self.strength: return data.flatten(1)
        extra = self.transform.forward(x) if self.transform else x
        return torch.cat((data.flatten(1), math.sqrt(self.strength)*extra.flatten(1)), 1)

    def adjoint(self, y):
        data = y[:, :self.data_size].reshape(y.shape[0], *self.operator.range_shape)
        x = self.operator.adjoint(data*self.fidelity.weights.sqrt())
        if self.strength:
            shape = self.transform.range_shape if self.transform else self.domain_shape
            extra = y[:, self.data_size:].reshape(y.shape[0], *shape)
            x = x+math.sqrt(self.strength)*(self.transform.adjoint(extra) if self.transform else extra)
        return x

    def rhs(self, measurement):
        data = (measurement*self.fidelity.weights.sqrt()).flatten(1)
        return torch.cat((data, data.new_zeros((data.shape[0], self.extra_size))), 1)


class WeightedLinearOperator(LinearOperator):
    def __init__(self,operator,weights):
        if tuple(weights.shape[1:])!=tuple(operator.range_shape):
            raise ValueError('weights must match the batched operator range')
        if not weights.is_floating_point() or not bool(torch.isfinite(weights).all()) or bool((weights<0).any()):
            raise ValueError('weights must be finite nonnegative floating values')
        self.operator=operator;self.sqrt_weights=weights.sqrt()
        self.domain_shape=operator.domain_shape;self.range_shape=operator.range_shape
    @property
    def nonnegative_matrix(self):return self.operator.nonnegative_matrix
    def whiten(self,value):
        return self.sqrt_weights*value
    def forward(self,x):return self.whiten(self.operator.forward(x))
    def adjoint(self,y):return self.operator.adjoint(self.whiten(y))
    def squared_norm_upper_bound(self,x):
        return float(self.sqrt_weights.square().max())*self.operator.squared_norm_upper_bound(x)
    def subset(self, indices):
        from inv_framework.solvers.subset import make_subset_operator
        return WeightedLinearOperator(make_subset_operator(self.operator, indices),
            self.sqrt_weights.square().index_select(-2, indices.to(self.sqrt_weights.device)))


def validate_haar(shape, level):
    if isinstance(level, bool) or not isinstance(level, int) or level <= 0:
        raise ValueError("Haar level must be a positive integer")
    if len(shape) < 2 or any(d <= 0 or d % (2 ** level) for d in shape[-2:]):
        raise ValueError("Haar H/W must be positive and divisible by 2**level")


class HaarOperator(LinearOperator):
    """Square orthogonal transform preserving leading dimensions and dtype/device."""

    transform_kind = 'W'

    def __init__(self, domain_shape, level=2):
        validate_haar(domain_shape, level)
        self.domain_shape = tuple(domain_shape)
        self.range_shape = self.domain_shape
        self.level = level

    def forward(self, x):
        out = x.clone()
        h, w = x.shape[-2:]
        s = math.sqrt(2.0)
        for _ in range(self.level):
            block = out[..., :h, :w].clone()
            low = (block[..., 0::2, :] + block[..., 1::2, :]) / s
            high = (block[..., 0::2, :] - block[..., 1::2, :]) / s
            out[..., :h//2, :w//2] = (low[..., 0::2] + low[..., 1::2]) / s
            out[..., :h//2, w//2:w] = (low[..., 0::2] - low[..., 1::2]) / s
            out[..., h//2:h, :w//2] = (high[..., 0::2] + high[..., 1::2]) / s
            out[..., h//2:h, w//2:w] = (high[..., 0::2] - high[..., 1::2]) / s
            h, w = h//2, w//2
        return out

    def adjoint(self, y):
        out = y.clone()
        s = math.sqrt(2.0)
        for depth in range(self.level - 1, -1, -1):
            h, w = y.shape[-2] // (2**depth), y.shape[-1] // (2**depth)
            block = out[..., :h, :w].clone()
            ll, lh = block[..., :h//2, :w//2], block[..., :h//2, w//2:w]
            hl, hh = block[..., h//2:h, :w//2], block[..., h//2:h, w//2:w]
            low, high = torch.empty_like(block[..., :h//2, :]), torch.empty_like(block[..., :h//2, :])
            low[..., 0::2], low[..., 1::2] = (ll + lh)/s, (ll - lh)/s
            high[..., 0::2], high[..., 1::2] = (hl + hh)/s, (hl - hh)/s
            out[..., :h:2, :w], out[..., 1:h:2, :w] = (low + high)/s, (low - high)/s
        return out

    def pseudo_inverse(self, y, **kwargs):
        return self.adjoint(y)

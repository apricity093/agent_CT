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
    """Spatial differences with component axis 1, independent batch/channels.

    Defaults: final two axes, unit spacing, zero terminal entries. Explicit
    axes allow 3D differences; periodic boundaries remain a distinct choice.
    """

    transform_kind = 'D'

    def __init__(self, domain_shape, axes=(-2, -1), spacing=None, boundary='neumann'):
        from inv_framework.core.array_math import Differences
        self.differences = Differences(axes, spacing, boundary)
        self.domain_shape = tuple(domain_shape)
        if len(self.domain_shape) < 2 or any(d <= 0 for d in self.domain_shape):
            raise ValueError("GradientOperator requires positive spatial dimensions.")
        self.range_shape = (len(axes), *self.domain_shape)

    def squared_norm_upper_bound(self, x):
        # Exact Cartesian Neumann spectrum; periodic differences use 4 sum h^-2.
        if self.differences.boundary == 'neumann':
            return sum(4*math.cos(math.pi/(2*x.shape[a]))**2/h**2
                       for a, h in zip(self.differences.axes, self.differences.spacing)
                       if x.shape[a] > 1)
        return self.differences.bound

    def nullspace_basis(self,reference):
        spatial = {a % reference.ndim for a in self.differences.axes}
        independent = [a for a in range(1, reference.ndim) if a not in spatial]
        import itertools
        basis = []
        for indices in itertools.product(*(range(reference.shape[a]) for a in independent)):
            vector = torch.zeros_like(reference)
            selector = [slice(None)]*reference.ndim
            for a, i in zip(independent, indices):
                selector[a] = i
            vector[tuple(selector)] = 1
            basis.append(vector)
        return basis

    def forward(self, x):
        return self.differences.forward(x).movedim(0, 1)

    def adjoint(self, y):
        return self.differences.adjoint(y.movedim(1, 0))

    def pseudo_inverse(self, y, **kwargs):
        raise NotImplementedError("GradientOperator has no configured pseudo-inverse.")


def gradient(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return vertical and horizontal differences with zero terminal entries."""
    from inv_framework.core.array_math import Differences
    return tuple(Differences((-2, -1)).forward(x))


def gradient_adjoint(vertical: torch.Tensor, horizontal: torch.Tensor) -> torch.Tensor:
    """Hermitian adjoint of zero-terminal unit-spacing 2D differences."""
    from inv_framework.core.array_math import Differences
    return Differences((-2, -1)).adjoint(torch.stack((vertical, horizontal)))


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
        if self.nonnegative_matrix or math.prod(self.domain_shape) <= 4096:
            return super().squared_norm_upper_bound(x)
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

    def squared_norm_upper_bound(self, x):
        return 1.

    def _haar(self, x):
        from inv_framework.core.array_math import OrthogonalHaar
        return OrthogonalHaar(x.shape, (-2, -1), self.level)

    def forward(self, x):
        return self._haar(x).forward(x)

    def adjoint(self, y):
        return self._haar(y).adjoint(y)

    def pseudo_inverse(self, y, **kwargs):
        return self.adjoint(y)

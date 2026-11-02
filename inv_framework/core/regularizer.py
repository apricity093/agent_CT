"""Regularizers and their primal, dual and joint proximal operators."""
from __future__ import annotations
from typing import Literal, Tuple, Optional
from inv_framework.core.operator import gradient, gradient_adjoint, LinearOperator, HaarOperator, validate_haar
from abc import ABC, abstractmethod
import torch
import math
import time


class Regularizer(ABC):
    """Penalty evaluated independently for each batch item."""

    has_full_proximal = False
    supports_box_proximal = False
    proximal_accuracy = 'closed_form'

    def validate_shape(self, shape):
        pass

    @abstractmethod
    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Return one regularization value per batch item."""


class ProximalOperator(ABC):
    """Penalty with an explicitly available proximal operator."""

    @abstractmethod
    def proximal(self, x: torch.Tensor, step_size: float) -> torch.Tensor:
        """Return ``prox_{step_size R}(x)`` without modifying ``x``."""


class TVRegularizer(Regularizer, ProximalOperator):
    """Isotropic or anisotropic TV over the final two tensor dimensions.

    The proximal map solves the ROF denoising subproblem with fast gradient
    projection on the dual problem. Batch and channel dimensions are kept
    independent; the implementation therefore works for ``(B, C, H, W)`` CT
    images and tensors with additional leading dimensions.
    """

    has_full_proximal = True
    proximal_accuracy = 'finite_FGP_no_gap_certificate'

    def __init__(
        self,
        mode: Literal["isotropic", "anisotropic"] = "isotropic",
        num_iterations: int = 50,
        tolerance: float = 1e-5,
    ):
        if mode not in ("isotropic", "anisotropic"):
            raise ValueError("TV mode must be 'isotropic' or 'anisotropic'.")
        if int(num_iterations) <= 0:
            raise ValueError("TV num_iterations must be positive.")
        if float(tolerance) < 0.0:
            raise ValueError("TV tolerance must be nonnegative.")
        self.mode = mode
        self.num_iterations = int(num_iterations)
        self.tolerance = float(tolerance)
        # Diagnostics only: detailed solvers read this value after each prox
        # call so early stopping inside the proximal map is accounted for
        # without changing the numerical update.
        self.last_prox_iterations = 0
        self.gradient_calls = self.gradient_adjoint_calls = 0

    def split_transform(self, shape):
        from inv_framework.core.operator import GradientOperator
        return GradientOperator(shape), 8.

    def coefficient_value(self, z):
        return z.square().sum(1).sqrt() if self.mode == 'isotropic' else z.abs()

    def coefficient_subgradient(self, z):
        """Minimum-norm coefficient subgradient, including zero differences."""
        if self.mode == 'anisotropic':
            return z.sign()
        return z / z.square().sum(1, keepdim=True).sqrt().clamp_min(torch.finfo(z.dtype).tiny)

    def coefficient_proximal(self, z, t):
        if self.mode == 'isotropic':
            return z*(1-t/z.square().sum(1,keepdim=True).sqrt().clamp_min(torch.finfo(z.dtype).tiny)).clamp_min(0)
        return z.sign()*(z.abs()-t).clamp_min(0)

    _gradient = staticmethod(gradient)
    _gradient_adjoint = staticmethod(gradient_adjoint)

    def _apply_gradient(self, x):
        self.gradient_calls += 1
        return self._gradient(x)

    def _apply_gradient_adjoint(self, vertical, horizontal):
        self.gradient_adjoint_calls += 1
        return self._gradient_adjoint(vertical, horizontal)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        vertical, horizontal = self._apply_gradient(x)
        if self.mode == "isotropic":
            density = torch.sqrt(vertical.square() + horizontal.square())
        else:
            density = vertical.abs() + horizontal.abs()
        return density.reshape(density.shape[0], -1).sum(dim=1)

    def proximal(self, x: torch.Tensor, step_size: float) -> torch.Tensor:
        self._validate_input(x)
        weight = float(step_size)
        if weight < 0.0:
            raise ValueError("TV proximal step_size must be nonnegative.")
        self.last_prox_iterations = 0
        if weight == 0.0:
            return x.clone()

        dual_v = torch.zeros_like(x)
        dual_h = torch.zeros_like(x)
        momentum_v = dual_v.clone()
        momentum_h = dual_h.clone()
        acceleration = 1.0
        previous = x

        # ||grad||^2 <= 8 for 2D forward differences, so 1 / (8 * weight)
        # is a valid dual gradient step after scaling the dual by weight.
        dual_step = 1.0 / (8.0 * weight)

        for iteration in range(1, self.num_iterations + 1):
            self.last_prox_iterations = iteration
            primal = x - weight * self._apply_gradient_adjoint(momentum_v, momentum_h)
            grad_v, grad_h = self._apply_gradient(primal)
            next_v = momentum_v + dual_step * grad_v
            next_h = momentum_h + dual_step * grad_h
            next_v, next_h = self._project_dual(next_v, next_h)

            next_acceleration = 0.5 * (1.0 + (1.0 + 4.0 * acceleration * acceleration) ** 0.5)
            momentum_scale = (acceleration - 1.0) / next_acceleration
            momentum_v = next_v + momentum_scale * (next_v - dual_v)
            momentum_h = next_h + momentum_scale * (next_h - dual_h)
            dual_v, dual_h = next_v, next_h
            acceleration = next_acceleration

            current = x - weight * self._apply_gradient_adjoint(dual_v, dual_h)
            if self.tolerance > 0.0:
                delta = (current - previous).reshape(current.shape[0], -1).norm(dim=1)
                scale = previous.reshape(previous.shape[0], -1).norm(dim=1).clamp_min(1.0)
                if bool(torch.all(delta <= self.tolerance * scale)):
                    return current
            previous = current

        return x - weight * self._apply_gradient_adjoint(dual_v, dual_h)

    def _project_dual(
        self,
        vertical: torch.Tensor,
        horizontal: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "anisotropic":
            return vertical.clamp(-1.0, 1.0), horizontal.clamp(-1.0, 1.0)
        norm = torch.sqrt(vertical.square() + horizontal.square()).clamp_min(1.0)
        return vertical / norm, horizontal / norm

    @staticmethod
    def _validate_input(x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"TV input must be a torch.Tensor; got {type(x).__name__}.")
        if x.ndim < 3:
            raise ValueError(
                "TV input must include a batch dimension and two spatial dimensions; "
                f"got shape {tuple(x.shape)}."
            )


class GradientHuberRegularizer(Regularizer):
    def __init__(self, delta=0.01):
        if not math.isfinite(float(delta)) or delta <= 0:
            raise ValueError("delta must be finite and positive")
        self.delta = float(delta)
        self.gradient_calls = self.gradient_adjoint_calls = 0

    def split_transform(self, shape):
        from inv_framework.core.operator import GradientOperator
        return GradientOperator(shape), 8.

    def coefficient_value(self, z):
        a = z.abs()
        return torch.where(a <= self.delta, z.square()/(2*self.delta), a-self.delta/2)

    def coefficient_proximal(self, v, t):
        if not math.isfinite(float(t)) or t < 0:
            raise ValueError("proximal step must be finite and nonnegative")
        return torch.where(v.abs() <= self.delta+t, self.delta*v/(self.delta+t), v-t*v.sign())

    def value(self, x):
        self.gradient_calls += 1
        z = torch.stack(gradient(x), dim=1)
        return self.coefficient_value(z).flatten(1).sum(1)

    def gradient(self, x):
        self.gradient_calls += 1
        self.gradient_adjoint_calls += 1
        v, h = gradient(x)
        return gradient_adjoint((v/self.delta).clamp(-1,1), (h/self.delta).clamp(-1,1))


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
        from inv_framework.core.operator import GradientOperator
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


class ProximalAccuracyError(RuntimeError):
    pass


def joint_prox(reg, value, weight, *, min_value=None, max_value=None,
               tolerance=1e-5, max_iterations=1000, iteration=1, check_budget=None):
    """Dual FGP for .5||x-v||² + weight R(Tx) + indicator_C(x).

    The primal minimizer for a dual vector p is P_C(v-T*p). The certified
    gap bounds distance to the exact prox by sqrt(2 gap). Requested distance
    errors decay as k^-2.1, with a summable iteration-weighted error sequence.
    Inner arithmetic uses float64; the returned dtype error is included.
    """
    from inv_framework.solvers._utils import apply_box_constraints
    transform, bound = reg.split_transform(value.shape[1:])
    def apply(point, adjoint=False):
        started=time.perf_counter()
        out=transform.adjoint(point) if adjoint else transform.forward(point)
        name=('synthesis_calls' if adjoint else 'analysis_calls') if type(reg).__name__=='HaarL1Regularizer' else ('gradient_adjoint_calls' if adjoint else 'gradient_calls')
        setattr(reg,name,getattr(reg,name,0)+1)
        if type(reg).__name__=='HaarL1Regularizer':
            reg.transform_seconds+=time.perf_counter()-started
        return out
    project = lambda x: apply_box_constraints(x, min_value=min_value, max_value=max_value)
    target = tolerance * max(float(value.norm()), 1.) / iteration**2.1
    v = value.double()
    p = torch.zeros_like(apply(v))
    momentum = p.clone()
    acceleration = 1.
    reg.last_prox_iterations = 0
    for inner in range(1, max_iterations+1):
        if check_budget is not None: check_budget()
        x = project(v-apply(momentum,True))
        trial = momentum+apply(x)/bound
        next_p = trial-reg.coefficient_proximal(trial, weight)
        x = project(v-apply(next_p,True))
        tx = apply(x)
        gap = max(0., float(weight*reg.coefficient_value(tx).sum()-(next_p*tx).sum()))
        result = x.to(value.dtype)
        error = math.sqrt(2*gap)+float((result.double()-x).norm())
        reg.last_prox_iterations = inner
        reg.last_prox_gap = gap
        reg.last_prox_error_bound = error
        reg.last_prox_target = target
        if error <= target:
            return result
        next_acceleration = .5*(1+math.sqrt(1+4*acceleration**2))
        momentum = next_p+((acceleration-1)/next_acceleration)*(next_p-p)
        p, acceleration = next_p, next_acceleration
    # Distinguish a solved double-precision prox whose requested certificate
    # cannot survive conversion back to the caller's dtype.
    if math.sqrt(2*gap) <= target and float((result.double()-x).norm()) > target:
        raise ProximalAccuracyError('joint_prox_returned_precision')
    raise ProximalAccuracyError('joint_prox_inner_max_iterations')


class HaarL1Regularizer(Regularizer, ProximalOperator):
    has_full_proximal = True
    def validate_shape(self, shape):
        validate_haar(shape, self.level)

    def __init__(self, level=2):
        if isinstance(level, bool) or not isinstance(level, int) or level <= 0:
            raise ValueError("Haar level must be a positive integer")
        self.level = level
        self.analysis_calls = self.synthesis_calls = 0
        self.transform_seconds = 0.0

    def split_transform(self, shape):
        return HaarOperator(shape, self.level), 1.

    def coefficient_mask(self, z):
        m = torch.ones_like(z)
        h,w = z.shape[-2]//2**self.level,z.shape[-1]//2**self.level
        m[..., :h, :w] = 0
        return m

    def coefficient_value(self, z):
        return z.abs()*self.coefficient_mask(z)

    def coefficient_proximal(self, z, t):
        return z.sign()*(z.abs()-t*self.coefficient_mask(z)).clamp_min(0)

    def _transform(self, x, synthesis=False):
        import time

        if x.is_cuda: torch.cuda.synchronize(x.device)
        started = time.perf_counter()
        if synthesis:
            self.synthesis_calls += 1
        else:
            self.analysis_calls += 1
        operator = HaarOperator(x.shape[1:], self.level)
        result = operator.adjoint(x) if synthesis else operator.forward(x)
        if x.is_cuda: torch.cuda.synchronize(x.device)
        self.transform_seconds += time.perf_counter() - started
        return result

    def value(self, x):
        coefficients = self._transform(x)
        return self.coefficient_value(coefficients).reshape(x.shape[0], -1).sum(dim=1)

    def proximal(self, x, step_size):
        if not math.isfinite(float(step_size)) or step_size < 0:
            raise ValueError("Haar proximal step_size must be finite and nonnegative")
        c = self._transform(x)
        result = self.coefficient_proximal(c, step_size)
        return self._transform(result, synthesis=True)


class ZeroRegularizer(Regularizer, ProximalOperator):
    has_full_proximal = supports_box_proximal = True

    def value(self, x):
        return x.new_zeros(x.shape[0])

    def gradient(self, x):
        return torch.zeros_like(x)

    def proximal(self, x, step_size):
        return x.clone()

    def split_transform(self, shape):
        return None, 0.

    def coefficient_value(self, z):
        return z.new_zeros(z.shape)

    def coefficient_proximal(self, z, t):
        return z.clone()


__all__ = [
    "Regularizer",
    "ProximalOperator",
    "TikhonovRegularizer",
    "TVRegularizer",
    "HaarL1Regularizer",
]


__all__ += ['GradientHuberRegularizer']


__all__ += ['ZeroRegularizer']

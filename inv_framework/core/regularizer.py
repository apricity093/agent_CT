"""Regularizers and their primal, dual and joint proximal operators."""
from __future__ import annotations
from typing import Literal, Tuple, Optional
from inv_framework.core.operator import gradient, gradient_adjoint, LinearOperator, HaarOperator, validate_haar
from abc import ABC, abstractmethod
import torch
import math
import time
from inv_framework.core.array_math import shrink


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
    """Isotropic or anisotropic TV on declared spatial axes (default final two).

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
        axes=(-2, -1), spacing=None, boundary="neumann",
    ):
        if mode not in ("isotropic", "anisotropic"):
            raise ValueError("TV mode must be 'isotropic' or 'anisotropic'.")
        if int(num_iterations) <= 0:
            raise ValueError("TV num_iterations must be positive.")
        if float(tolerance) < 0.0:
            raise ValueError("TV tolerance must be nonnegative.")
        from inv_framework.core.array_math import Differences
        self.differences = Differences(axes, spacing, boundary)
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
        return GradientOperator(shape, self.differences.axes, self.differences.spacing, self.differences.boundary), self.differences.bound

    def coefficient_value(self, z):
        return z.abs().square().sum(1).sqrt() if self.mode == 'isotropic' else z.abs()

    def coefficient_subgradient(self, z):
        """Minimum-norm coefficient subgradient, including zero differences."""
        if self.mode == 'anisotropic':
            return z.sgn()
        return z / z.abs().square().sum(1, keepdim=True).sqrt().clamp_min(torch.finfo(z.real.dtype).tiny)

    def coefficient_proximal(self, z, t):
        if self.mode == 'isotropic':
            return shrink(z, t, axis=1)
        return shrink(z, t)

    _gradient = staticmethod(gradient)
    _gradient_adjoint = staticmethod(gradient_adjoint)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        z = self.differences.forward(x).movedim(0, 1)
        return self.coefficient_value(z).flatten(1).sum(1)

    def proximal(self, x, step_size):
        weight = float(step_size)
        if weight < 0:
            raise ValueError('TV proximal step must be nonnegative')
        self.last_prox_iterations = 0
        if weight == 0:
            return x.clone()
        p = torch.zeros_like(self.differences.forward(x))
        momentum = p.clone(); acceleration = 1.; previous = x
        for iteration in range(1, self.num_iterations+1):
            primal = x-weight*self.differences.adjoint(momentum)
            trial = momentum+self.differences.forward(primal)/(self.differences.bound*weight)
            next_p = trial-shrink(trial, 1., axis=0 if self.mode == 'isotropic' else None)
            next_acceleration = .5*(1+math.sqrt(1+4*acceleration**2))
            momentum = next_p+(acceleration-1)/next_acceleration*(next_p-p)
            p, acceleration = next_p, next_acceleration
            current = x-weight*self.differences.adjoint(p)
            self.last_prox_iterations = iteration
            if self.tolerance > 0 and float((current-previous).norm()) <= self.tolerance*max(1., float(previous.norm())):
                return current
            previous = current
        return current



class GradientHuberRegularizer(Regularizer):
    """h_delta(r)=r²/(2 delta) for r<=delta, else r-delta/2.

    grouped=True uses the spatial gradient group modulus; default False
    preserves the existing sum of component Huber penalties. Final two axes,
    unit spacing, zero-terminal boundaries, independent channels in both.
    """
    def __init__(self, delta=0.01, grouped=False):
        if not math.isfinite(float(delta)) or delta <= 0:
            raise ValueError("delta must be finite and positive")
        self.delta = float(delta)
        self.grouped = grouped
        self.gradient_calls = self.gradient_adjoint_calls = 0

    def split_transform(self, shape):
        from inv_framework.core.operator import GradientOperator
        return GradientOperator(shape), 8.

    def _magnitude(self, z):
        return z.abs().square().sum(1, keepdim=True).sqrt() if self.grouped else z.abs()

    def coefficient_value(self, z):
        a = self._magnitude(z)
        return torch.where(a <= self.delta, a.square()/(2*self.delta), a-self.delta/2)

    def coefficient_proximal(self, z, t):
        a = self._magnitude(z)
        scale = torch.maximum(a, a.new_tensor(self.delta+t)).clamp_min(torch.finfo(a.dtype).tiny)
        return z*(1-t/scale)

    def value(self, x):
        z = torch.stack(gradient(x), dim=1)
        return self.coefficient_value(z).flatten(1).sum(1)

    def gradient(self, x):
        z = torch.stack(gradient(x), dim=1)
        derivative = z/self._magnitude(z).clamp_min(self.delta)
        return gradient_adjoint(derivative[:, 0], derivative[:, 1])


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
        return self.operator, 1. if self.operator is None else self.operator.differences.bound

    def coefficient_value(self, z):
        return .5*z.abs().square()

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
        return 0.5 * transformed.reshape(transformed.shape[0], -1).abs().square().sum(dim=1)

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``L^T L x`` (or ``x`` when ``L`` is the identity)."""
        if self.operator is None:
            return x
        return self.operator.adjoint(self.operator.forward(x))


class ProximalAccuracyError(RuntimeError):
    pass


def _proximal_residual_certificate(reg, transform, x, v, dual, tx, weight,
                                   min_value, max_value, gap):
    """Distance certificates for the box-constrained, strongly convex prox.

    A valid coefficient subgradient
    gives ||x-prox(v)|| <= dist(0, x-v+T*p+N_C(x)). Roundoff is bounded with
    gamma_n=n*u/(1-n*u), counting real arithmetic operations (normal finite
    arithmetic). Near zero coefficients use a feasible-dual gap certificate;
    subgradient selection uncertainty is not an arithmetic precision floor.
    Global gap subtraction is never used for acceptance.
    """
    u = torch.finfo(x.real.dtype).eps/2
    def gamma(n):
        return n*u/(1-n*u) if n*u < 1 else math.inf
    def upper_norm(a):
        # Products, real/imaginary addition, reduction and square root.
        reduction_error = gamma(2*a.numel()+2)
        return float(a.norm())/(1-reduction_error) if reduction_error < 1 else math.inf

    haar = isinstance(reg, HaarL1Regularizer)
    grouped = not haar and reg.mode == 'isotropic'
    axis = 1 if grouped else None
    magnitude = tx.abs().square().sum(axis, keepdim=True).sqrt() if grouped else tx.abs()
    tiny = torch.finfo(x.real.dtype).tiny
    unit = tx/magnitude.clamp_min(tiny)
    radius = weight*(reg.coefficient_mask(tx) if haar else torch.ones_like(magnitude))
    # At an exact zero gradient any feasible coefficient dual is a subgradient.
    dual_size = dual.abs().square().sum(axis, keepdim=True).sqrt() if grouped else dual.abs()
    zero_dual = dual*(radius/dual_size.clamp_min(tiny)).clamp_max(1)
    p = torch.where(magnitude > 0, radius*unit, zero_dual)

    if haar:
        operations = 6*reg.level  # two axes, addition/division/constant per axis
        transform_error = gamma(operations)*upper_norm(x)
        # An unresolved Haar coefficient may have either sign/phase. Its
        # subgradient uncertainty is retained, rather than treating fl(Tx)=0
        # as an exact zero of the mathematical transform.
        separation = (magnitude-transform_error).clamp_min(tiny)
        direction_error = (2*transform_error/separation).clamp_max(2)
        subgradient_uncertainty = upper_norm(radius*direction_error)
        coefficient_error = 0.
        adjoint_error = gamma(operations)*upper_norm(p)
        transform_norm = 1.
    else:
        dimensions = len(transform.differences.axes)
        transform_norm = math.sqrt(transform.differences.bound)
        # Difference, spacing representation and division: gamma_3 relative
        # forward error. Normalizing a nonzero group adds two such errors.
        subgradient_uncertainty = 0.
        coefficient_error = (2*gamma(3)/(1-gamma(3)))*upper_norm(radius.expand_as(magnitude))
        adjoint_error = gamma(2*dimensions+3)*transform_norm*upper_norm(p)

    # Modulus, normalization, radius multiplication and zero-dual projection.
    group_size = tx.shape[1] if grouped else 1
    coefficient_error += gamma(2*group_size+8)*upper_norm(radius.expand_as(magnitude))
    adjoint = transform.adjoint(p)
    residual = x-v+adjoint
    # Bounds are the representable endpoints used by the numerical projection.
    if min_value is not None:
        residual = torch.where((x == min_value) & (residual > 0), 0., residual)
    if max_value is not None:
        residual = torch.where((x == max_value) & (residual < 0), 0., residual)
    arithmetic_error = gamma(2)*(upper_norm(x)+upper_norm(v)+upper_norm(adjoint))
    rounding_bound = transform_norm*coefficient_error+adjoint_error+arithmetic_error
    residual_bound = upper_norm(residual)
    error = residual_bound+rounding_bound+transform_norm*subgradient_uncertainty
    if haar or grouped or x.is_complex():
        # Use the current dual, including at zero/near-zero coefficients.
        # s*dual is mathematically feasible: the group norm is bounded above,
        # and scalar division is rounded towards zero. No iterate is changed.
        norm_error = gamma(2*group_size+4)
        max_dual_norm = float(dual_size.max())/(1-norm_error)
        scale = min(1., weight/max_dual_norm) if max_dual_norm else 1.
        scale = math.nextafter(scale, 0.)
        dot = (dual.conj()*tx).real
        dot_abs = dual.abs()*tx.abs()
        if grouped:
            dot = dot.sum(1, keepdim=True)
            dot_abs = dot_abs.sum(1, keepdim=True)
        density = (radius*magnitude-scale*dot).clamp_min(0)
        # Bound each small dot/norm first, then sum nonnegative quantities;
        # subtracting rounded global objectives would lose the small gap.
        local_roundoff = gamma(2*group_size+8)*(radius*magnitude+scale*dot_abs)
        reduction = 1-gamma(density.numel()+2)
        gap_value = float(density.sum())/reduction
        gap_roundoff = float(local_roundoff.sum())/reduction
        forward_error = (gamma(6*reg.level)*upper_norm(x) if haar else
                         gamma(3)/(1-gamma(3))*upper_norm(tx))
        gap_roundoff += 2*weight*math.sqrt(magnitude.numel())*forward_error
        # Projection was formed using dual, rather than its exact feasible
        # rescaling. Include that perturbation and the adjoint/subtraction errors.
        dual_norm = upper_norm(dual)
        adjoint_roundoff = gamma(6*reg.level if haar else 2*dimensions+3)*transform_norm*dual_norm
        projection_error = (adjoint_roundoff+(1-scale)*transform_norm*dual_norm
                            +gamma(1)*(upper_norm(v)+transform_norm*dual_norm+adjoint_roundoff))
        gap_error = projection_error+math.sqrt(projection_error**2+2*(gap_value+gap_roundoff))
        gap_floor = projection_error+math.sqrt(projection_error**2+2*gap_roundoff)
        error = min(error, gap_error)
        rounding_bound = min(rounding_bound, gap_floor)
    if not haar and not grouped and not x.is_complex():
        # Real anisotropic differences preserve signs under rounding. Define
        # the exact feasible dual by rescaling the represented radius to the
        # requested weight. The nonnegative, per-coefficient Fenchel defect
        # then has a relative error bound, including subtraction near zero;
        # there is no subtraction of two independently rounded global sums.
        gap_upper = gap/(1-gamma(tx.numel()+6))+torch.finfo(x.dtype).tiny*tx.numel()
        projection_error = (adjoint_error
                            + transform_norm*gamma(1)*upper_norm(dual)
                            + gamma(1)*(upper_norm(v)+transform_norm*upper_norm(dual)+adjoint_error))
        # x=P_C(v-T*p+e), ||e||<=projection_error, hence by convexity
        # .5*d² <= gap_upper + ||e||*d for d=||x-prox(v)||.
        gap_error = projection_error+math.sqrt(projection_error**2+2*gap_upper)
        error = min(error, gap_error)
        rounding_bound = min(rounding_bound, 2*projection_error)
    return error, rounding_bound


def joint_prox(reg, value, weight, *, min_value=None, max_value=None,
               tolerance=1e-5, max_iterations=1000, iteration=1, check_budget=None):
    """Dual FGP for .5||x-v||² + weight R(Tx) + indicator_C(x).

    Requested distance errors decay as k^-2.1. Acceptance uses subgradient
    residual or feasible-dual gap bounds with arithmetic error included,
    not a rounded zero gap. Arrays retain the caller precision and device. A target
    below the certificate's arithmetic bound reports returned_precision;
    otherwise a spent iteration budget reports inner_max_iterations.
    """
    from inv_framework.solvers._utils import apply_box_constraints
    transform, bound = reg.split_transform(value.shape[1:])
    bound = transform.squared_norm_upper_bound(value)
    if bound == 0:
        return value.clamp(min_value, max_value) if min_value is not None or max_value is not None else value.clone()
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
    v = value.clone()
    p = torch.zeros_like(apply(v))
    momentum = p.clone()
    acceleration = 1.
    reg.last_prox_iterations = 0
    for inner in range(1, max_iterations+1):
        if check_budget is not None: check_budget()
        x = project(v-apply(momentum,True))
        trial = momentum+apply(x)/bound
        separable = isinstance(reg, HaarL1Regularizer) or getattr(reg, 'mode', None) == 'anisotropic'
        if separable:
            radius = weight*(reg.coefficient_mask(trial) if isinstance(reg, HaarL1Regularizer) else torch.ones_like(trial.real))
            next_p = trial.sgn()*torch.minimum(trial.abs(), radius)
        else:
            next_p = trial-reg.coefficient_proximal(trial, weight)
        x = project(v-apply(next_p,True))
        tx = apply(x)
        if separable:
            # Per-coefficient Fenchel gap avoids cancellation of global sums.
            gap = max(0., float((tx.abs()*(radius-(next_p.conj()*tx.sgn()).real)).sum()))
        else:
            gap = max(0., float(weight*reg.coefficient_value(tx).sum()-(next_p.conj()*tx).sum().real))
        error, rounding_bound = _proximal_residual_certificate(
            reg, transform, x, v, next_p, tx, weight, min_value, max_value, gap)
        reg.last_prox_iterations = inner
        reg.last_prox_gap = gap
        reg.last_prox_error_bound = error
        reg.last_prox_target = target
        reg.last_prox_rounding_bound = rounding_bound
        if error <= target:
            return x
        if rounding_bound > target:
            raise ProximalAccuracyError('joint_prox_returned_precision')
        next_acceleration = .5*(1+math.sqrt(1+4*acceleration**2))
        momentum = next_p+((acceleration-1)/next_acceleration)*(next_p-p)
        p, acceleration = next_p, next_acceleration
    raise ProximalAccuracyError('joint_prox_inner_max_iterations')


class HaarL1Regularizer(Regularizer, ProximalOperator):
    has_full_proximal = True
    def validate_shape(self, shape):
        validate_haar(shape, self.level)

    def __init__(self, level=2, penalize_lowpass=False):
        if isinstance(level, bool) or not isinstance(level, int) or level <= 0:
            raise ValueError("Haar level must be a positive integer")
        self.level = level
        self.penalize_lowpass = penalize_lowpass
        self.analysis_calls = self.synthesis_calls = 0
        self.transform_seconds = 0.0

    def split_transform(self, shape):
        return HaarOperator(shape, self.level), 1.

    def coefficient_mask(self, z):
        m = torch.ones_like(z.real)
        h,w = z.shape[-2]//2**self.level,z.shape[-1]//2**self.level
        if not self.penalize_lowpass:
            m[..., :h, :w] = 0
        return m

    def coefficient_value(self, z):
        return z.abs()*self.coefficient_mask(z)

    def coefficient_proximal(self, z, t):
        return shrink(z, t*self.coefficient_mask(z))

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

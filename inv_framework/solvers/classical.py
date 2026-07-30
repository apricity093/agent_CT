"""Classical iterative and direct solvers.

All three require a `LinearOperator` (an explicit A^T is essential): the
functions assert this and the solver wrappers below raise a clear error if
a non-linear operator is passed.

- `fbp`: Filtered Back-Projection. Designed for parallel-beam Radon; uses
  the Ram-Lak filter along the detector axis.
- `sirt`: Simultaneous Iterative Reconstruction Technique with row/column
  weighting.
- `landweber`: plain gradient descent on ||A x - y||^2.
"""

import math
import torch
from torch.fft import rfft, irfft

from ..operators.base import ForwardOperator, LinearOperator
from .base import InverseProblemSolver


def _require_linear(op: ForwardOperator, name: str) -> None:
    if not isinstance(op, LinearOperator):
        raise TypeError(
            f"{name} requires a LinearOperator (needs an explicit adjoint); "
            f"got {type(op).__name__}. Use DIPSolver/INRSolver/DPSSolver for "
            f"non-linear forward models."
        )


def _ram_lak(n: int, device) -> torch.Tensor:
    f = torch.zeros(n, device=device)
    f[0] = 0.25
    odd = torch.arange(1, n, 2, device=device).float()
    odd_adj = torch.where(2 * odd > n, n - odd, odd)
    f[1::2] = -1.0 / (math.pi * odd_adj) ** 2
    return f


def _filter_sino(sino: torch.Tensor, padded: bool = True) -> torch.Tensor:
    """Apply the Ram-Lak filter along the last (detector) dimension."""
    W = sino.shape[-1]
    filt_w = 2 * W if padded else W
    filt = _ram_lak(filt_w, sino.device)
    filt_rfft = rfft(filt)
    sino_rfft = rfft(sino, n=filt_w) * filt_rfft
    out = irfft(sino_rfft, n=filt_w)
    return out[..., :W]


def fbp(operator: LinearOperator,
        y: torch.Tensor,
        scale: float = None) -> torch.Tensor:
    """Filtered Back-Projection reconstruction."""
    _require_linear(operator, "fbp")
    y_filt = _filter_sino(y)
    rec = operator.adjoint(y_filt)
    if scale is None:
        # Standard parallel-beam FBP scaling.
        num_angles = getattr(operator, "num_angles", y.shape[-2])
        scale = math.pi / num_angles
    return rec * scale


def sirt(operator: LinearOperator,
         y: torch.Tensor,
         num_iterations: int = 100,
         min_value: float = None,
         max_value: float = None,
         x_init: torch.Tensor = None) -> torch.Tensor:
    """SIRT with row/column weighting."""
    _require_linear(operator, "sirt")
    device = y.device
    B = y.shape[0]
    domain = (B, *operator.domain_shape)
    range_ = (B, *operator.range_shape)

    R = operator.forward(torch.ones(domain, device=device))
    R = torch.where(R < 1e-8, torch.full_like(R, float('inf')), R).reciprocal()
    C = operator.adjoint(torch.ones(range_, device=device))
    C = torch.where(C < 1e-8, torch.full_like(C, float('inf')), C).reciprocal()

    x = (torch.zeros(domain, device=device)
         if x_init is None else x_init.clone())
    for _ in range(num_iterations):
        residual = operator.forward(x) - y
        x = x - C * operator.adjoint(R * residual)
        if min_value is not None or max_value is not None:
            x = x.clamp(min_value, max_value)
    return x


def landweber(operator: LinearOperator,
              y: torch.Tensor,
              num_iterations: int = 100,
              step_size: float = None,
              x_init: torch.Tensor = None,
              min_value: float = None,
              max_value: float = None) -> torch.Tensor:
    """Landweber: x <- x - alpha A^T (A x - y)."""
    _require_linear(operator, "landweber")
    device = y.device
    B = y.shape[0]
    domain = (B, *operator.domain_shape)

    if step_size is None:
        step_size = 1e-3

    x = (torch.zeros(domain, device=device)
         if x_init is None else x_init.clone())
    for _ in range(num_iterations):
        residual = operator.forward(x) - y
        x = x - step_size * operator.adjoint(residual)
        if min_value is not None or max_value is not None:
            x = x.clamp(min_value, max_value)
    return x


class FBPSolver(InverseProblemSolver):
    def __init__(self, scale: float = None):
        self.scale = scale

    def solve(self, measurement, operator, **kwargs):
        return fbp(operator, measurement, scale=self.scale)


class SIRTSolver(InverseProblemSolver):
    def __init__(self,
                 num_iterations: int = 100,
                 min_value: float = None,
                 max_value: float = None):
        self.num_iterations = num_iterations
        self.min_value = min_value
        self.max_value = max_value

    def solve(self, measurement, operator, x_init=None, **kwargs):
        return sirt(operator, measurement,
                    num_iterations=self.num_iterations,
                    min_value=self.min_value,
                    max_value=self.max_value,
                    x_init=x_init)


class LandweberSolver(InverseProblemSolver):
    def __init__(self,
                 num_iterations: int = 100,
                 step_size: float = None,
                 min_value: float = None,
                 max_value: float = None):
        self.num_iterations = num_iterations
        self.step_size = step_size
        self.min_value = min_value
        self.max_value = max_value

    def solve(self, measurement, operator, x_init=None, **kwargs):
        return landweber(operator, measurement,
                         num_iterations=self.num_iterations,
                         step_size=self.step_size,
                         x_init=x_init,
                         min_value=self.min_value,
                         max_value=self.max_value)


# Add-on CT solvers appended after the 2026-06-08 frozen implementation.


def _ct_batch_sqnorm(z: torch.Tensor) -> torch.Tensor:
    return (z.reshape(z.shape[0], -1) ** 2).sum(dim=1)


def _ct_batch_norm(z: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(_ct_batch_sqnorm(z).clamp_min(0.0))


def _ct_batch_view(coeff: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return coeff.reshape((coeff.shape[0],) + (1,) * (target.ndim - 1))


def cgls(operator: LinearOperator,
         y: torch.Tensor,
         num_iterations: int = 10,
         tol: float = 1e-6,
         x_init: torch.Tensor = None,
         min_value: float = None,
         max_value: float = None,
         eps: float = 1e-12) -> torch.Tensor:
    """Conjugate-gradient least squares for min_x ||A x - y||_2^2."""
    from ._utils import (
        apply_box_constraints,
        prepare_initial_image,
        require_linear_operator,
        validate_measurement_shape,
    )

    require_linear_operator(operator, "cgls")
    validate_measurement_shape(y, operator, "cgls")

    x = prepare_initial_image(y, operator, x_init=x_init, initial_value=0.0)
    r = y - operator.forward(x)
    s = operator.adjoint(r)
    p = s.clone()
    gamma = _ct_batch_sqnorm(s)

    for _ in range(int(num_iterations)):
        if torch.all(torch.sqrt(gamma.clamp_min(0.0)) <= float(tol)):
            break
        q = operator.forward(p)
        delta = _ct_batch_sqnorm(q).clamp_min(float(eps))
        alpha = gamma / delta
        x = x + _ct_batch_view(alpha, x) * p
        x = apply_box_constraints(x, min_value=min_value, max_value=max_value)
        r = y - operator.forward(x)
        s_new = operator.adjoint(r)
        gamma_new = _ct_batch_sqnorm(s_new)
        beta = gamma_new / gamma.clamp_min(float(eps))
        p = s_new + _ct_batch_view(beta, p) * p
        gamma = gamma_new
    return x


def lsqr(operator: LinearOperator,
         y: torch.Tensor,
         num_iterations: int = 10,
         damping: float = 0.0,
         atol: float = 1e-6,
         btol: float = 1e-6,
         x_init: torch.Tensor = None,
         min_value: float = None,
         max_value: float = None,
         eps: float = 1e-12) -> torch.Tensor:
    """LSQR using Golub-Kahan bidiagonalization."""
    from ._utils import (
        apply_box_constraints,
        prepare_initial_image,
        require_linear_operator,
        validate_measurement_shape,
    )

    require_linear_operator(operator, "lsqr")
    validate_measurement_shape(y, operator, "lsqr")

    eps = float(eps)
    damping = float(damping)
    x = prepare_initial_image(y, operator, x_init=x_init, initial_value=0.0)
    rhs_norm = _ct_batch_norm(y)

    u = y - operator.forward(x)
    beta = _ct_batch_norm(u)
    u = u / _ct_batch_view(beta.clamp_min(eps), u)

    v = operator.adjoint(u)
    alpha = _ct_batch_norm(v)
    v = v / _ct_batch_view(alpha.clamp_min(eps), v)

    w = v.clone()
    phi_bar = beta.clone()
    rho_bar = alpha.clone()

    for _ in range(int(num_iterations)):
        residual_norm = _ct_batch_norm(y - operator.forward(x))
        stop_tol = float(atol) * rhs_norm + float(btol)
        if torch.all(residual_norm <= stop_tol):
            break

        u = operator.forward(v) - _ct_batch_view(alpha, u) * u
        beta = _ct_batch_norm(u)
        u = u / _ct_batch_view(beta.clamp_min(eps), u)

        v = operator.adjoint(u) - _ct_batch_view(beta, v) * v
        alpha = _ct_batch_norm(v)
        v = v / _ct_batch_view(alpha.clamp_min(eps), v)

        if damping > 0.0:
            rho_damped = torch.sqrt(rho_bar * rho_bar + damping * damping).clamp_min(eps)
            phi_bar = (rho_bar / rho_damped) * phi_bar
            rho_bar = rho_damped

        rho = torch.sqrt(rho_bar * rho_bar + beta * beta).clamp_min(eps)
        c = rho_bar / rho
        s = beta / rho
        theta = s * alpha
        rho_bar = -c * alpha
        phi = c * phi_bar
        phi_bar = s * phi_bar

        x = x + _ct_batch_view(phi / rho, x) * w
        x = apply_box_constraints(x, min_value=min_value, max_value=max_value)
        w = v - _ct_batch_view(theta / rho, w) * w
    return x


def fdk(operator: LinearOperator, y: torch.Tensor, **kwargs) -> torch.Tensor:
    """Geometry-specific FDK reconstruction via an operator backend."""
    from ._utils import require_linear_operator, validate_measurement_shape

    require_linear_operator(operator, "fdk")
    validate_measurement_shape(y, operator, "fdk")
    backend = getattr(operator, "fdk", None)
    if not callable(backend):
        raise NotImplementedError(
            "fdk requires a LinearOperator with an operator.fdk(y, **kwargs) "
            "backend; this project does not provide a generic FDK kernel."
        )
    rec = backend(y, **kwargs)
    if not isinstance(rec, torch.Tensor):
        raise TypeError(f"operator.fdk(...) must return a torch.Tensor; got {type(rec).__name__}.")
    expected = (y.shape[0], *tuple(operator.domain_shape))
    if tuple(rec.shape) != expected:
        raise ValueError(f"operator.fdk(...) must return shape {expected}; got {tuple(rec.shape)}.")
    return rec.to(device=y.device, dtype=y.dtype)


class CGLSSolver(InverseProblemSolver):
    def __init__(self,
                 num_iterations: int = 10,
                 tol: float = 1e-6,
                 min_value: float = None,
                 max_value: float = None):
        self.num_iterations = num_iterations
        self.tol = tol
        self.min_value = min_value
        self.max_value = max_value

    def solve(self, measurement, operator, x_init=None, **kwargs):
        return cgls(operator, measurement,
                    num_iterations=self.num_iterations,
                    tol=self.tol,
                    x_init=x_init,
                    min_value=self.min_value,
                    max_value=self.max_value)


class LSQRSolver(InverseProblemSolver):
    def __init__(self,
                 num_iterations: int = 10,
                 damping: float = 0.0,
                 atol: float = 1e-6,
                 btol: float = 1e-6,
                 min_value: float = None,
                 max_value: float = None):
        self.num_iterations = num_iterations
        self.damping = damping
        self.atol = atol
        self.btol = btol
        self.min_value = min_value
        self.max_value = max_value

    def solve(self, measurement, operator, x_init=None, **kwargs):
        return lsqr(operator, measurement,
                    num_iterations=self.num_iterations,
                    damping=self.damping,
                    atol=self.atol,
                    btol=self.btol,
                    x_init=x_init,
                    min_value=self.min_value,
                    max_value=self.max_value)


class FDKSolver(InverseProblemSolver):
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)

    def solve(self, measurement, operator, **kwargs):
        params = dict(self.kwargs)
        params.update(kwargs)
        return fdk(operator, measurement, **params)

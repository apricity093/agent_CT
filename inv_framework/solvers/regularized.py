"""Tikhonov and total-variation reconstruction solvers."""

from typing import Optional

import torch

from ..operators.base import ForwardOperator, LinearOperator
from ..regularizers import TikhonovRegularizer, TVRegularizer
from ._utils import (
    apply_box_constraints,
    prepare_initial_image,
    require_linear_operator,
    validate_measurement_shape,
)
from .base import InverseProblemSolver


def _batch_inner(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left * right).reshape(left.shape[0], -1).sum(dim=1)


def _batch_norm(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(_batch_inner(x, x).clamp_min(0.0))


def _batch_view(coefficients: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return coefficients.reshape((coefficients.shape[0],) + (1,) * (target.ndim - 1))


def _validate_regularization_operator(
    reconstruction_operator: LinearOperator,
    regularization_operator: Optional[LinearOperator],
) -> None:
    if regularization_operator is None:
        return
    if not isinstance(regularization_operator, LinearOperator):
        raise TypeError(
            "regularization_operator must be a LinearOperator or None; "
            f"got {type(regularization_operator).__name__}."
        )
    if tuple(regularization_operator.domain_shape) != tuple(reconstruction_operator.domain_shape):
        raise ValueError(
            "regularization_operator.domain_shape must match operator.domain_shape; "
            f"got {tuple(regularization_operator.domain_shape)} and "
            f"{tuple(reconstruction_operator.domain_shape)}."
        )


def tikhonov(
    operator: LinearOperator,
    y: torch.Tensor,
    reg_strength: float = 1e-2,
    num_iterations: int = 100,
    tolerance: float = 1e-6,
    x_init: Optional[torch.Tensor] = None,
    regularization_operator: Optional[LinearOperator] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Solve a linear Tikhonov problem by batched conjugate gradients.

    The objective is ``0.5 ||Ax-y||^2 + 0.5 * reg_strength ||Lx||^2``.
    ``L`` is the identity unless ``regularization_operator`` is supplied.
    Optional box constraints are applied once to the converged result; they do
    not alter the conjugate-gradient recurrence.
    """
    require_linear_operator(operator, "tikhonov")
    validate_measurement_shape(y, operator, "tikhonov")
    _validate_regularization_operator(operator, regularization_operator)
    if float(reg_strength) < 0.0:
        raise ValueError("reg_strength must be nonnegative.")
    if int(num_iterations) <= 0:
        raise ValueError("num_iterations must be positive.")
    if float(tolerance) < 0.0:
        raise ValueError("tolerance must be nonnegative.")

    regularizer = TikhonovRegularizer(regularization_operator)
    strength = float(reg_strength)
    epsilon = float(eps)

    def normal_equation(z: torch.Tensor) -> torch.Tensor:
        return operator.adjoint(operator.forward(z)) + strength * regularizer.gradient(z)

    x = prepare_initial_image(y, operator, x_init=x_init, initial_value=0.0)
    rhs = operator.adjoint(y)
    residual = rhs - normal_equation(x)
    direction = residual.clone()
    residual_sq = _batch_inner(residual, residual)
    threshold = float(tolerance) * _batch_norm(rhs).clamp_min(1.0)

    for _ in range(int(num_iterations)):
        if bool(torch.all(torch.sqrt(residual_sq.clamp_min(0.0)) <= threshold)):
            break
        normal_direction = normal_equation(direction)
        denominator = _batch_inner(direction, normal_direction)
        alpha = torch.where(
            denominator.abs() > epsilon,
            residual_sq / denominator,
            torch.zeros_like(denominator),
        )
        x = x + _batch_view(alpha, x) * direction
        next_residual = residual - _batch_view(alpha, residual) * normal_direction
        next_residual_sq = _batch_inner(next_residual, next_residual)
        beta = torch.where(
            residual_sq > epsilon,
            next_residual_sq / residual_sq,
            torch.zeros_like(residual_sq),
        )
        direction = next_residual + _batch_view(beta, direction) * direction
        residual = next_residual
        residual_sq = next_residual_sq

    return apply_box_constraints(x, min_value=min_value, max_value=max_value)


def _estimate_lipschitz(
    operator: LinearOperator,
    reference: torch.Tensor,
    num_iterations: int,
    eps: float = 1e-12,
) -> float:
    sample_size = 1
    for value in operator.domain_shape:
        sample_size *= int(value)
    seed = torch.linspace(
        0.5,
        1.5,
        steps=sample_size,
        dtype=reference.dtype,
        device=reference.device,
    ).reshape((1, *tuple(operator.domain_shape)))
    vector = seed / _batch_view(_batch_norm(seed).clamp_min(eps), seed)

    for _ in range(int(num_iterations)):
        next_vector = operator.adjoint(operator.forward(vector))
        norm = _batch_norm(next_vector)
        if bool(torch.all(norm <= eps)):
            return 0.0
        vector = next_vector / _batch_view(norm.clamp_min(eps), next_vector)

    image = operator.adjoint(operator.forward(vector))
    return float(_batch_inner(vector, image).clamp_min(0.0).max().item())


def tv_fista(
    operator: LinearOperator,
    y: torch.Tensor,
    reg_strength: float = 1e-3,
    num_iterations: int = 50,
    step_size: Optional[float] = None,
    x_init: Optional[torch.Tensor] = None,
    regularizer: Optional[TVRegularizer] = None,
    tolerance: float = 1e-5,
    power_iterations: int = 12,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> torch.Tensor:
    """FISTA for ``0.5 ||Ax-y||^2 + reg_strength * TV(x)``.

    When ``step_size`` is omitted, a deterministic power iteration estimates
    ``||A||^2``. The TV proximal is the dual FGP implementation provided by
    ``TVRegularizer`` and is independent of TIGRE/TomoPy backends.
    """
    require_linear_operator(operator, "tv_fista")
    validate_measurement_shape(y, operator, "tv_fista")
    if len(tuple(operator.domain_shape)) < 2:
        raise ValueError("tv_fista requires at least two spatial domain dimensions.")
    if float(reg_strength) < 0.0:
        raise ValueError("reg_strength must be nonnegative.")
    if int(num_iterations) <= 0:
        raise ValueError("num_iterations must be positive.")
    if float(tolerance) < 0.0:
        raise ValueError("tolerance must be nonnegative.")
    if int(power_iterations) <= 0:
        raise ValueError("power_iterations must be positive.")

    tv = regularizer or TVRegularizer()
    if not isinstance(tv, TVRegularizer):
        raise TypeError(f"regularizer must be a TVRegularizer; got {type(tv).__name__}.")

    if step_size is None:
        lipschitz = _estimate_lipschitz(operator, y, int(power_iterations))
        data_step = 1.0 if lipschitz <= 1e-12 else 0.99 / lipschitz
    else:
        data_step = float(step_size)
        if data_step <= 0.0:
            raise ValueError("step_size must be positive when provided.")

    x = prepare_initial_image(y, operator, x_init=x_init, initial_value=0.0)
    momentum = x.clone()
    acceleration = 1.0

    for _ in range(int(num_iterations)):
        data_gradient = operator.adjoint(operator.forward(momentum) - y)
        candidate = momentum - data_step * data_gradient
        next_x = tv.proximal(candidate, data_step * float(reg_strength))
        next_x = apply_box_constraints(next_x, min_value=min_value, max_value=max_value)

        next_acceleration = 0.5 * (1.0 + (1.0 + 4.0 * acceleration * acceleration) ** 0.5)
        momentum_scale = (acceleration - 1.0) / next_acceleration
        momentum = next_x + momentum_scale * (next_x - x)

        if float(tolerance) > 0.0:
            delta = _batch_norm(next_x - x)
            scale = _batch_norm(x).clamp_min(1.0)
            if bool(torch.all(delta <= float(tolerance) * scale)):
                x = next_x
                break
        x = next_x
        acceleration = next_acceleration

    return x


class TikhonovSolver(InverseProblemSolver):
    """Unified solver wrapper for conjugate-gradient Tikhonov reconstruction."""

    def __init__(
        self,
        reg_strength: float = 1e-2,
        num_iterations: int = 100,
        tolerance: float = 1e-6,
        regularization_operator: Optional[LinearOperator] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ):
        self.reg_strength = float(reg_strength)
        self.num_iterations = int(num_iterations)
        self.tolerance = float(tolerance)
        self.regularization_operator = regularization_operator
        self.min_value = min_value
        self.max_value = max_value

    def solve(
        self,
        measurement: torch.Tensor,
        operator: ForwardOperator,
        x_init: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        return tikhonov(
            operator,
            measurement,
            reg_strength=self.reg_strength,
            num_iterations=self.num_iterations,
            tolerance=self.tolerance,
            x_init=x_init,
            regularization_operator=self.regularization_operator,
            min_value=self.min_value,
            max_value=self.max_value,
        )


class TVFISTASolver(InverseProblemSolver):
    """Unified solver wrapper for TV-regularized FISTA reconstruction."""

    def __init__(
        self,
        reg_strength: float = 1e-3,
        num_iterations: int = 50,
        step_size: Optional[float] = None,
        regularizer: Optional[TVRegularizer] = None,
        tolerance: float = 1e-5,
        power_iterations: int = 12,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ):
        self.reg_strength = float(reg_strength)
        self.num_iterations = int(num_iterations)
        self.step_size = step_size
        self.regularizer = regularizer or TVRegularizer()
        self.tolerance = float(tolerance)
        self.power_iterations = int(power_iterations)
        self.min_value = min_value
        self.max_value = max_value

    def solve(
        self,
        measurement: torch.Tensor,
        operator: ForwardOperator,
        x_init: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        return tv_fista(
            operator,
            measurement,
            reg_strength=self.reg_strength,
            num_iterations=self.num_iterations,
            step_size=self.step_size,
            x_init=x_init,
            regularizer=self.regularizer,
            tolerance=self.tolerance,
            power_iterations=self.power_iterations,
            min_value=self.min_value,
            max_value=self.max_value,
        )

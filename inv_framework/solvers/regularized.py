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
    ).reshape((1, *tuple(operator.domain_shape))).expand(reference.shape[0], *tuple(operator.domain_shape))
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
    """Legacy TV entry point using the shared composite FISTA core."""
    from .fista import fista

    tv = regularizer if regularizer is not None else TVRegularizer()
    if not isinstance(tv, TVRegularizer):
        raise TypeError("regularizer must be a TVRegularizer")
    return fista(operator, y, regularizer=tv, reg_strength=reg_strength,
                 num_iterations=num_iterations, step_size=step_size, x_init=x_init,
                 tolerance=tolerance, power_iterations=power_iterations,
                 min_value=min_value, max_value=max_value, legacy=True)


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
        if kwargs.get('fidelity') is not None:
            result=self.solve_detailed(measurement,operator,x_init=x_init,**kwargs)
            if result.status in ('invalid_parameters','numerical_error'):raise ValueError(result.stopping_reason)
            return result.reconstruction
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

    def solve_detailed(
        self,
        measurement: torch.Tensor,
        operator: ForwardOperator,
        x_init: Optional[torch.Tensor] = None,
        *,
        control=None,
        callback=None,
        **kwargs,
    ):
        from .detailed import solve_tikhonov_detailed

        from ..operators.gradient import GradientOperator
        fidelity=kwargs.get("fidelity")
        fidelity_before={k:getattr(fidelity,k,0) for k in ("weight_multiplications","prox_calls","prox_seconds")}
        auxiliary = kwargs.get("regularization_operator", self.regularization_operator)
        before = auxiliary.stats() if isinstance(auxiliary, GradientOperator) else {}
        if fidelity is not None and kwargs.get('min_value',self.min_value) is None and kwargs.get('max_value',self.max_value) is None and callable(getattr(fidelity,'validate_normal',None)):
            try:
                reference=prepare_initial_image(measurement,operator,x_init=x_init,initial_value=0.)
                fidelity.validate_normal(operator,auxiliary,reference)
            except ValueError as error:
                from .detailed import _parameter_failure_result
                return _parameter_failure_result('tikhonov',measurement,operator,x_init=x_init,errors=[str(error)],max_iterations=self.num_iterations,parameters={})

        result = solve_tikhonov_detailed(
            operator,
            measurement,
            reg_strength=kwargs.pop("reg_strength", self.reg_strength),
            num_iterations=kwargs.pop("num_iterations", self.num_iterations),
            tolerance=kwargs.pop("tolerance", self.tolerance),
            x_init=x_init,
            regularization_operator=kwargs.pop("regularization_operator", self.regularization_operator),
            min_value=kwargs.pop("min_value", self.min_value),
            max_value=kwargs.pop("max_value", self.max_value),
            control=control,
            callback=callback,
            fidelity=kwargs.pop("fidelity",None),
        )
        if fidelity is not None:
            from dataclasses import replace
            result=replace(result,resources={**result.resources,**{'fidelity_'+k:getattr(fidelity,k,0)-v for k,v in fidelity_before.items()}},metadata={**result.metadata,'regularized_normal_residual_formula':'||A* gradient f(Ax)+lambda gradient R(x)|| / max(||A* linear_rhs(f)||+lambda ||gradient R(x)||,eps)'})
        if isinstance(auxiliary, GradientOperator):
            from dataclasses import replace
            work = {name: value - before[name] for name, value in auxiliary.stats().items()}
            result = replace(result, resources={**result.resources, **work})
        return result


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

    def solve_detailed(
        self,
        measurement: torch.Tensor,
        operator: ForwardOperator,
        x_init: Optional[torch.Tensor] = None,
        *,
        control=None,
        callback=None,
        **kwargs,
    ):
        from .detailed import solve_tv_fista_detailed

        return solve_tv_fista_detailed(
            operator,
            measurement,
            reg_strength=kwargs.pop("reg_strength", self.reg_strength),
            num_iterations=kwargs.pop("num_iterations", self.num_iterations),
            step_size=kwargs.pop("step_size", self.step_size),
            x_init=x_init,
            regularizer=kwargs.pop("regularizer", self.regularizer),
            tolerance=kwargs.pop("tolerance", self.tolerance),
            power_iterations=kwargs.pop("power_iterations", self.power_iterations),
            min_value=kwargs.pop("min_value", self.min_value),
            max_value=kwargs.pop("max_value", self.max_value),
            control=control,
            callback=callback,
        )

"""Detailed execution paths for the classical CT solvers.

The functions in this module intentionally mirror the established solver
implementations while adding a small amount of instrumentation at the loop
boundary.  The old functions and ``solve()`` methods remain untouched for
callers that only need a reconstruction tensor.  ``solve_detailed`` uses
these paths so that iteration counts, native stopping criteria, trajectory
values and operator counters describe the actual run.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Optional, Sequence

import torch

from ..operators.base import ForwardOperator, LinearOperator
from ..regularizers import TikhonovRegularizer, TVRegularizer
from .base import (
    IterationRecorder,
    IterationRecord,
    SolveControl,
    SolveResult,
    resolve_control,
)
from ._utils import (
    apply_box_constraints,
    make_angle_subsets,
    make_subset_operator,
    prepare_initial_image,
    require_linear_operator,
    select_measurement_subset,
    validate_measurement_shape,
)


Callback = Callable[[IterationRecord], bool | None]


def _sqnorm(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(value.shape[0], -1).square().sum(dim=1)


def _norm(value: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(_sqnorm(value).clamp_min(0.0))


def _global_norm(value: torch.Tensor) -> float:
    return float(value.detach().reshape(-1).norm().item())


def _relative_change(previous: torch.Tensor, current: torch.Tensor) -> float:
    return _global_norm(current - previous) / max(_global_norm(previous), 1e-12)


def _objective(residual: torch.Tensor) -> float:
    return float(0.5 * residual.detach().square().sum().item())


def _normalized(value: float, denominator: float) -> float:
    return float(value) / max(float(denominator), 1e-12)


def _tolerance_reached(
    control: SolveControl,
    *,
    residual: float | None,
    change: float | None,
    require_both: bool = False,
) -> bool:
    tolerance = control.tolerance
    if tolerance is None or float(tolerance) <= 0.0:
        return False
    residual_ok = residual is not None and residual <= float(tolerance)
    change_ok = (
        change is not None
        and change <= float(tolerance)
        and (residual is None or residual <= max(10.0 * float(tolerance), 1e-8))
    )
    return (residual_ok and change_ok) if require_both else (residual_ok or change_ok)


def _finish_prediction(operator: LinearOperator, x: torch.Tensor) -> torch.Tensor:
    # Leave finite validation to the shared recorder so callers receive a
    # structured ``numerical_failure`` SolveResult instead of an opaque
    # exception from the endpoint check.
    return operator.forward(x).detach()


def _finish_status(
    *,
    actual: int,
    limit: int,
    converged: bool,
    cancelled: bool,
    numerical_failure: bool = False,
) -> str:
    if numerical_failure:
        return "numerical_failure"
    if cancelled:
        return "cancelled"
    if converged:
        return "converged"
    if actual >= limit:
        return "max_iterations"
    return "partial"


def _safe_record(
    recorder: IterationRecorder,
    *args: Any,
    **kwargs: Any,
) -> bool:
    """Record one row and let callbacks request cancellation."""

    return recorder.record(*args, **kwargs)


def solve_fbp_detailed(
    operator: LinearOperator,
    measurement: torch.Tensor,
    *,
    scale: float | None = None,
    control: SolveControl | None = None,
    callback: Callback | None = None,
    **kwargs: Any,
) -> SolveResult:
    from .classical import fbp

    _ = kwargs
    control = resolve_control(control, default_iterations=1, callback=callback)
    recorder = IterationRecorder(control, measurement, operator, algorithm="fbp")
    reconstruction = fbp(operator, measurement, scale=scale)
    return recorder.finish(
        reconstruction,
        actual_iterations=0,
        status="non_iterative_completed",
        stopping_reason="direct_solver_completed",
        resources={"trajectory_available": False},
        metadata={"direct": True},
    )


def solve_fdk_detailed(
    operator: LinearOperator,
    measurement: torch.Tensor,
    *,
    control: SolveControl | None = None,
    callback: Callback | None = None,
    **kwargs: Any,
) -> SolveResult:
    from .classical import fdk

    control = resolve_control(control, default_iterations=1, callback=callback)
    recorder = IterationRecorder(control, measurement, operator, algorithm="fdk")
    reconstruction = fdk(operator, measurement, **kwargs)
    return recorder.finish(
        reconstruction,
        actual_iterations=0,
        status="non_iterative_completed",
        stopping_reason="direct_solver_completed",
        resources={"trajectory_available": False},
        metadata={"direct": True},
    )


def solve_sirt_detailed(
    operator: LinearOperator,
    measurement: torch.Tensor,
    *,
    num_iterations: int = 100,
    min_value: float | None = None,
    max_value: float | None = None,
    x_init: torch.Tensor | None = None,
    control: SolveControl | None = None,
    callback: Callback | None = None,
) -> SolveResult:
    require_linear_operator(operator, "sirt")
    batch = validate_measurement_shape(measurement, operator, "sirt")
    control = resolve_control(
        control, default_iterations=int(num_iterations), default_tolerance=1e-5, callback=callback
    )
    limit = min(int(num_iterations), int(control.max_iterations or num_iterations))
    x = prepare_initial_image(measurement, operator, x_init=x_init, initial_value=0.0)
    recorder = IterationRecorder(control, measurement, operator, algorithm="sirt")
    recorder.set_initial(x)
    domain = (batch, *tuple(operator.domain_shape))
    range_ = (batch, *tuple(operator.range_shape))
    row_weight = operator.forward(torch.ones(domain, device=measurement.device, dtype=measurement.dtype))
    row_weight = torch.where(row_weight < 1e-8, torch.full_like(row_weight, float("inf")), row_weight).reciprocal()
    column_weight = operator.adjoint(torch.ones(range_, device=measurement.device, dtype=measurement.dtype))
    column_weight = torch.where(column_weight < 1e-8, torch.full_like(column_weight, float("inf")), column_weight).reciprocal()
    measurement_norm = _global_norm(measurement)
    actual = 0
    converged = False
    cancelled = False
    residual = torch.zeros_like(measurement)
    for iteration in range(1, limit + 1):
        residual = operator.forward(x) - measurement
        previous = x
        x = x - column_weight * operator.adjoint(row_weight * residual)
        x = apply_box_constraints(x, min_value=min_value, max_value=max_value)
        change = _relative_change(previous, x)
        residual_value = _normalized(_global_norm(residual), measurement_norm)
        converged = _tolerance_reached(control, residual=residual_value, change=change)
        actual = iteration
        if not _safe_record(
            recorder,
            iteration,
            x,
            residual=residual,
            objective=_objective(residual),
            algorithm_residual=residual_value,
            stopping_candidate=converged,
            metadata={"state": "pre_update"},
        ):
            cancelled = True
            break
        if converged and control.stop_on_convergence:
            break
    prediction = _finish_prediction(operator, x)
    final_residual_tensor = prediction - measurement
    final_residual = _normalized(_global_norm(final_residual_tensor), measurement_norm)
    status = _finish_status(
        actual=actual,
        limit=limit,
        converged=converged,
        cancelled=cancelled,
        numerical_failure=recorder.numerical_failure,
    )
    if actual == 0 and not cancelled:
        status = "partial"
    return recorder.finish(
        x,
        actual_iterations=actual,
        status=status,
        stopping_reason=(
            "relative_residual_and_iterate_tolerance" if converged else
            "callback_cancelled" if cancelled else
            "maximum_iterations_reached"
        ),
        final_residual=final_residual,
        final_objective=_objective(final_residual_tensor),
        predicted_measurement=prediction,
        metadata={"criterion": "normalized_data_residual_or_relative_iterate_change"},
    )


def solve_landweber_detailed(
    operator: LinearOperator,
    measurement: torch.Tensor,
    *,
    num_iterations: int = 100,
    step_size: float | None = None,
    x_init: torch.Tensor | None = None,
    min_value: float | None = None,
    max_value: float | None = None,
    control: SolveControl | None = None,
    callback: Callback | None = None,
) -> SolveResult:
    require_linear_operator(operator, "landweber")
    validate_measurement_shape(measurement, operator, "landweber")
    control = resolve_control(
        control, default_iterations=int(num_iterations), default_tolerance=1e-5, callback=callback
    )
    limit = min(int(num_iterations), int(control.max_iterations or num_iterations))
    x = prepare_initial_image(measurement, operator, x_init=x_init, initial_value=0.0)
    recorder = IterationRecorder(control, measurement, operator, algorithm="landweber")
    recorder.set_initial(x)
    step = 1e-3 if step_size is None else float(step_size)
    measurement_norm = _global_norm(measurement)
    actual = 0
    converged = False
    cancelled = False
    residual = torch.zeros_like(measurement)
    for iteration in range(1, limit + 1):
        residual = operator.forward(x) - measurement
        previous = x
        x = x - step * operator.adjoint(residual)
        x = apply_box_constraints(x, min_value=min_value, max_value=max_value)
        change = _relative_change(previous, x)
        residual_value = _normalized(_global_norm(residual), measurement_norm)
        converged = _tolerance_reached(control, residual=residual_value, change=change)
        actual = iteration
        if not _safe_record(
            recorder,
            iteration,
            x,
            residual=residual,
            objective=_objective(residual),
            algorithm_residual=residual_value,
            step_size=step,
            stopping_candidate=converged,
            metadata={"state": "pre_update"},
        ):
            cancelled = True
            break
        if converged and control.stop_on_convergence:
            break
    prediction = _finish_prediction(operator, x)
    final_residual_tensor = prediction - measurement
    final_residual = _normalized(_global_norm(final_residual_tensor), measurement_norm)
    status = _finish_status(
        actual=actual,
        limit=limit,
        converged=converged,
        cancelled=cancelled,
        numerical_failure=recorder.numerical_failure,
    )
    return recorder.finish(
        x,
        actual_iterations=actual,
        status=status,
        stopping_reason=(
            "relative_residual_and_iterate_tolerance" if converged else
            "callback_cancelled" if cancelled else
            "maximum_iterations_reached"
        ),
        final_residual=final_residual,
        final_objective=_objective(final_residual_tensor),
        predicted_measurement=prediction,
        metadata={"criterion": "normalized_data_residual_or_relative_iterate_change", "step_size": step},
    )


def solve_cgls_detailed(
    operator: LinearOperator,
    measurement: torch.Tensor,
    *,
    num_iterations: int = 10,
    tol: float = 1e-6,
    x_init: torch.Tensor | None = None,
    min_value: float | None = None,
    max_value: float | None = None,
    eps: float = 1e-12,
    control: SolveControl | None = None,
    callback: Callback | None = None,
) -> SolveResult:
    require_linear_operator(operator, "cgls")
    validate_measurement_shape(measurement, operator, "cgls")
    control = resolve_control(control, default_iterations=int(num_iterations), default_tolerance=None, callback=callback)
    limit = min(int(num_iterations), int(control.max_iterations or num_iterations))
    x = prepare_initial_image(measurement, operator, x_init=x_init, initial_value=0.0)
    recorder = IterationRecorder(control, measurement, operator, algorithm="cgls")
    recorder.set_initial(x)
    residual = measurement - operator.forward(x)
    adjoint_residual = operator.adjoint(residual)
    gamma = _sqnorm(adjoint_residual)
    initial_normal = max(_global_norm(adjoint_residual), float(eps))
    normal_tolerance = float(tol)
    if control.tolerance is not None:
        normal_tolerance = float(control.tolerance)
    actual = 0
    converged = False
    cancelled = False
    if normal_tolerance > 0.0 and _global_norm(adjoint_residual) <= normal_tolerance:
        data_residual = -residual
        recorder.record(
            0,
            x,
            residual=data_residual,
            objective=_objective(data_residual),
            algorithm_residual=_normalized(_global_norm(adjoint_residual), initial_normal),
            stopping_candidate=True,
        )
        converged = True
    direction = adjoint_residual.clone()
    while not converged and actual < limit:
        q = operator.forward(direction)
        denominator = _sqnorm(q)
        if bool(torch.any(denominator <= float(eps))):
            if _global_norm(adjoint_residual) <= normal_tolerance:
                converged = True
                break
            status = "numerical_failure"
            prediction = operator.forward(x).detach()
            final_residual_tensor = prediction - measurement
            return recorder.finish(
                x,
                actual_iterations=actual,
                status=status,
                stopping_reason="normal_equation_breakdown",
                final_residual=_normalized(_global_norm(final_residual_tensor), _global_norm(measurement)),
                final_objective=_objective(final_residual_tensor),
                predicted_measurement=prediction,
            )
        alpha = gamma / denominator.clamp_min(float(eps))
        previous = x
        x = x + alpha.reshape((alpha.shape[0],) + (1,) * (x.ndim - 1)) * direction
        x = apply_box_constraints(x, min_value=min_value, max_value=max_value)
        residual = measurement - operator.forward(x)
        next_adjoint = operator.adjoint(residual)
        next_gamma = _sqnorm(next_adjoint)
        change = _relative_change(previous, x)
        normal_value = _global_norm(next_adjoint)
        data_residual = -residual
        residual_value = _normalized(_global_norm(data_residual), _global_norm(measurement))
        converged = normal_tolerance > 0.0 and normal_value <= normal_tolerance
        actual += 1
        if not _safe_record(
            recorder,
            actual,
            x,
            residual=data_residual,
            objective=_objective(data_residual),
            algorithm_residual=_normalized(normal_value, initial_normal),
            stopping_candidate=converged,
            metadata={"criterion": "normal_residual", "normal_residual_absolute": normal_value},
        ):
            cancelled = True
            break
        if converged and control.stop_on_convergence:
            break
        beta = next_gamma / gamma.clamp_min(float(eps))
        direction = next_adjoint + beta.reshape((beta.shape[0],) + (1,) * (direction.ndim - 1)) * direction
        adjoint_residual, gamma = next_adjoint, next_gamma
    # Keep endpoint evaluation explicit and counted.  This makes the final
    # prediction phase distinguishable from the last Krylov update and keeps
    # resource accounting consistent with the benchmark protocol.
    prediction = _finish_prediction(operator, x)
    final_residual_tensor = prediction - measurement
    status = _finish_status(
        actual=actual,
        limit=limit,
        converged=converged,
        cancelled=cancelled,
        numerical_failure=recorder.numerical_failure,
    )
    return recorder.finish(
        x,
        actual_iterations=actual,
        status=status,
        stopping_reason=(
            "normal_residual_tolerance" if converged else
            "callback_cancelled" if cancelled else
            "maximum_iterations_reached"
        ),
        final_residual=_normalized(_global_norm(final_residual_tensor), _global_norm(measurement)),
        final_objective=_objective(final_residual_tensor),
        predicted_measurement=prediction,
        metadata={"criterion": "absolute_normal_residual", "tolerance": normal_tolerance},
    )


def solve_lsqr_detailed(
    operator: LinearOperator,
    measurement: torch.Tensor,
    *,
    num_iterations: int = 10,
    damping: float = 0.0,
    atol: float = 1e-6,
    btol: float = 1e-6,
    x_init: torch.Tensor | None = None,
    min_value: float | None = None,
    max_value: float | None = None,
    eps: float = 1e-12,
    control: SolveControl | None = None,
    callback: Callback | None = None,
) -> SolveResult:
    require_linear_operator(operator, "lsqr")
    validate_measurement_shape(measurement, operator, "lsqr")
    control = resolve_control(control, default_iterations=int(num_iterations), default_tolerance=None, callback=callback)
    limit = min(int(num_iterations), int(control.max_iterations or num_iterations))
    epsilon = float(eps)
    x = prepare_initial_image(measurement, operator, x_init=x_init, initial_value=0.0)
    recorder = IterationRecorder(control, measurement, operator, algorithm="lsqr")
    recorder.set_initial(x)
    rhs_norm = _global_norm(measurement)
    u = measurement - operator.forward(x)
    beta = _norm(u)
    u = u / beta.clamp_min(epsilon).reshape((beta.shape[0],) + (1,) * (u.ndim - 1))
    v = operator.adjoint(u)
    alpha = _norm(v)
    v = v / alpha.clamp_min(epsilon).reshape((alpha.shape[0],) + (1,) * (v.ndim - 1))
    direction = v.clone()
    phi_bar = beta.clone()
    rho_bar = alpha.clone()
    actual = 0
    converged = False
    cancelled = False
    stop_tol = float(atol) * rhs_norm + float(btol)
    residual_tensor = measurement - operator.forward(x)
    for _ in range(limit):
        residual_norm = _global_norm(residual_tensor)
        if residual_norm <= stop_tol:
            converged = True
            break
        u = operator.forward(v) - alpha.reshape((alpha.shape[0],) + (1,) * (u.ndim - 1)) * u
        beta = _norm(u)
        u = u / beta.clamp_min(epsilon).reshape((beta.shape[0],) + (1,) * (u.ndim - 1))
        v = operator.adjoint(u) - beta.reshape((beta.shape[0],) + (1,) * (v.ndim - 1)) * v
        alpha = _norm(v)
        v = v / alpha.clamp_min(epsilon).reshape((alpha.shape[0],) + (1,) * (v.ndim - 1))
        if float(damping) > 0.0:
            rho_damped = torch.sqrt(rho_bar.square() + float(damping) ** 2).clamp_min(epsilon)
            phi_bar = (rho_bar / rho_damped) * phi_bar
            rho_bar = rho_damped
        rho = torch.sqrt(rho_bar.square() + beta.square()).clamp_min(epsilon)
        cosine = rho_bar / rho
        sine = beta / rho
        theta = sine * alpha
        rho_bar = -cosine * alpha
        phi = cosine * phi_bar
        phi_bar = sine * phi_bar
        previous = x
        x = x + (phi / rho).reshape((phi.shape[0],) + (1,) * (x.ndim - 1)) * direction
        x = apply_box_constraints(x, min_value=min_value, max_value=max_value)
        direction = v - (theta / rho).reshape((theta.shape[0],) + (1,) * (direction.ndim - 1)) * direction
        actual += 1
        residual_tensor = measurement - operator.forward(x)
        residual_value = _normalized(_global_norm(residual_tensor), rhs_norm)
        change = _relative_change(previous, x)
        converged = residual_norm <= stop_tol or (
            stop_tol > 0.0 and residual_value <= _normalized(stop_tol, rhs_norm)
        )
        if not _safe_record(
            recorder,
            actual,
            x,
            residual=-residual_tensor,
            objective=_objective(residual_tensor),
            algorithm_residual=residual_value,
            stopping_candidate=converged,
            metadata={"criterion": "simplified_atol_btol"},
        ):
            cancelled = True
            break
        if converged and control.stop_on_convergence:
            break
    prediction = (measurement - residual_tensor).detach()
    final_residual_tensor = prediction - measurement
    status = _finish_status(
        actual=actual,
        limit=limit,
        converged=converged,
        cancelled=cancelled,
        numerical_failure=recorder.numerical_failure,
    )
    return recorder.finish(
        x,
        actual_iterations=actual,
        status=status,
        stopping_reason=(
            "atol_btol_tolerance" if converged else
            "callback_cancelled" if cancelled else
            "maximum_iterations_reached"
        ),
        final_residual=_normalized(_global_norm(final_residual_tensor), rhs_norm),
        final_objective=_objective(final_residual_tensor),
        predicted_measurement=prediction,
        metadata={"criterion": "simplified_atol_times_measurement_norm_plus_btol", "atol": float(atol), "btol": float(btol)},
    )


def _subset_solver_detailed(
    algorithm: str,
    operator: LinearOperator,
    measurement: torch.Tensor,
    *,
    num_iterations: int,
    block_size: int | None,
    subset_indices: Optional[Iterable[Sequence[int]]],
    order_strategy: str,
    seed: int | None,
    relaxation: float,
    min_value: float | None,
    max_value: float | None,
    x_init: torch.Tensor | None,
    eps: float,
    control: SolveControl | None,
    callback: Callback | None,
) -> SolveResult:
    require_linear_operator(operator, algorithm)
    batch = validate_measurement_shape(measurement, operator, algorithm)
    control = resolve_control(control, default_iterations=int(num_iterations), default_tolerance=1e-5, callback=callback)
    limit = min(int(num_iterations), int(control.max_iterations or num_iterations))
    x = prepare_initial_image(measurement, operator, x_init=x_init, initial_value=0.0)
    recorder = IterationRecorder(control, measurement, operator, algorithm=algorithm)
    recorder.set_initial(x)
    subsets = make_angle_subsets(
        num_angles=int(operator.range_shape[-2]),
        block_size=block_size,
        subset_indices=subset_indices,
        order_strategy=order_strategy,
        seed=seed,
        device=measurement.device,
    )
    domain = (batch, *tuple(operator.domain_shape))
    measurement_norm = _global_norm(measurement)
    actual = 0
    converged = False
    cancelled = False
    prediction = torch.zeros_like(measurement)
    for epoch in range(1, limit + 1):
        previous = x
        for index, indices in enumerate(subsets):
            sub_operator = make_subset_operator(operator, indices)
            y_sub = select_measurement_subset(measurement, indices)
            ones_range = torch.ones((batch, *tuple(sub_operator.range_shape)), device=measurement.device, dtype=measurement.dtype)
            row_weight = sub_operator.forward(torch.ones(domain, device=measurement.device, dtype=measurement.dtype))
            row_weight = torch.where(row_weight.abs() < float(eps), torch.full_like(row_weight, float("inf")), row_weight).reciprocal()
            column_weight = sub_operator.adjoint(ones_range)
            column_weight = torch.where(column_weight.abs() < float(eps), torch.full_like(column_weight, float("inf")), column_weight).reciprocal()
            residual = sub_operator.forward(x) - y_sub
            x = x - float(relaxation) * column_weight * sub_operator.adjoint(row_weight * residual)
            x = apply_box_constraints(x, min_value=min_value, max_value=max_value)
        prediction = _finish_prediction(operator, x)
        residual = prediction - measurement
        change = _relative_change(previous, x)
        residual_value = _normalized(_global_norm(residual), measurement_norm)
        converged = _tolerance_reached(control, residual=residual_value, change=change, require_both=True)
        actual = epoch
        if not _safe_record(
            recorder,
            epoch,
            x,
            residual=residual,
            objective=_objective(residual),
            algorithm_residual=residual_value,
            relaxation=float(relaxation),
            stopping_candidate=converged,
            epoch=epoch,
            subset_count=len(subsets),
            metadata={"complete_sweep": True, "subset_sizes": [int(item.numel()) for item in subsets]},
        ):
            cancelled = True
            break
        if converged and control.stop_on_convergence:
            break
    status = _finish_status(
        actual=actual,
        limit=limit,
        converged=converged,
        cancelled=cancelled,
        numerical_failure=recorder.numerical_failure,
    )
    return recorder.finish(
        x,
        actual_iterations=actual,
        status=status,
        stopping_reason=(
            "complete_sweep_residual_and_iterate_tolerance" if converged else
            "callback_cancelled" if cancelled else
            "maximum_epochs_reached"
        ),
        final_residual=_normalized(_global_norm(prediction - measurement), measurement_norm),
        final_objective=_objective(prediction - measurement),
        predicted_measurement=prediction,
        metadata={"complete_sweep_count": actual, "subset_count": len(subsets)},
    )


def solve_sart_detailed(
    operator: LinearOperator,
    measurement: torch.Tensor,
    *,
    num_iterations: int = 100,
    block_size: int = 1,
    subset_indices: Optional[Iterable[Sequence[int]]] = None,
    order_strategy: str = "ordered",
    seed: int | None = None,
    relaxation: float = 1.0,
    min_value: float | None = None,
    max_value: float | None = None,
    x_init: torch.Tensor | None = None,
    eps: float = 1e-8,
    control: SolveControl | None = None,
    callback: Callback | None = None,
) -> SolveResult:
    return _subset_solver_detailed(
        "sart", operator, measurement, num_iterations=num_iterations, block_size=block_size,
        subset_indices=subset_indices, order_strategy=order_strategy, seed=seed,
        relaxation=relaxation, min_value=min_value, max_value=max_value, x_init=x_init,
        eps=eps, control=control, callback=callback,
    )


def solve_os_sart_detailed(
    operator: LinearOperator,
    measurement: torch.Tensor,
    *,
    num_iterations: int = 100,
    block_size: int | None = None,
    subset_indices: Optional[Iterable[Sequence[int]]] = None,
    order_strategy: str = "ordered",
    seed: int | None = None,
    relaxation: float = 1.0,
    min_value: float | None = None,
    max_value: float | None = None,
    x_init: torch.Tensor | None = None,
    eps: float = 1e-8,
    control: SolveControl | None = None,
    callback: Callback | None = None,
) -> SolveResult:
    if block_size is None and subset_indices is None:
        block_size = max(int(operator.range_shape[-2]) // 10, 1)
    return _subset_solver_detailed(
        "os_sart", operator, measurement, num_iterations=num_iterations, block_size=block_size,
        subset_indices=subset_indices, order_strategy=order_strategy, seed=seed,
        relaxation=relaxation, min_value=min_value, max_value=max_value, x_init=x_init,
        eps=eps, control=control, callback=callback,
    )


def _poisson_deviance(prediction: torch.Tensor, measurement: torch.Tensor, eps: float) -> float:
    pred = prediction.clamp_min(float(eps))
    observed = measurement.clamp_min(0.0)
    value = pred - observed + torch.where(
        observed > 0.0,
        observed * torch.log((observed + float(eps)) / pred),
        torch.zeros_like(observed),
    )
    return float((2.0 * value).sum().item())


def _validate_count_data(measurement: torch.Tensor, solver: str) -> None:
    if not torch.isfinite(measurement).all():
        raise ValueError(f"{solver} requires finite observations")
    if bool(torch.any(measurement < 0.0)):
        raise ValueError(
            f"{solver} requires nonnegative emission/count observations; "
            "log-domain or signed line-integral data are incompatible"
        )


def solve_mlem_detailed(
    operator: LinearOperator,
    measurement: torch.Tensor,
    *,
    num_iterations: int = 50,
    x_init: torch.Tensor | None = None,
    initial_value: float = 1e-6,
    min_value: float = 0.0,
    max_value: float | None = None,
    eps: float = 1e-8,
    control: SolveControl | None = None,
    callback: Callback | None = None,
) -> SolveResult:
    require_linear_operator(operator, "mlem")
    validate_measurement_shape(measurement, operator, "mlem")
    _validate_count_data(measurement, "mlem")
    control = resolve_control(control, default_iterations=int(num_iterations), default_tolerance=1e-5, callback=callback)
    limit = min(int(num_iterations), int(control.max_iterations or num_iterations))
    x = prepare_initial_image(measurement, operator, x_init=x_init, initial_value=float(initial_value)).clamp_min(float(min_value))
    recorder = IterationRecorder(control, measurement, operator, algorithm="mlem")
    recorder.set_initial(x)
    sensitivity = operator.adjoint(torch.ones_like(measurement)).clamp_min(float(eps))
    measurement_norm = _global_norm(measurement)
    actual = 0
    converged = False
    cancelled = False
    prediction = torch.zeros_like(measurement)
    for epoch in range(1, limit + 1):
        previous = x
        prediction_before = operator.forward(x).clamp_min(float(eps))
        ratio = measurement / prediction_before
        correction = operator.adjoint(ratio) / sensitivity
        x = x * correction.clamp_min(0.0)
        x = apply_box_constraints(x, min_value=min_value, max_value=max_value)
        prediction = _finish_prediction(operator, x).clamp_min(float(eps))
        change = _relative_change(previous, x)
        deviance = _poisson_deviance(prediction, measurement, eps)
        residual = prediction - measurement
        converged = _tolerance_reached(control, residual=_normalized(_global_norm(residual), measurement_norm), change=change, require_both=True)
        actual = epoch
        if not _safe_record(
            recorder,
            epoch,
            x,
            residual=residual,
            objective=deviance,
            algorithm_residual=deviance,
            stopping_candidate=converged,
            epoch=epoch,
            metadata={"poisson_deviance": deviance, "complete_epoch": True},
        ):
            cancelled = True
            break
        if converged and control.stop_on_convergence:
            break
    residual = prediction - measurement
    return recorder.finish(
        x,
        actual_iterations=actual,
        status=_finish_status(
            actual=actual,
            limit=limit,
            converged=converged,
            cancelled=cancelled,
            numerical_failure=recorder.numerical_failure,
        ),
        stopping_reason=(
            "poisson_deviance_and_epoch_change_tolerance" if converged else
            "callback_cancelled" if cancelled else
            "maximum_epochs_reached"
        ),
        final_residual=_normalized(_global_norm(residual), measurement_norm),
        final_objective=_poisson_deviance(prediction, measurement, eps),
        predicted_measurement=prediction,
        metadata={"likelihood": "poisson_emission_style", "complete_epoch_count": actual},
    )


def solve_osem_detailed(
    operator: LinearOperator,
    measurement: torch.Tensor,
    *,
    num_iterations: int = 50,
    block_size: int | None = None,
    subset_indices: Optional[Iterable[Sequence[int]]] = None,
    order_strategy: str = "ordered",
    seed: int | None = None,
    x_init: torch.Tensor | None = None,
    initial_value: float = 1e-6,
    min_value: float = 0.0,
    max_value: float | None = None,
    eps: float = 1e-8,
    control: SolveControl | None = None,
    callback: Callback | None = None,
) -> SolveResult:
    require_linear_operator(operator, "osem")
    validate_measurement_shape(measurement, operator, "osem")
    _validate_count_data(measurement, "osem")
    control = resolve_control(control, default_iterations=int(num_iterations), default_tolerance=1e-5, callback=callback)
    limit = min(int(num_iterations), int(control.max_iterations or num_iterations))
    if block_size is None and subset_indices is None:
        block_size = max(int(operator.range_shape[-2]) // 10, 1)
    subsets = make_angle_subsets(
        num_angles=int(operator.range_shape[-2]), block_size=block_size,
        subset_indices=subset_indices, order_strategy=order_strategy, seed=seed, device=measurement.device,
    )
    x = prepare_initial_image(measurement, operator, x_init=x_init, initial_value=float(initial_value)).clamp_min(float(min_value))
    recorder = IterationRecorder(control, measurement, operator, algorithm="osem")
    recorder.set_initial(x)
    measurement_norm = _global_norm(measurement)
    actual = 0
    converged = False
    cancelled = False
    prediction = torch.zeros_like(measurement)
    for epoch in range(1, limit + 1):
        previous = x
        for indices in subsets:
            sub_operator = make_subset_operator(operator, indices)
            y_sub = select_measurement_subset(measurement, indices)
            sensitivity = sub_operator.adjoint(torch.ones_like(y_sub)).clamp_min(float(eps))
            prediction_sub = sub_operator.forward(x).clamp_min(float(eps))
            ratio = y_sub / prediction_sub
            correction = sub_operator.adjoint(ratio) / sensitivity
            x = x * correction.clamp_min(0.0)
            x = apply_box_constraints(x, min_value=min_value, max_value=max_value)
        prediction = _finish_prediction(operator, x).clamp_min(float(eps))
        residual = prediction - measurement
        change = _relative_change(previous, x)
        deviance = _poisson_deviance(prediction, measurement, eps)
        converged = _tolerance_reached(control, residual=_normalized(_global_norm(residual), measurement_norm), change=change, require_both=True)
        actual = epoch
        if not _safe_record(
            recorder,
            epoch,
            x,
            residual=residual,
            objective=deviance,
            algorithm_residual=deviance,
            stopping_candidate=converged,
            epoch=epoch,
            subset_count=len(subsets),
            metadata={"poisson_deviance": deviance, "complete_epoch": True, "subset_sizes": [int(item.numel()) for item in subsets]},
        ):
            cancelled = True
            break
        if converged and control.stop_on_convergence:
            break
    residual = prediction - measurement
    return recorder.finish(
        x,
        actual_iterations=actual,
        status=_finish_status(
            actual=actual,
            limit=limit,
            converged=converged,
            cancelled=cancelled,
            numerical_failure=recorder.numerical_failure,
        ),
        stopping_reason=(
            "poisson_deviance_and_epoch_change_tolerance" if converged else
            "callback_cancelled" if cancelled else
            "maximum_epochs_reached"
        ),
        final_residual=_normalized(_global_norm(residual), measurement_norm),
        final_objective=_poisson_deviance(prediction, measurement, eps),
        predicted_measurement=prediction,
        metadata={"likelihood": "poisson_emission_style", "complete_epoch_count": actual, "subset_count": len(subsets)},
    )


def _estimate_lipschitz_for_detail(operator: LinearOperator, reference: torch.Tensor, iterations: int) -> float:
    from .regularized import _estimate_lipschitz

    return float(_estimate_lipschitz(operator, reference, int(iterations)))


def solve_tikhonov_detailed(
    operator: LinearOperator,
    measurement: torch.Tensor,
    *,
    reg_strength: float = 1e-2,
    num_iterations: int = 100,
    tolerance: float = 1e-6,
    x_init: torch.Tensor | None = None,
    regularization_operator: LinearOperator | None = None,
    min_value: float | None = None,
    max_value: float | None = None,
    eps: float = 1e-12,
    control: SolveControl | None = None,
    callback: Callback | None = None,
) -> SolveResult:
    require_linear_operator(operator, "tikhonov")
    validate_measurement_shape(measurement, operator, "tikhonov")
    if regularization_operator is not None and not isinstance(regularization_operator, LinearOperator):
        raise TypeError("regularization_operator must be a LinearOperator or None")
    if float(reg_strength) < 0.0:
        raise ValueError("reg_strength must be nonnegative")
    control = resolve_control(control, default_iterations=int(num_iterations), default_tolerance=float(tolerance), callback=callback)
    limit = min(int(num_iterations), int(control.max_iterations or num_iterations))
    regularizer = TikhonovRegularizer(regularization_operator)
    strength = float(reg_strength)
    x = prepare_initial_image(measurement, operator, x_init=x_init, initial_value=0.0)
    recorder = IterationRecorder(control, measurement, operator, algorithm="tikhonov")
    recorder.set_initial(x)
    # Cache A*x and A*d.  The cached predictions keep the per-iteration
    # objective exact while avoiding a second forward call solely for
    # diagnostics.  A final explicit endpoint call remains part of the
    # accounting contract.  Box constraints invalidate the linear update,
    # so that less common path recomputes the prediction safely.
    cache_prediction = min_value is None and max_value is None
    prediction = operator.forward(x)
    rhs = operator.adjoint(measurement)
    normal_residual = rhs - (operator.adjoint(prediction) + strength * regularizer.gradient(x))
    direction = normal_residual.clone()
    residual_sq = _sqnorm(normal_residual)
    rhs_norm = max(_global_norm(rhs), 1.0)
    actual = 0
    converged = False
    cancelled = False
    for iteration in range(1, limit + 1):
        normal_value = _global_norm(normal_residual)
        if float(control.tolerance or 0.0) > 0.0 and _normalized(normal_value, rhs_norm) <= float(control.tolerance):
            converged = True
            break
        direction_prediction = operator.forward(direction)
        normal_direction = operator.adjoint(direction_prediction) + strength * regularizer.gradient(direction)
        denominator = (direction * normal_direction).reshape(direction.shape[0], -1).sum(dim=1)
        if bool(torch.any(denominator.abs() <= float(eps))):
            # Conjugate-gradient breakdown after the normal residual has
            # already reached zero is a harmless exact-solution plateau.  In
            # fixed-compute mode keep the remaining counted iterations and
            # report the configured budget stop; a nonzero residual remains a
            # genuine numerical failure.
            # Floating point CG can leave a tiny nonzero normal residual even
            # for an exact solution (for example, identity operators with a
            # positive Tikhonov term).  Treat that machine-precision plateau
            # as a benign breakdown, but keep a materially nonzero residual
            # as a numerical failure.
            if _global_norm(normal_residual) <= max(float(eps), 1e-6 * rhs_norm):
                prediction = _finish_prediction(operator, x) if not cache_prediction else prediction
                residual = prediction - measurement
                actual = iteration
                if not _safe_record(
                    recorder,
                    iteration,
                    x,
                    residual=residual,
                    objective=_objective(residual) + strength * float(regularizer.value(x).sum().item()),
                    algorithm_residual=0.0,
                    stopping_candidate=False,
                    metadata={"normal_residual": 0.0, "breakdown": "exact_solution_plateau"},
                ):
                    cancelled = True
                    break
                continue
            prediction = _finish_prediction(operator, x)
            residual = prediction - measurement
            return recorder.finish(
                x,
                actual_iterations=actual,
                status="numerical_failure",
                stopping_reason="normal_equation_breakdown",
                final_residual=_normalized(_global_norm(residual), _global_norm(measurement)),
                final_objective=_objective(residual) + strength * float(regularizer.value(x).sum().item()),
                predicted_measurement=prediction,
            )
        alpha = residual_sq / denominator
        previous = x
        x = x + alpha.reshape((alpha.shape[0],) + (1,) * (x.ndim - 1)) * direction
        x = apply_box_constraints(x, min_value=min_value, max_value=max_value)
        next_normal_residual = normal_residual - alpha.reshape((alpha.shape[0],) + (1,) * (normal_residual.ndim - 1)) * normal_direction
        next_residual_sq = _sqnorm(next_normal_residual)
        beta = next_residual_sq / residual_sq.clamp_min(float(eps))
        direction = next_normal_residual + beta.reshape((beta.shape[0],) + (1,) * (direction.ndim - 1)) * direction
        normal_residual = next_normal_residual
        residual_sq = next_residual_sq
        if cache_prediction:
            prediction = prediction + alpha.reshape((alpha.shape[0],) + (1,) * (prediction.ndim - 1)) * direction_prediction
        else:
            prediction = _finish_prediction(operator, x)
        residual = prediction - measurement
        objective = _objective(residual) + strength * float(regularizer.value(x).sum().item())
        normal_value = _global_norm(normal_residual)
        normal_relative = _normalized(normal_value, rhs_norm)
        change = _relative_change(previous, x)
        converged = float(control.tolerance or 0.0) > 0.0 and _tolerance_reached(
            control, residual=normal_relative, change=change, require_both=True
        )
        actual = iteration
        if not _safe_record(
            recorder,
            iteration,
            x,
            residual=residual,
            objective=objective,
            algorithm_residual=normal_relative,
            stopping_candidate=converged,
            metadata={"normal_residual": normal_value, "complete_objective": True},
        ):
            cancelled = True
            break
        if converged and control.stop_on_convergence:
            break
    # Keep the final evaluation explicit, even though the prediction above is
    # already exact in the unconstrained linear path.  This distinguishes the
    # solver trajectory from the endpoint diagnostics and gives the equal-call
    # protocol its advertised n+2 forward / n+2 adjoint shape.
    prediction = _finish_prediction(operator, x)
    residual = prediction - measurement
    return recorder.finish(
        x,
        actual_iterations=actual,
        status=_finish_status(
            actual=actual,
            limit=limit,
            converged=converged,
            cancelled=cancelled,
            numerical_failure=recorder.numerical_failure,
        ),
        stopping_reason=(
            "normal_residual_and_iterate_tolerance" if converged else
            "callback_cancelled" if cancelled else
            "maximum_iterations_reached"
        ),
        final_residual=_normalized(_global_norm(residual), _global_norm(measurement)),
        final_objective=_objective(residual) + strength * float(regularizer.value(x).sum().item()),
        predicted_measurement=prediction,
        metadata={"criterion": "normalized_normal_residual_and_relative_iterate_change", "regularization_strength": strength},
    )


def solve_tv_fista_detailed(
    operator: LinearOperator,
    measurement: torch.Tensor,
    *,
    reg_strength: float = 1e-3,
    num_iterations: int = 50,
    step_size: float | None = None,
    x_init: torch.Tensor | None = None,
    regularizer: TVRegularizer | None = None,
    tolerance: float = 1e-5,
    power_iterations: int = 12,
    min_value: float | None = None,
    max_value: float | None = None,
    control: SolveControl | None = None,
    callback: Callback | None = None,
) -> SolveResult:
    require_linear_operator(operator, "tv_fista")
    validate_measurement_shape(measurement, operator, "tv_fista")
    if float(reg_strength) < 0.0:
        raise ValueError("reg_strength must be nonnegative")
    tv = regularizer or TVRegularizer()
    if not isinstance(tv, TVRegularizer):
        raise TypeError("regularizer must be a TVRegularizer")
    control = resolve_control(control, default_iterations=int(num_iterations), default_tolerance=float(tolerance), callback=callback)
    limit = min(int(num_iterations), int(control.max_iterations or num_iterations))
    lipschitz = None
    if step_size is None:
        lipschitz = _estimate_lipschitz_for_detail(operator, measurement, int(power_iterations))
        step = 1.0 if lipschitz <= 1e-12 else 0.99 / lipschitz
    else:
        step = float(step_size)
        if step <= 0.0:
            raise ValueError("step_size must be positive")
    x = prepare_initial_image(measurement, operator, x_init=x_init, initial_value=0.0)
    momentum = x.clone()
    acceleration = 1.0
    recorder = IterationRecorder(control, measurement, operator, algorithm="tv_fista")
    recorder.set_initial(x)
    measurement_norm = _global_norm(measurement)
    actual = 0
    converged = False
    cancelled = False
    prediction = torch.zeros_like(measurement)
    for iteration in range(1, limit + 1):
        residual_at_momentum = operator.forward(momentum) - measurement
        gradient = operator.adjoint(residual_at_momentum)
        candidate = momentum - step * gradient
        next_x = tv.proximal(candidate, step * float(reg_strength))
        next_x = apply_box_constraints(next_x, min_value=min_value, max_value=max_value)
        next_acceleration = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * acceleration * acceleration))
        momentum_scale = (acceleration - 1.0) / next_acceleration
        next_momentum = next_x + momentum_scale * (next_x - x)
        # The forward evaluation is the objective at the momentum state used
        # by this FISTA update.  Record that state; a second forward call for
        # next_x would double the per-iteration cost and break the declared
        # equal-call protocol.  The returned reconstruction still receives a
        # separate, explicit endpoint evaluation below.
        prediction = residual_at_momentum + measurement
        residual = residual_at_momentum
        objective = _objective(residual) + float(reg_strength) * float(tv.value(momentum).sum().item())
        change = _relative_change(momentum, next_momentum)
        converged = _tolerance_reached(
            control,
            residual=_normalized(_global_norm(residual), measurement_norm),
            change=change,
            require_both=True,
        )
        actual = iteration
        if not _safe_record(
            recorder,
            iteration,
            momentum,
            residual=residual,
            objective=objective,
            algorithm_residual=_normalized(_global_norm(residual), measurement_norm),
            step_size=step,
            stopping_candidate=converged,
            metadata={"complete_objective": True, "objective_state": "momentum", "fista_momentum": float(momentum_scale)},
        ):
            cancelled = True
            x = next_x
            break
        x = next_x
        momentum = next_momentum
        acceleration = next_acceleration
        if converged and control.stop_on_convergence:
            break
    prediction = _finish_prediction(operator, x)
    residual = prediction - measurement
    resources = {"prox_iterations": int(actual) * int(tv.num_iterations), "prox_configured_iterations": int(tv.num_iterations)}
    return recorder.finish(
        x,
        actual_iterations=actual,
        status=_finish_status(
            actual=actual,
            limit=limit,
            converged=converged,
            cancelled=cancelled,
            numerical_failure=recorder.numerical_failure,
        ),
        stopping_reason=(
            "complete_objective_and_iterate_tolerance" if converged else
            "callback_cancelled" if cancelled else
            "maximum_iterations_reached"
        ),
        final_residual=_normalized(_global_norm(residual), measurement_norm),
        final_objective=_objective(residual) + float(reg_strength) * float(tv.value(x).sum().item()),
        predicted_measurement=prediction,
        resources=resources,
        metadata={
            "criterion": "complete_tv_objective_and_relative_iterate_change",
            "step_size": step,
            "operator_norm_squared": lipschitz,
            "power_iterations": int(power_iterations) if step_size is None else 0,
            "tv_mode": tv.mode,
            "tv_tolerance": tv.tolerance,
        },
    )

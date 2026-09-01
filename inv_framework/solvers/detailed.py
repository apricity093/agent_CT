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
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import torch

from ..operators.base import ForwardOperator, LinearOperator
from ..regularizers import TikhonovRegularizer, TVRegularizer
from .base import (
    ConsecutiveStoppingMonitor,
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


def _policy_active(control: SolveControl) -> bool:
    return control.discrepancy_target is not None


def _policy_status(decision: Any) -> tuple[str | None, str | None]:
    if decision.diverged:
        return "diverged", "persistent_residual_or_objective_increase"
    if decision.stalled:
        return "stalled", "stalled_before_discrepancy"
    return None, None


def _finite_tensor(value: Any) -> bool:
    return isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all().item())


def _finite_scalar(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _operator_stats(operator: Any) -> dict[str, Any]:
    stats = getattr(operator, "stats", None)
    if not callable(stats):
        return {}
    return dict(stats() or {})


def _operator_call_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, int]:
    delta: dict[str, int] = {}
    for name in ("forward_calls", "adjoint_calls", "total_operator_calls"):
        if before.get(name) is not None and after.get(name) is not None:
            delta[name] = int(after[name]) - int(before[name])
    return delta


def _common_parameter_errors(
    num_iterations: Any,
    min_value: Any,
    max_value: Any,
) -> list[str]:
    errors: list[str] = []
    if isinstance(num_iterations, bool) or not isinstance(num_iterations, int) or num_iterations <= 0:
        errors.append("num_iterations must be a positive integer")
    for name, value in (("min_value", min_value), ("max_value", max_value)):
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not _finite_scalar(value)
        ):
            errors.append(f"{name} must be a finite number or None")
    if (
        min_value is not None
        and max_value is not None
        and _finite_scalar(min_value)
        and _finite_scalar(max_value)
        and float(min_value) > float(max_value)
    ):
        errors.append("min_value must be less than or equal to max_value")
    return errors


def _parameter_failure_result(
    algorithm: str,
    measurement: torch.Tensor,
    operator: LinearOperator,
    *,
    x_init: torch.Tensor | None,
    errors: Iterable[str],
    max_iterations: int | None,
    parameters: Mapping[str, Any],
    resources: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SolveResult:
    """Return a structured no-loop result for direct detailed callers."""

    x = prepare_initial_image(measurement, operator, x_init=x_init, initial_value=0.0)
    return SolveResult(
        reconstruction=x,
        actual_iterations=0,
        status="invalid_parameters",
        stopping_reason="parameter_validation_failed",
        resources={**_operator_stats(operator), **dict(resources or {})},
        metadata={
            "algorithm": algorithm,
            "max_iterations": max_iterations,
            "validation_errors": [str(error) for error in errors],
            "parameters": dict(parameters),
            **dict(metadata or {}),
        },
    )


def _numerical_result(
    algorithm: str,
    measurement: torch.Tensor,
    operator: LinearOperator,
    *,
    x_init: torch.Tensor | None,
    max_iterations: int | None,
    reason: str,
    parameters: Mapping[str, Any],
) -> SolveResult:
    """Return a structured pre-loop numerical failure without hiding it."""

    x = prepare_initial_image(measurement, operator, x_init=x_init, initial_value=0.0)
    return SolveResult(
        reconstruction=x,
        actual_iterations=0,
        status="numerical_error",
        stopping_reason=reason,
        resources=_operator_stats(operator),
        metadata={
            "algorithm": algorithm,
            "max_iterations": max_iterations,
            "parameters": dict(parameters),
        },
    )


def _native_tolerance(control: SolveControl) -> float | None:
    if control.relative_iterate_tolerance is not None:
        return float(control.relative_iterate_tolerance)
    # A policy is still meaningful for direct detailed callers that only set
    # the discrepancy target.  Keep the same documented native threshold as
    # the runtime policy instead of silently making the second criterion
    # impossible to satisfy.
    return 1e-4 if control.discrepancy_target is not None else None


def _row_action_criteria(
    control: SolveControl,
    residual_value: float,
    change: float,
    *,
    epoch: bool = False,
) -> tuple[dict[str, bool], float | None]:
    if control.discrepancy_target is None:
        return {}, _native_tolerance(control)
    native_tolerance = _native_tolerance(control)
    assert native_tolerance is not None
    native_name = "relative_epoch_change" if epoch else "relative_iterate_change"
    return {
        "discrepancy": residual_value <= float(control.discrepancy_target),
        native_name: change <= native_tolerance,
    }, native_tolerance


def _known_operator_norm_squared(control: SolveControl) -> float | None:
    metadata = dict(control.metadata or {})
    if metadata.get("operator_norm_squared") is not None:
        value = float(metadata["operator_norm_squared"])
    elif metadata.get("operator_norm_estimate") is not None:
        value = float(metadata["operator_norm_estimate"]) ** 2
    else:
        return None
    return value if _finite_scalar(value) and value >= 0.0 else float("nan")


def _estimate_operator_norm_squared(
    operator: LinearOperator,
    reference: torch.Tensor,
    *,
    iterations: int = 8,
) -> float:
    """Deterministically estimate ``||A||^2`` for Landweber preflight."""

    domain_shape = tuple(int(value) for value in operator.domain_shape)
    sample_size = math.prod(domain_shape)
    seed = torch.linspace(
        0.5,
        1.5,
        steps=sample_size,
        dtype=reference.dtype,
        device=reference.device,
    ).reshape((1, *domain_shape))
    denominator = seed.reshape(seed.shape[0], -1).norm(dim=1).clamp_min(1e-12)
    vector = seed / denominator.reshape((1,) + (1,) * len(domain_shape))
    for _ in range(max(1, int(iterations))):
        next_vector = operator.adjoint(operator.forward(vector))
        if not _finite_tensor(next_vector):
            return float("nan")
        norm = next_vector.reshape(next_vector.shape[0], -1).norm(dim=1)
        if bool(torch.all(norm <= 1e-12)):
            return 0.0
        vector = next_vector / norm.reshape((next_vector.shape[0],) + (1,) * len(domain_shape)).clamp_min(1e-12)
    image = operator.adjoint(operator.forward(vector))
    if not _finite_tensor(image):
        return float("nan")
    value = (vector * image).reshape(vector.shape[0], -1).sum(dim=1).clamp_min(0.0).max()
    return float(value.item())


def _landweber_step_preflight(
    operator: LinearOperator,
    measurement: torch.Tensor,
    control: SolveControl,
    step_size: Any,
) -> tuple[float, float, list[str], dict[str, Any]]:
    errors: list[str] = []
    if isinstance(step_size, bool) or not isinstance(step_size, (int, float)):
        errors.append("step_size must be a finite positive number")
        return 0.0, float("nan"), errors, {}
    step = float(step_size)
    if not _finite_scalar(step) or step <= 0.0:
        errors.append("step_size must be a finite positive number")
        return step, float("nan"), errors, {}
    norm_squared = _known_operator_norm_squared(control)
    estimates: dict[str, Any] = {}
    if norm_squared is None:
        power_iterations = dict(control.metadata or {}).get("power_iterations", 8)
        try:
            power_iterations = int(power_iterations)
        except (TypeError, ValueError):
            power_iterations = 8
        if power_iterations <= 0:
            power_iterations = 8
        norm_squared = _estimate_operator_norm_squared(
            operator, measurement, iterations=power_iterations
        )
        estimates.update({
            "operator_norm_squared": norm_squared,
            "operator_norm_estimator": f"power_iteration_{power_iterations}",
        })
    else:
        estimates.update({
            "operator_norm_squared": norm_squared,
            "operator_norm_estimator": "control_metadata",
        })
    if not _finite_scalar(norm_squared) or norm_squared < 0.0:
        errors.append("operator norm estimate must be finite and nonnegative")
    elif norm_squared > 0.0:
        upper = 2.0 / float(norm_squared)
        if not step < upper:
            errors.append(
                "step_size must satisfy 0 < step_size < 2 / ||A||^2 "
                f"({upper:.6g})"
            )
    return step, float(norm_squared), errors, estimates


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
        status="not_applicable",
        stopping_reason="direct_reconstruction",
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
        status="not_applicable",
        stopping_reason="direct_reconstruction",
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
    parameter_errors = _common_parameter_errors(num_iterations, min_value, max_value)
    if parameter_errors:
        return _parameter_failure_result(
            "sirt", measurement, operator, x_init=x_init, errors=parameter_errors,
            max_iterations=None,
            parameters={
                "num_iterations": num_iterations,
                "min_value": min_value,
                "max_value": max_value,
            },
        )
    control = resolve_control(
        control, default_iterations=int(num_iterations), default_tolerance=1e-5, callback=callback
    )
    limit = min(int(num_iterations), int(control.max_iterations or num_iterations))
    parameters = {
        "num_iterations": int(num_iterations),
        "min_value": min_value,
        "max_value": max_value,
    }
    if not _finite_tensor(measurement):
        return _numerical_result(
            "sirt", measurement, operator, x_init=x_init, max_iterations=limit,
            reason="non_finite_measurement", parameters=parameters,
        )
    x = prepare_initial_image(measurement, operator, x_init=x_init, initial_value=0.0)
    if not _finite_tensor(x):
        return _numerical_result(
            "sirt", measurement, operator, x_init=x_init, max_iterations=limit,
            reason="non_finite_initial_reconstruction", parameters=parameters,
        )
    recorder = IterationRecorder(control, measurement, operator, algorithm="sirt")
    monitor = ConsecutiveStoppingMonitor(control)
    recorder.set_initial(x)
    domain = (batch, *tuple(operator.domain_shape))
    range_ = (batch, *tuple(operator.range_shape))
    optimization_before = _operator_stats(operator)
    row_weight = operator.forward(torch.ones(domain, device=measurement.device, dtype=measurement.dtype))
    row_weight = torch.where(row_weight < 1e-8, torch.full_like(row_weight, float("inf")), row_weight).reciprocal()
    if not _finite_tensor(row_weight):
        recorder.mark_numerical_error("non_finite_row_normalization")
        return recorder.finish(
            x, actual_iterations=0, status="numerical_error",
            stopping_reason="non_finite_row_normalization",
            resources={"optimization_calls": _operator_call_delta(optimization_before, _operator_stats(operator))},
            metadata={"criterion": "normalized_data_residual_and_relative_iterate_change", **parameters},
        )
    column_weight = operator.adjoint(torch.ones(range_, device=measurement.device, dtype=measurement.dtype))
    column_weight = torch.where(column_weight < 1e-8, torch.full_like(column_weight, float("inf")), column_weight).reciprocal()
    if not _finite_tensor(column_weight):
        recorder.mark_numerical_error("non_finite_column_normalization")
        return recorder.finish(
            x, actual_iterations=0, status="numerical_error",
            stopping_reason="non_finite_column_normalization",
            resources={"optimization_calls": _operator_call_delta(optimization_before, _operator_stats(operator))},
            metadata={"criterion": "normalized_data_residual_and_relative_iterate_change", **parameters},
        )
    measurement_norm = _global_norm(measurement)
    actual = 0
    converged = False
    cancelled = False
    terminal_status = None
    terminal_reason = None
    prediction = operator.forward(x)
    residual = prediction - measurement
    if not _finite_tensor(prediction) or not _finite_tensor(residual):
        recorder.mark_numerical_error("non_finite_initial_residual")
        return recorder.finish(
            x, actual_iterations=0, status="numerical_error",
            stopping_reason="non_finite_initial_residual",
            predicted_measurement=prediction,
            resources={"optimization_calls": _operator_call_delta(optimization_before, _operator_stats(operator))},
            metadata={"criterion": "normalized_data_residual_and_relative_iterate_change", **parameters},
        )
    native_tolerance = _native_tolerance(control)
    for iteration in range(1, limit + 1):
        previous = x
        x = x - column_weight * operator.adjoint(row_weight * residual)
        x = apply_box_constraints(x, min_value=min_value, max_value=max_value)
        actual = iteration
        if not _finite_tensor(x):
            recorder.mark_numerical_error("non_finite_reconstruction")
            recorder.record(
                iteration, x, metadata={"state": "post_update", "checked": False, "non_finite_field": "reconstruction"}
            )
            break
        prediction = operator.forward(x)
        residual = prediction - measurement
        if not _finite_tensor(prediction):
            recorder.mark_numerical_error("non_finite_prediction")
            recorder.record(
                iteration, x, residual=residual,
                metadata={"state": "post_update", "checked": False, "non_finite_field": "prediction"}
            )
            break
        if not _finite_tensor(residual):
            recorder.mark_numerical_error("non_finite_residual")
            recorder.record(
                iteration, x, residual=residual,
                metadata={"state": "post_update", "checked": False, "non_finite_field": "residual"}
            )
            break
        change = _relative_change(previous, x)
        residual_value = _normalized(_global_norm(residual), measurement_norm)
        if not _finite_scalar(change):
            recorder.mark_numerical_error("non_finite_relative_iterate_change")
            recorder.record(
                iteration, x, residual=residual,
                metadata={"state": "post_update", "checked": False, "non_finite_field": "relative_iterate_change"}
            )
            break
        if not _finite_scalar(residual_value):
            recorder.mark_numerical_error("non_finite_discrepancy")
            recorder.record(
                iteration, x, residual=residual,
                metadata={"state": "post_update", "checked": False, "non_finite_field": "discrepancy"}
            )
            break
        criteria, native_tolerance = _row_action_criteria(
            control, residual_value, change, epoch=False
        )
        decision = monitor.observe(iteration, criteria=criteria, relative_change=change, monitor_value=residual_value)
        converged = decision.converged if _policy_active(control) else _tolerance_reached(control, residual=residual_value, change=change)
        terminal_status, terminal_reason = _policy_status(decision) if _policy_active(control) else (None, None)
        objective = _objective(residual)
        if not _finite_scalar(objective):
            recorder.mark_numerical_error("non_finite_objective")
            recorder.record(
                iteration, x, residual=residual,
                metadata={"state": "post_update", "checked": decision.checked, "non_finite_field": "objective"}
            )
            break
        if not _safe_record(
            recorder,
            iteration,
            x,
            residual=residual,
            objective=objective,
            algorithm_residual=residual_value,
            stopping_candidate=converged,
            consecutive_criteria_count=decision.consecutive,
            criteria=criteria,
            native_criterion_name="relative_iterate_change",
            native_criterion_value=change,
            native_criterion_threshold=native_tolerance,
            metadata={"state": "post_update", "checked": decision.checked, "unit": "iterations"},
        ):
            cancelled = True
            break
        if terminal_status or (converged and control.stop_on_convergence):
            break
    if recorder.numerical_failure:
        return recorder.finish(
            x,
            actual_iterations=actual,
            status="numerical_error",
            stopping_reason=recorder.numerical_error_reason or "non_finite_solver_state",
            predicted_measurement=prediction,
            metadata={
                "criterion": "normalized_data_residual_and_relative_iterate_change",
                "iteration_unit": "iterations",
                **parameters,
            },
        )
    optimization_after = _operator_stats(operator)
    endpoint_prediction = prediction
    verification_calls: dict[str, int] = {}
    endpoint_confirmation: dict[str, Any] = {"requested": False}
    if _policy_active(control) and not cancelled:
        endpoint_before = _operator_stats(operator)
        endpoint_prediction = _finish_prediction(operator, x)
        verification_calls = _operator_call_delta(endpoint_before, _operator_stats(operator))
        if not _finite_tensor(endpoint_prediction):
            recorder.mark_numerical_error("non_finite_endpoint_prediction")
        else:
            endpoint_residual = endpoint_prediction - measurement
            endpoint_value = _normalized(_global_norm(endpoint_residual), measurement_norm)
            trajectory_value = _normalized(_global_norm(prediction - measurement), measurement_norm)
            endpoint_confirmation = {
                "requested": True,
                "independent": True,
                "trajectory_normalized_data_residual": trajectory_value,
                "endpoint_normalized_data_residual": endpoint_value,
                "trajectory_endpoint_consistent": (
                    abs(trajectory_value - endpoint_value)
                    <= 1e-7 + 1e-4 * max(abs(trajectory_value), abs(endpoint_value))
                ),
            }
    if recorder.numerical_failure:
        return recorder.finish(
            x,
            actual_iterations=actual,
            status="numerical_error",
            stopping_reason=recorder.numerical_error_reason or "non_finite_solver_state",
            predicted_measurement=endpoint_prediction,
            resources={
                "optimization_" + key: value for key, value in _operator_call_delta(optimization_before, optimization_after).items()
            } | {"verification_" + key: value for key, value in verification_calls.items()},
            metadata={
                "criterion": "normalized_data_residual_and_relative_iterate_change",
                "endpoint_confirmation": endpoint_confirmation,
                "iteration_unit": "iterations",
                **parameters,
            },
        )
    final_residual_tensor = endpoint_prediction - measurement
    final_residual = _normalized(_global_norm(final_residual_tensor), measurement_norm)
    final_objective = _objective(final_residual_tensor)
    status = terminal_status or _finish_status(
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
            terminal_reason if terminal_reason else
            "discrepancy_and_relative_iterate_change_patience" if converged and _policy_active(control) else
            "relative_residual_and_iterate_tolerance" if converged else
            "callback_cancelled" if cancelled else
            "maximum_iterations_reached"
        ),
        final_residual=final_residual,
        final_objective=final_objective,
        predicted_measurement=endpoint_prediction,
        resources={
            "optimization_" + key: value for key, value in _operator_call_delta(optimization_before, optimization_after).items()
        } | {"verification_" + key: value for key, value in verification_calls.items()},
        metadata={
            "criterion": "normalized_data_residual_and_relative_iterate_change",
            "iteration_unit": "iterations",
            "endpoint_confirmation": endpoint_confirmation,
            "max_iterations": limit,
            **parameters,
        },
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
    parameter_errors = _common_parameter_errors(num_iterations, min_value, max_value)
    if parameter_errors:
        return _parameter_failure_result(
            "landweber", measurement, operator, x_init=x_init, errors=parameter_errors,
            max_iterations=None,
            parameters={
                "num_iterations": num_iterations,
                "step_size": step_size,
                "min_value": min_value,
                "max_value": max_value,
            },
        )
    control = resolve_control(
        control, default_iterations=int(num_iterations), default_tolerance=1e-5, callback=callback
    )
    limit = min(int(num_iterations), int(control.max_iterations or num_iterations))
    parameters = {
        "num_iterations": int(num_iterations),
        "step_size": 1e-3 if step_size is None else step_size,
        "min_value": min_value,
        "max_value": max_value,
    }
    if not _finite_tensor(measurement):
        return _numerical_result(
            "landweber", measurement, operator, x_init=x_init, max_iterations=limit,
            reason="non_finite_measurement", parameters=parameters,
        )
    x = prepare_initial_image(measurement, operator, x_init=x_init, initial_value=0.0)
    if not _finite_tensor(x):
        return _numerical_result(
            "landweber", measurement, operator, x_init=x_init, max_iterations=limit,
            reason="non_finite_initial_reconstruction", parameters=parameters,
        )
    step, norm_squared, step_errors, norm_estimates = _landweber_step_preflight(
        operator, measurement, control, parameters["step_size"]
    )
    if step_errors:
        return _parameter_failure_result(
            "landweber", measurement, operator, x_init=x_init, errors=step_errors,
            max_iterations=limit,
            parameters={**parameters, "step_size": step},
            metadata={"operator_norm_squared": norm_squared, **norm_estimates},
        )
    parameters["step_size"] = step
    x = prepare_initial_image(measurement, operator, x_init=x_init, initial_value=0.0)
    recorder = IterationRecorder(control, measurement, operator, algorithm="landweber")
    monitor = ConsecutiveStoppingMonitor(control)
    recorder.set_initial(x)
    optimization_before = _operator_stats(operator)
    measurement_norm = _global_norm(measurement)
    actual = 0
    converged = False
    cancelled = False
    terminal_status = None
    terminal_reason = None
    prediction = operator.forward(x)
    residual = prediction - measurement
    if not _finite_tensor(prediction) or not _finite_tensor(residual):
        recorder.mark_numerical_error("non_finite_initial_residual")
        return recorder.finish(
            x, actual_iterations=0, status="numerical_error",
            stopping_reason="non_finite_initial_residual",
            predicted_measurement=prediction,
            resources={"parameter_estimation": norm_estimates},
            metadata={"criterion": "normalized_data_residual_and_relative_iterate_change", **parameters, **norm_estimates},
        )
    native_tolerance = _native_tolerance(control)
    for iteration in range(1, limit + 1):
        previous = x
        x = x - step * operator.adjoint(residual)
        x = apply_box_constraints(x, min_value=min_value, max_value=max_value)
        actual = iteration
        if not _finite_tensor(x):
            recorder.mark_numerical_error("non_finite_reconstruction")
            recorder.record(
                iteration, x, metadata={"state": "post_update", "checked": False, "non_finite_field": "reconstruction"}
            )
            break
        prediction = operator.forward(x)
        residual = prediction - measurement
        if not _finite_tensor(prediction):
            recorder.mark_numerical_error("non_finite_prediction")
            recorder.record(
                iteration, x, residual=residual,
                metadata={"state": "post_update", "checked": False, "non_finite_field": "prediction"}
            )
            break
        if not _finite_tensor(residual):
            recorder.mark_numerical_error("non_finite_residual")
            recorder.record(
                iteration, x, residual=residual,
                metadata={"state": "post_update", "checked": False, "non_finite_field": "residual"}
            )
            break
        change = _relative_change(previous, x)
        residual_value = _normalized(_global_norm(residual), measurement_norm)
        if not _finite_scalar(change):
            recorder.mark_numerical_error("non_finite_relative_iterate_change")
            recorder.record(
                iteration, x, residual=residual,
                metadata={"state": "post_update", "checked": False, "non_finite_field": "relative_iterate_change"}
            )
            break
        if not _finite_scalar(residual_value):
            recorder.mark_numerical_error("non_finite_discrepancy")
            recorder.record(
                iteration, x, residual=residual,
                metadata={"state": "post_update", "checked": False, "non_finite_field": "discrepancy"}
            )
            break
        criteria, native_tolerance = _row_action_criteria(
            control, residual_value, change, epoch=False
        )
        decision = monitor.observe(iteration, criteria=criteria, relative_change=change, monitor_value=residual_value)
        converged = decision.converged if _policy_active(control) else _tolerance_reached(control, residual=residual_value, change=change)
        terminal_status, terminal_reason = _policy_status(decision) if _policy_active(control) else (None, None)
        objective = _objective(residual)
        if not _finite_scalar(objective):
            recorder.mark_numerical_error("non_finite_objective")
            recorder.record(
                iteration, x, residual=residual,
                metadata={"state": "post_update", "checked": decision.checked, "non_finite_field": "objective"}
            )
            break
        if not _safe_record(
            recorder,
            iteration,
            x,
            residual=residual,
            objective=objective,
            algorithm_residual=residual_value,
            step_size=step,
            stopping_candidate=converged,
            consecutive_criteria_count=decision.consecutive,
            criteria=criteria,
            native_criterion_name="relative_iterate_change",
            native_criterion_value=change,
            native_criterion_threshold=native_tolerance,
            metadata={"state": "post_update", "checked": decision.checked, "unit": "iterations"},
        ):
            cancelled = True
            break
        if terminal_status or (converged and control.stop_on_convergence):
            break
    if recorder.numerical_failure:
        return recorder.finish(
            x,
            actual_iterations=actual,
            status="numerical_error",
            stopping_reason=recorder.numerical_error_reason or "non_finite_solver_state",
            predicted_measurement=prediction,
            metadata={"criterion": "normalized_data_residual_and_relative_iterate_change", **parameters, **norm_estimates},
        )
    optimization_after = _operator_stats(operator)
    endpoint_prediction = prediction
    verification_calls: dict[str, int] = {}
    endpoint_confirmation: dict[str, Any] = {"requested": False}
    if _policy_active(control) and not cancelled:
        endpoint_before = _operator_stats(operator)
        endpoint_prediction = _finish_prediction(operator, x)
        verification_calls = _operator_call_delta(endpoint_before, _operator_stats(operator))
        if not _finite_tensor(endpoint_prediction):
            recorder.mark_numerical_error("non_finite_endpoint_prediction")
        else:
            endpoint_residual = endpoint_prediction - measurement
            endpoint_value = _normalized(_global_norm(endpoint_residual), measurement_norm)
            trajectory_value = _normalized(_global_norm(prediction - measurement), measurement_norm)
            endpoint_confirmation = {
                "requested": True,
                "independent": True,
                "trajectory_normalized_data_residual": trajectory_value,
                "endpoint_normalized_data_residual": endpoint_value,
                "trajectory_endpoint_consistent": (
                    abs(trajectory_value - endpoint_value)
                    <= 1e-7 + 1e-4 * max(abs(trajectory_value), abs(endpoint_value))
                ),
            }
    if recorder.numerical_failure:
        return recorder.finish(
            x,
            actual_iterations=actual,
            status="numerical_error",
            stopping_reason=recorder.numerical_error_reason or "non_finite_solver_state",
            predicted_measurement=endpoint_prediction,
            resources={
                "optimization_" + key: value for key, value in _operator_call_delta(optimization_before, optimization_after).items()
            } | {"verification_" + key: value for key, value in verification_calls.items()},
            metadata={"criterion": "normalized_data_residual_and_relative_iterate_change", "endpoint_confirmation": endpoint_confirmation, **parameters, **norm_estimates},
        )
    final_residual_tensor = endpoint_prediction - measurement
    final_residual = _normalized(_global_norm(final_residual_tensor), measurement_norm)
    final_objective = _objective(final_residual_tensor)
    status = terminal_status or _finish_status(
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
            terminal_reason if terminal_reason else
            "discrepancy_and_relative_iterate_change_patience" if converged and _policy_active(control) else
            "relative_residual_and_iterate_tolerance" if converged else
            "callback_cancelled" if cancelled else
            "maximum_iterations_reached"
        ),
        final_residual=final_residual,
        final_objective=final_objective,
        predicted_measurement=endpoint_prediction,
        resources={
            "optimization_" + key: value for key, value in _operator_call_delta(optimization_before, optimization_after).items()
        } | {"verification_" + key: value for key, value in verification_calls.items()},
        metadata={
            "criterion": "normalized_data_residual_and_relative_iterate_change",
            "endpoint_confirmation": endpoint_confirmation,
            "max_iterations": limit,
            **parameters,
            **norm_estimates,
        },
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
    monitor = ConsecutiveStoppingMonitor(control)
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
    terminal_status = None
    terminal_reason = None
    if not _policy_active(control) and normal_tolerance > 0.0 and _global_norm(adjoint_residual) <= normal_tolerance:
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
        operator_norm = float(control.metadata.get("operator_norm_estimate", 1.0) or 1.0)
        normalized_normal = normal_value / max(operator_norm * _global_norm(data_residual), float(eps))
        criteria = {
            "discrepancy": residual_value <= float(control.discrepancy_target or -1.0),
            "krylov_native": (
                normalized_normal <= float(control.normalized_normal_residual_tolerance or -1.0)
                or change <= float(control.relative_iterate_tolerance or -1.0)
            ),
        }
        decision = monitor.observe(actual + 1, criteria=criteria, relative_change=change, monitor_value=residual_value)
        converged = decision.converged if _policy_active(control) else normal_tolerance > 0.0 and normal_value <= normal_tolerance
        terminal_status, terminal_reason = _policy_status(decision) if _policy_active(control) else (None, None)
        actual += 1
        if not _safe_record(
            recorder,
            actual,
            x,
            residual=data_residual,
            objective=_objective(data_residual),
            algorithm_residual=normalized_normal if _policy_active(control) else _normalized(normal_value, initial_normal),
            stopping_candidate=converged,
            consecutive_criteria_count=decision.consecutive,
            criteria=criteria,
            native_criterion_name="normalized_normal_residual",
            native_criterion_value=normalized_normal,
            native_criterion_threshold=control.normalized_normal_residual_tolerance,
            metadata={"criterion": "normal_residual", "normal_residual_absolute": normal_value},
        ):
            cancelled = True
            break
        if terminal_status or (converged and control.stop_on_convergence):
            break
        beta = next_gamma / gamma.clamp_min(float(eps))
        direction = next_adjoint + beta.reshape((beta.shape[0],) + (1,) * (direction.ndim - 1)) * direction
        adjoint_residual, gamma = next_adjoint, next_gamma
    # Keep endpoint evaluation explicit and counted.  This makes the final
    # prediction phase distinguishable from the last Krylov update and keeps
    # resource accounting consistent with the benchmark protocol.
    prediction = _finish_prediction(operator, x)
    final_residual_tensor = prediction - measurement
    status = terminal_status or _finish_status(
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
            terminal_reason if terminal_reason else
            "discrepancy_and_krylov_native_patience" if converged and _policy_active(control) else
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
    monitor = ConsecutiveStoppingMonitor(control)
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
    terminal_status = None
    terminal_reason = None
    stop_tol = float(atol) * rhs_norm + float(btol)
    residual_tensor = measurement - operator.forward(x)
    for _ in range(limit):
        residual_norm = _global_norm(residual_tensor)
        if not _policy_active(control) and residual_norm <= stop_tol:
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
        normal_value = _global_norm(operator.adjoint(residual_tensor)) if _policy_active(control) else residual_value
        operator_norm = float(control.metadata.get("operator_norm_estimate", 1.0) or 1.0)
        normalized_normal = normal_value / max(operator_norm * _global_norm(residual_tensor), epsilon) if _policy_active(control) else residual_value
        criteria = {
            "discrepancy": residual_value <= float(control.discrepancy_target or -1.0),
            "krylov_native": (
                normalized_normal <= float(control.normalized_normal_residual_tolerance or -1.0)
                or change <= float(control.relative_iterate_tolerance or -1.0)
            ),
        }
        decision = monitor.observe(actual, criteria=criteria, relative_change=change, monitor_value=residual_value)
        converged = decision.converged if _policy_active(control) else residual_norm <= stop_tol or (stop_tol > 0.0 and residual_value <= _normalized(stop_tol, rhs_norm))
        terminal_status, terminal_reason = _policy_status(decision) if _policy_active(control) else (None, None)
        if not _safe_record(
            recorder,
            actual,
            x,
            residual=-residual_tensor,
            objective=_objective(residual_tensor),
            algorithm_residual=normalized_normal if _policy_active(control) else residual_value,
            stopping_candidate=converged,
            consecutive_criteria_count=decision.consecutive,
            criteria=criteria,
            native_criterion_name="normalized_normal_residual",
            native_criterion_value=normalized_normal,
            native_criterion_threshold=control.normalized_normal_residual_tolerance,
            metadata={"criterion": "simplified_atol_btol"},
        ):
            cancelled = True
            break
        if terminal_status or (converged and control.stop_on_convergence):
            break
    prediction = (measurement - residual_tensor).detach()
    final_residual_tensor = prediction - measurement
    status = terminal_status or _finish_status(
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
            terminal_reason if terminal_reason else
            "discrepancy_and_krylov_native_patience" if converged and _policy_active(control) else
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
    parameter_errors = _common_parameter_errors(num_iterations, min_value, max_value)
    if isinstance(relaxation, bool) or not isinstance(relaxation, (int, float)) or not _finite_scalar(relaxation):
        parameter_errors.append("relaxation must be a finite number")
    elif not 0.0 < float(relaxation) <= 1.0:
        parameter_errors.append("relaxation must satisfy 0 < relaxation <= 1")
    if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not _finite_scalar(eps) or float(eps) <= 0.0:
        parameter_errors.append("eps must be a finite positive number")
    if block_size is not None and (
        isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0
    ):
        parameter_errors.append("block_size must be a positive integer or None")
    elif block_size is not None and int(block_size) > int(operator.range_shape[-2]):
        parameter_errors.append(
            "block_size must be no greater than the number of views "
            f"({int(operator.range_shape[-2])})"
        )
    subset_spec: list[tuple[int, ...]] | None = None
    if subset_indices is not None:
        try:
            subset_spec = [tuple(int(value) for value in indices) for indices in subset_indices]
        except (TypeError, ValueError):
            parameter_errors.append("subset_indices must be an iterable of integer index groups")
    if parameter_errors:
        return _parameter_failure_result(
            algorithm, measurement, operator, x_init=x_init, errors=parameter_errors,
            max_iterations=None,
            parameters={
                "num_iterations": num_iterations,
                "block_size": block_size,
                "subset_indices": subset_spec,
                "relaxation": relaxation,
                "eps": eps,
                "min_value": min_value,
                "max_value": max_value,
            },
        )
    control = resolve_control(control, default_iterations=int(num_iterations), default_tolerance=1e-5, callback=callback)
    limit = min(int(num_iterations), int(control.max_iterations or num_iterations))
    parameters = {
        "num_iterations": int(num_iterations),
        "block_size": block_size,
        "subset_indices": subset_spec,
        "order_strategy": order_strategy,
        "seed": seed,
        "relaxation": float(relaxation),
        "eps": float(eps),
        "min_value": min_value,
        "max_value": max_value,
    }
    if not _finite_tensor(measurement):
        return _numerical_result(
            algorithm, measurement, operator, x_init=x_init, max_iterations=limit,
            reason="non_finite_measurement", parameters=parameters,
        )
    x = prepare_initial_image(measurement, operator, x_init=x_init, initial_value=0.0)
    if not _finite_tensor(x):
        return _numerical_result(
            algorithm, measurement, operator, x_init=x_init, max_iterations=limit,
            reason="non_finite_initial_reconstruction", parameters=parameters,
        )
    recorder = IterationRecorder(control, measurement, operator, algorithm=algorithm)
    monitor = ConsecutiveStoppingMonitor(control)
    recorder.set_initial(x)
    try:
        subsets = make_angle_subsets(
            num_angles=int(operator.range_shape[-2]),
            block_size=block_size,
            subset_indices=subset_spec,
            order_strategy=order_strategy,
            seed=seed,
            device=measurement.device,
        )
    except (TypeError, ValueError) as error:
        return _parameter_failure_result(
            algorithm, measurement, operator, x_init=x_init, errors=[str(error)],
            max_iterations=limit, parameters=parameters,
        )
    domain = (batch, *tuple(operator.domain_shape))
    measurement_norm = _global_norm(measurement)
    optimization_before = _operator_stats(operator)
    actual = 0
    converged = False
    cancelled = False
    terminal_status = None
    terminal_reason = None
    prediction = torch.zeros_like(measurement)
    native_tolerance = _native_tolerance(control)
    partial_epoch: int | None = None
    subset_updates = 0
    for epoch in range(1, limit + 1):
        previous = x
        epoch_complete = True
        for index, indices in enumerate(subsets):
            sub_operator = make_subset_operator(operator, indices)
            y_sub = select_measurement_subset(measurement, indices)
            ones_range = torch.ones((batch, *tuple(sub_operator.range_shape)), device=measurement.device, dtype=measurement.dtype)
            row_weight = sub_operator.forward(torch.ones(domain, device=measurement.device, dtype=measurement.dtype))
            row_weight = torch.where(row_weight.abs() < float(eps), torch.full_like(row_weight, float("inf")), row_weight).reciprocal()
            if not _finite_tensor(row_weight):
                recorder.mark_numerical_error("non_finite_row_normalization")
                epoch_complete = False
                break
            column_weight = sub_operator.adjoint(ones_range)
            column_weight = torch.where(column_weight.abs() < float(eps), torch.full_like(column_weight, float("inf")), column_weight).reciprocal()
            if not _finite_tensor(column_weight):
                recorder.mark_numerical_error("non_finite_column_normalization")
                epoch_complete = False
                break
            residual = sub_operator.forward(x) - y_sub
            if not _finite_tensor(residual):
                recorder.mark_numerical_error("non_finite_residual")
                epoch_complete = False
                break
            x = x - float(relaxation) * column_weight * sub_operator.adjoint(row_weight * residual)
            x = apply_box_constraints(x, min_value=min_value, max_value=max_value)
            subset_updates += 1
            if not _finite_tensor(x):
                recorder.mark_numerical_error("non_finite_reconstruction")
                epoch_complete = False
                break
        if not epoch_complete:
            partial_epoch = epoch
            break
        prediction = _finish_prediction(operator, x)
        residual = prediction - measurement
        if not _finite_tensor(prediction) or not _finite_tensor(residual):
            recorder.mark_numerical_error("non_finite_epoch_prediction")
            partial_epoch = epoch
            break
        change = _relative_change(previous, x)
        residual_value = _normalized(_global_norm(residual), measurement_norm)
        if not _finite_scalar(change):
            recorder.mark_numerical_error("non_finite_relative_epoch_change")
            partial_epoch = epoch
            break
        if not _finite_scalar(residual_value):
            recorder.mark_numerical_error("non_finite_discrepancy")
            partial_epoch = epoch
            break
        criteria, native_tolerance = _row_action_criteria(
            control, residual_value, change, epoch=True
        )
        decision = monitor.observe(epoch, criteria=criteria, relative_change=change, monitor_value=residual_value)
        converged = decision.converged if _policy_active(control) else _tolerance_reached(control, residual=residual_value, change=change, require_both=True)
        terminal_status, terminal_reason = _policy_status(decision) if _policy_active(control) else (None, None)
        actual = epoch
        objective = _objective(residual)
        if not _finite_scalar(objective):
            recorder.mark_numerical_error("non_finite_objective")
            partial_epoch = epoch
            break
        if not _safe_record(
            recorder,
            epoch,
            x,
            residual=residual,
            objective=objective,
            algorithm_residual=residual_value,
            relaxation=float(relaxation),
            stopping_candidate=converged,
            consecutive_criteria_count=decision.consecutive,
            criteria=criteria,
            native_criterion_name="relative_epoch_change",
            native_criterion_value=change,
            native_criterion_threshold=native_tolerance,
            epoch=epoch,
            subset_count=len(subsets),
            metadata={
                "complete_sweep": True,
                "epoch_boundary": True,
                "checked": decision.checked,
                "unit": "epochs",
                "subset_sizes": [int(item.numel()) for item in subsets],
            },
        ):
            cancelled = True
            break
        if terminal_status or (converged and control.stop_on_convergence):
            break
    if recorder.numerical_failure:
        return recorder.finish(
            x,
            actual_iterations=actual,
            status="numerical_error",
            stopping_reason=recorder.numerical_error_reason or "non_finite_solver_state",
            predicted_measurement=prediction,
            metadata={
                "criterion": "complete_epoch_discrepancy_and_relative_epoch_change",
                "iteration_unit": "epochs",
                "subset_count": len(subsets),
                "internal_subset_updates": subset_updates,
                "partial_epoch": partial_epoch,
                **parameters,
            },
        )
    optimization_after = _operator_stats(operator)
    endpoint_prediction = prediction
    verification_calls: dict[str, int] = {}
    endpoint_confirmation: dict[str, Any] = {"requested": False}
    if _policy_active(control) and not cancelled:
        endpoint_before = _operator_stats(operator)
        endpoint_prediction = _finish_prediction(operator, x)
        verification_calls = _operator_call_delta(endpoint_before, _operator_stats(operator))
        if not _finite_tensor(endpoint_prediction):
            recorder.mark_numerical_error("non_finite_endpoint_prediction")
        else:
            endpoint_residual = endpoint_prediction - measurement
            endpoint_value = _normalized(_global_norm(endpoint_residual), measurement_norm)
            trajectory_value = _normalized(_global_norm(prediction - measurement), measurement_norm)
            endpoint_confirmation = {
                "requested": True,
                "independent": True,
                "complete_epoch": True,
                "trajectory_normalized_data_residual": trajectory_value,
                "endpoint_normalized_data_residual": endpoint_value,
                "trajectory_endpoint_consistent": (
                    abs(trajectory_value - endpoint_value)
                    <= 1e-7 + 1e-4 * max(abs(trajectory_value), abs(endpoint_value))
                ),
            }
    if recorder.numerical_failure:
        return recorder.finish(
            x,
            actual_iterations=actual,
            status="numerical_error",
            stopping_reason=recorder.numerical_error_reason or "non_finite_solver_state",
            predicted_measurement=endpoint_prediction,
            resources={
                "optimization_" + key: value for key, value in _operator_call_delta(optimization_before, optimization_after).items()
            } | {"verification_" + key: value for key, value in verification_calls.items()},
            metadata={
                "criterion": "complete_epoch_discrepancy_and_relative_epoch_change",
                "endpoint_confirmation": endpoint_confirmation,
                "iteration_unit": "epochs",
                "subset_count": len(subsets),
                "internal_subset_updates": subset_updates,
                "partial_epoch": partial_epoch,
                **parameters,
            },
        )
    final_residual_tensor = endpoint_prediction - measurement
    final_residual = _normalized(_global_norm(final_residual_tensor), measurement_norm)
    final_objective = _objective(final_residual_tensor)
    status = terminal_status or _finish_status(
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
            terminal_reason if terminal_reason else
            "discrepancy_and_relative_epoch_change_patience" if converged and _policy_active(control) else
            "complete_sweep_residual_and_iterate_tolerance" if converged else
            "callback_cancelled" if cancelled else
            "maximum_epochs_reached"
        ),
        final_residual=final_residual,
        final_objective=final_objective,
        predicted_measurement=endpoint_prediction,
        resources={
            "optimization_" + key: value for key, value in _operator_call_delta(optimization_before, optimization_after).items()
        } | {"verification_" + key: value for key, value in verification_calls.items()},
        metadata={
            "complete_sweep_count": actual,
            "subset_count": len(subsets),
            "iteration_unit": "epochs",
            "internal_subset_updates": subset_updates,
            "partial_epoch": partial_epoch,
            "endpoint_confirmation": endpoint_confirmation,
            "criterion": "complete_epoch_discrepancy_and_relative_epoch_change",
            "max_iterations": limit,
            **parameters,
        },
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
    monitor = ConsecutiveStoppingMonitor(control)
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
    terminal_status = None
    terminal_reason = None
    for iteration in range(1, limit + 1):
        normal_value = _global_norm(normal_residual)
        if not _policy_active(control) and float(control.tolerance or 0.0) > 0.0 and _normalized(normal_value, rhs_norm) <= float(control.tolerance):
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
        data_relative = _normalized(_global_norm(residual), _global_norm(measurement))
        reg_denominator = _global_norm(rhs) + strength * _global_norm(regularizer.gradient(x))
        reg_normalized = normal_value / max(reg_denominator, float(eps))
        criteria = {
            "discrepancy": data_relative <= float(control.discrepancy_target or -1.0),
            "regularized_normal_residual": reg_normalized <= float(control.normalized_normal_residual_tolerance or -1.0),
            "relative_iterate_change": change <= float(control.relative_iterate_tolerance or -1.0),
        }
        decision = monitor.observe(iteration, criteria=criteria, relative_change=change, monitor_value=objective)
        converged = decision.converged if _policy_active(control) else float(control.tolerance or 0.0) > 0.0 and _tolerance_reached(control, residual=normal_relative, change=change, require_both=True)
        terminal_status, terminal_reason = _policy_status(decision) if _policy_active(control) else (None, None)
        actual = iteration
        if not _safe_record(
            recorder,
            iteration,
            x,
            residual=residual,
            objective=objective,
            algorithm_residual=reg_normalized if _policy_active(control) else normal_relative,
            stopping_candidate=converged,
            consecutive_criteria_count=decision.consecutive,
            criteria=criteria,
            native_criterion_name="normalized_regularized_normal_residual",
            native_criterion_value=reg_normalized,
            native_criterion_threshold=control.normalized_normal_residual_tolerance,
            metadata={"normal_residual": normal_value, "complete_objective": True},
        ):
            cancelled = True
            break
        if terminal_status or (converged and control.stop_on_convergence):
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
        status=terminal_status or _finish_status(
            actual=actual,
            limit=limit,
            converged=converged,
            cancelled=cancelled,
            numerical_failure=recorder.numerical_failure,
        ),
        stopping_reason=(
            terminal_reason if terminal_reason else
            "discrepancy_regularized_normal_and_iterate_patience" if converged and _policy_active(control) else
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
    monitor = ConsecutiveStoppingMonitor(control)
    recorder.set_initial(x)
    measurement_norm = _global_norm(measurement)
    actual = 0
    converged = False
    cancelled = False
    terminal_status = None
    terminal_reason = None
    previous_objective = None
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
        data_relative = _normalized(_global_norm(residual), measurement_norm)
        mapping = (momentum - next_x) / step
        normalized_mapping = _global_norm(mapping) / max(_global_norm(momentum) / step, 1e-12)
        objective_change = None if previous_objective is None else abs(objective - previous_objective) / max(abs(previous_objective), 1e-12)
        criteria = {
            "discrepancy": data_relative <= float(control.discrepancy_target or -1.0),
            "prox_gradient_mapping": normalized_mapping <= float(control.prox_gradient_mapping_tolerance or -1.0),
            "relative_composite_objective_change": objective_change is not None and objective_change <= float(control.relative_objective_tolerance or -1.0),
        }
        decision = monitor.observe(iteration, criteria=criteria, relative_change=change, monitor_value=objective)
        converged = decision.converged if _policy_active(control) else _tolerance_reached(control, residual=data_relative, change=change, require_both=True)
        terminal_status, terminal_reason = _policy_status(decision) if _policy_active(control) else (None, None)
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
            consecutive_criteria_count=decision.consecutive,
            criteria=criteria,
            native_criterion_name="normalized_prox_gradient_mapping",
            native_criterion_value=normalized_mapping,
            native_criterion_threshold=control.prox_gradient_mapping_tolerance,
            metadata={"complete_objective": True, "objective_state": "momentum", "relative_composite_objective_change": objective_change, "normalized_prox_gradient_mapping": normalized_mapping, "fista_momentum": float(momentum_scale)},
        ):
            cancelled = True
            x = next_x
            break
        x = next_x
        momentum = next_momentum
        acceleration = next_acceleration
        previous_objective = objective
        if terminal_status or (converged and control.stop_on_convergence):
            break
    prediction = _finish_prediction(operator, x)
    residual = prediction - measurement
    resources = {"prox_iterations": int(actual) * int(tv.num_iterations), "prox_configured_iterations": int(tv.num_iterations)}
    return recorder.finish(
        x,
        actual_iterations=actual,
        status=terminal_status or _finish_status(
            actual=actual,
            limit=limit,
            converged=converged,
            cancelled=cancelled,
            numerical_failure=recorder.numerical_failure,
        ),
        stopping_reason=(
            terminal_reason if terminal_reason else
            "discrepancy_prox_gradient_and_objective_patience" if converged and _policy_active(control) else
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

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


def _control_threshold(control: SolveControl, name: str, default: float = -1.0) -> float:
    """Read a policy threshold without treating a valid zero as missing."""

    value = getattr(control, name, None)
    if value is not None:
        return float(value)
    if _policy_active(control):
        return {
            "relative_iterate_tolerance": 1e-4,
            "normalized_normal_residual_tolerance": 1e-4,
            "relative_objective_tolerance": 1e-5,
            "prox_gradient_mapping_tolerance": 1e-4,
        }.get(name, default)
    return default


def _finite_positive(value: Any) -> bool:
    return not isinstance(value, bool) and _finite_scalar(value) and float(value) > 0.0


def _finite_nonnegative(value: Any) -> bool:
    return not isinstance(value, bool) and _finite_scalar(value) and float(value) >= 0.0


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
    if actual >= limit:
        return "max_iterations"
    if converged:
        return "converged"
    return "partial"


def _safe_record(
    recorder: IterationRecorder,
    *args: Any,
    **kwargs: Any,
) -> bool:
    """Record one row and let callbacks request cancellation."""

    return recorder.record(*args, **kwargs)


def _policy_active(control: SolveControl) -> bool:
    # Runtime policies are retained in metadata even when discrepancy is not
    # statistically justified.  Such runs still need native monitoring; the
    # absence of a target must not silently downgrade them to legacy stopping.
    return (
        control.discrepancy_target is not None
        or (control.metadata or {}).get("effective_stopping_policy") is not None
        or any(
            getattr(control, name, None) is not None
            for name in (
                "relative_iterate_tolerance",
                "normalized_normal_residual_tolerance",
                "relative_objective_tolerance",
                "prox_gradient_mapping_tolerance",
            )
        )
        or control.stall_enabled
        or control.divergence_enabled
    )


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

    try:
        x = prepare_initial_image(measurement, operator, x_init=x_init, initial_value=0.0)
    except (AttributeError, TypeError, ValueError):
        x = _placeholder_reconstruction(measurement, operator)
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

    try:
        x = prepare_initial_image(measurement, operator, x_init=x_init, initial_value=0.0)
    except (AttributeError, TypeError, ValueError):
        x = _placeholder_reconstruction(measurement, operator)
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


def _placeholder_reconstruction(
    measurement: Any,
    operator: Any,
) -> torch.Tensor:
    """Build a safe tensor for structured pre-loop failures.

    Detailed solver entry points report validation failures as ``SolveResult``
    values.  The placeholder is never presented as a reconstruction; it only
    keeps that error result serializable when the input itself has a bad shape
    or dtype.
    """

    try:
        batch = int(measurement.shape[0]) if isinstance(measurement, torch.Tensor) and measurement.ndim else 1
    except (AttributeError, TypeError, ValueError):
        batch = 1
    try:
        domain_shape = tuple(int(value) for value in operator.domain_shape)
    except (AttributeError, TypeError, ValueError):
        domain_shape = ()
    if isinstance(measurement, torch.Tensor):
        device = measurement.device
        dtype = measurement.dtype if measurement.dtype.is_floating_point else torch.float32
    else:
        device = torch.device("cpu")
        dtype = torch.float32
    return torch.zeros((max(1, batch), *domain_shape), device=device, dtype=dtype)


def _native_tolerance(control: SolveControl) -> float | None:
    if control.relative_iterate_tolerance is not None:
        return float(control.relative_iterate_tolerance)
    # A policy is still meaningful for direct detailed callers that only set
    # the discrepancy target.  Keep the same documented native threshold as
    # the runtime policy instead of silently making the second criterion
    # impossible to satisfy.
    return 1e-4 if _policy_active(control) else None


def _row_action_criteria(
    control: SolveControl,
    residual_value: float,
    change: float,
    *,
    epoch: bool = False,
) -> tuple[dict[str, bool], float | None]:
    if not _policy_active(control):
        return {}, _native_tolerance(control)
    native_tolerance = _native_tolerance(control)
    assert native_tolerance is not None
    native_name = "relative_epoch_change" if epoch else "relative_iterate_change"
    criteria = {native_name: change <= native_tolerance}
    if control.discrepancy_target is not None:
        criteria = {
            "discrepancy": residual_value <= float(control.discrepancy_target),
            **criteria,
        }
    return criteria, native_tolerance


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


def _resolve_operator_norm_estimate(
    operator: LinearOperator,
    reference: torch.Tensor,
    control: SolveControl,
    *,
    explicit: float | None = None,
    explicit_squared: float | None = None,
) -> tuple[float, dict[str, Any]]:
    """Resolve ``||A||`` for scale-aware Krylov diagnostics.

    Runtime calls normally provide the estimate in ``SolveControl.metadata``.
    Direct detailed callers may omit it; in that case use the same
    deterministic power iteration as the parameter-validation path.  A
    supplied squared estimate is deliberately converted exactly once here so
    the residual normalization cannot accidentally use ``||A||^2``.
    """

    metadata = dict(control.metadata or {})
    source = "explicit_argument"
    norm: float | None = None
    if explicit is not None:
        norm = float(explicit)
    elif explicit_squared is not None:
        squared = float(explicit_squared)
        if not _finite_nonnegative(squared):
            raise ValueError("operator_norm_squared must be finite and nonnegative")
        norm = math.sqrt(squared)
        source = "explicit_squared"
    elif metadata.get("operator_norm_estimate") is not None:
        norm = float(metadata["operator_norm_estimate"])
        source = str(metadata.get("operator_norm_estimator", "control_metadata"))
    elif metadata.get("operator_norm") is not None:
        norm = float(metadata["operator_norm"])
        source = str(metadata.get("operator_norm_estimator", "control_metadata"))
    elif metadata.get("operator_norm_squared") is not None:
        squared = float(metadata["operator_norm_squared"])
        if not _finite_nonnegative(squared):
            raise ValueError("operator_norm_squared must be finite and nonnegative")
        norm = math.sqrt(squared)
        source = str(metadata.get("operator_norm_estimator", "control_metadata_squared"))

    if norm is None:
        iterations = metadata.get("power_iterations", 8)
        try:
            iterations = int(iterations)
        except (TypeError, ValueError):
            iterations = 8
        if iterations <= 0:
            raise ValueError("power_iterations must be positive")
        squared = _estimate_operator_norm_squared(
            operator, reference, iterations=iterations
        )
        if not _finite_nonnegative(squared):
            raise FloatingPointError("operator norm power iteration returned a non-finite value")
        norm = math.sqrt(squared)
        source = f"power_iteration_{iterations}"

    if not _finite_nonnegative(norm):
        raise ValueError("operator_norm_estimate must be finite and nonnegative")
    return float(norm), {
        "operator_norm_estimate": float(norm),
        "operator_norm_squared": float(norm * norm),
        "operator_norm_estimator": source,
        "normal_residual_normalization": (
            "||A^T(Ax-b)||_2 / max(||A||_estimate * ||Ax-b||_2, eps)"
        ),
    }


def _krylov_normal_metrics(
    operator: LinearOperator,
    residual: torch.Tensor,
    operator_norm_estimate: float,
    *,
    eps: float,
    x: torch.Tensor | None = None,
    damping: float = 0.0,
) -> tuple[torch.Tensor, float, float, float]:
    """Return gradient, norm, scale-aware backward error, and denominator."""

    gradient = operator.adjoint(residual)
    damping_sq = float(damping) ** 2
    if damping_sq > 0.0:
        if x is None:
            raise ValueError("x is required for damped Krylov diagnostics")
        gradient = gradient + damping_sq * x
    gradient_norm = _global_norm(gradient)
    denominator = float(operator_norm_estimate) * _global_norm(residual)
    if damping_sq > 0.0 and x is not None:
        denominator += damping_sq * _global_norm(x)
    normalized = gradient_norm / max(denominator, float(eps))
    return gradient, gradient_norm, normalized, denominator


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


def _direct_result(
    algorithm: str,
    measurement: Any,
    operator: Any,
    *,
    status: str,
    stopping_reason: str,
    parameters: Mapping[str, Any],
    error: BaseException | None = None,
) -> SolveResult:
    """Return a structured direct-solver outcome without entering a loop."""

    metadata: dict[str, Any] = {
        "algorithm": algorithm,
        "direct": True,
        "max_iterations": 1,
        "parameters": dict(parameters),
    }
    if error is not None:
        metadata["failure_reason"] = str(error)
    return SolveResult(
        reconstruction=_placeholder_reconstruction(measurement, operator),
        actual_iterations=0,
        status=status,
        stopping_reason=stopping_reason,
        resources={"trajectory_available": False, **_operator_stats(operator)},
        metadata=metadata,
    )


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
    parameters = {"scale": scale}
    try:
        require_linear_operator(operator, "fbp")
        validate_measurement_shape(measurement, operator, "fbp")
    except (TypeError, ValueError) as error:
        return _direct_result(
            "fbp", measurement, operator, status="invalid_parameters",
            stopping_reason="parameter_validation_failed", parameters=parameters,
            error=error,
        )
    if scale is not None and not _finite_positive(scale):
        return _direct_result(
            "fbp", measurement, operator, status="invalid_parameters",
            stopping_reason="invalid_scale", parameters=parameters,
            error=ValueError("scale must be a finite positive number or None"),
        )
    if not _finite_tensor(measurement):
        return _direct_result(
            "fbp", measurement, operator, status="numerical_error",
            stopping_reason="non_finite_measurement", parameters=parameters,
            error=FloatingPointError("FBP measurement contains NaN or Inf"),
        )
    try:
        control = resolve_control(control, default_iterations=1, callback=callback)
    except (TypeError, ValueError) as error:
        return _direct_result(
            "fbp", measurement, operator, status="invalid_parameters",
            stopping_reason="invalid_control", parameters=parameters, error=error,
        )
    try:
        reconstruction = fbp(operator, measurement, scale=scale)
    except Exception as error:
        return _direct_result(
            "fbp", measurement, operator, status="numerical_error",
            stopping_reason="fbp_execution_failed", parameters=parameters, error=error,
        )
    expected = (measurement.shape[0], *tuple(operator.domain_shape))
    if not isinstance(reconstruction, torch.Tensor) or tuple(reconstruction.shape) != expected:
        return _direct_result(
            "fbp", measurement, operator, status="numerical_error",
            stopping_reason="invalid_reconstruction_shape", parameters=parameters,
            error=ValueError(
                f"FBP returned {type(reconstruction).__name__} with shape "
                f"{getattr(reconstruction, 'shape', None)}; expected {expected}"
            ),
        )
    if not _finite_tensor(reconstruction):
        return _direct_result(
            "fbp", measurement, operator, status="numerical_error",
            stopping_reason="non_finite_reconstruction", parameters=parameters,
            error=FloatingPointError("FBP returned NaN or Inf"),
        )
    recorder = IterationRecorder(control, measurement, operator, algorithm="fbp")
    return recorder.finish(
        reconstruction,
        actual_iterations=0,
        status="completed_valid",
        stopping_reason="direct_reconstruction_valid",
        resources={"trajectory_available": False, "direct": True},
        metadata={"direct": True, "parameters": parameters, "completed_valid": True},
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

    parameters = dict(kwargs)
    for name, value in (
        ("filter_type", parameters.get("filter_type", "ram-lak")),
        ("filter", parameters.get("filter")),
    ):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            return _direct_result(
                "fdk", measurement, operator, status="invalid_parameters",
                stopping_reason=f"invalid_{name}", parameters=parameters,
                error=ValueError(f"{name} must be a non-empty string"),
            )
    filter_type = str(parameters.get("filter_type", "ram-lak")).strip().lower().replace("_", "-")
    filter_alias = parameters.get("filter")
    if filter_type in {"ramp", "ramlak"}:
        filter_type = "ram-lak"
    if filter_alias is not None:
        filter_alias = str(filter_alias).strip().lower().replace("_", "-")
        if filter_alias in {"ramp", "ramlak"}:
            filter_alias = "ram-lak"
    if filter_type != "ram-lak" or (
        filter_alias is not None and filter_alias != filter_type
    ):
        return _direct_result(
            "fdk", measurement, operator, status="invalid_parameters",
            stopping_reason="invalid_filter", parameters=parameters,
            error=ValueError(
                "FDK supports only the native Ram-Lak filter and compatible aliases"
            ),
        )
    short_scan = parameters.get("short_scan", False)
    if not isinstance(short_scan, bool):
        return _direct_result(
            "fdk", measurement, operator, status="invalid_parameters",
            stopping_reason="invalid_short_scan", parameters=parameters,
            error=TypeError("short_scan must be a bool"),
        )
    supersampling = parameters.get("voxel_supersampling", 1)
    if (
        isinstance(supersampling, bool)
        or not isinstance(supersampling, int)
        or supersampling <= 0
    ):
        return _direct_result(
            "fdk", measurement, operator, status="invalid_parameters",
            stopping_reason="invalid_voxel_supersampling", parameters=parameters,
            error=ValueError("voxel_supersampling must be a positive integer"),
        )
    try:
        require_linear_operator(operator, "fdk")
        validate_measurement_shape(measurement, operator, "fdk")
    except (TypeError, ValueError) as error:
        return _direct_result(
            "fdk", measurement, operator, status="invalid_parameters",
            stopping_reason="parameter_validation_failed", parameters=parameters,
            error=error,
        )
    if not _finite_tensor(measurement):
        return _direct_result(
            "fdk", measurement, operator, status="numerical_error",
            stopping_reason="non_finite_measurement", parameters=parameters,
            error=FloatingPointError("FDK measurement contains NaN or Inf"),
        )
    try:
        control = resolve_control(control, default_iterations=1, callback=callback)
    except (TypeError, ValueError) as error:
        return _direct_result(
            "fdk", measurement, operator, status="invalid_parameters",
            stopping_reason="invalid_control", parameters=parameters, error=error,
        )
    try:
        reconstruction = fdk(operator, measurement, **parameters)
    except ImportError as error:
        return _direct_result(
            "fdk", measurement, operator, status="unavailable",
            stopping_reason="backend_unavailable", parameters=parameters, error=error,
        )
    except NotImplementedError as error:
        message = str(error).lower()
        invalid_backend_options = (
            "would be ignored",
            "requires cone geometry",
            "unsupported",
            "filter",
        )
        status = "invalid_parameters" if any(token in message for token in invalid_backend_options) else "unavailable"
        return _direct_result(
            "fdk", measurement, operator, status=status,
            stopping_reason="parameter_validation_failed" if status == "invalid_parameters" else "backend_unavailable",
            parameters=parameters, error=error,
        )
    except RuntimeError as error:
        message = str(error).lower()
        unavailable_tokens = ("cuda", "astra", "not installed", "unavailable", "no fdk backend")
        status = "unavailable" if any(token in message for token in unavailable_tokens) else "numerical_error"
        return _direct_result(
            "fdk", measurement, operator, status=status,
            stopping_reason="backend_unavailable" if status == "unavailable" else "fdk_execution_failed",
            parameters=parameters, error=error,
        )
    except (TypeError, ValueError) as error:
        message = str(error).lower()
        # The wrapper's own return-contract errors describe an invalid
        # backend result, while adapter option/geometry errors are invalid
        # parameters.  Preserve that distinction in the detailed API.
        output_error = "must return" in message or "returned shape" in message
        return _direct_result(
            "fdk", measurement, operator,
            status="numerical_error" if output_error else "invalid_parameters",
            stopping_reason="invalid_reconstruction_output" if output_error else "parameter_validation_failed",
            parameters=parameters, error=error,
        )
    except Exception as error:
        return _direct_result(
            "fdk", measurement, operator, status="numerical_error",
            stopping_reason="fdk_execution_failed", parameters=parameters, error=error,
        )
    expected = (measurement.shape[0], *tuple(operator.domain_shape))
    if not isinstance(reconstruction, torch.Tensor) or tuple(reconstruction.shape) != expected:
        return _direct_result(
            "fdk", measurement, operator, status="numerical_error",
            stopping_reason="invalid_reconstruction_shape", parameters=parameters,
            error=ValueError(f"FDK returned shape {getattr(reconstruction, 'shape', None)}; expected {expected}"),
        )
    if not _finite_tensor(reconstruction):
        return _direct_result(
            "fdk", measurement, operator, status="numerical_error",
            stopping_reason="non_finite_reconstruction", parameters=parameters,
            error=FloatingPointError("FDK returned NaN or Inf"),
        )
    recorder = IterationRecorder(control, measurement, operator, algorithm="fdk")
    return recorder.finish(
        reconstruction,
        actual_iterations=0,
        status="completed_valid",
        stopping_reason="direct_reconstruction_valid",
        resources={"trajectory_available": False, "direct": True},
        metadata={"direct": True, "parameters": parameters, "completed_valid": True},
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
    operator_norm_estimate: float | None = None,
    operator_norm_squared: float | None = None,
) -> SolveResult:
    require_linear_operator(operator, "cgls")
    validate_measurement_shape(measurement, operator, "cgls")
    parameter_errors = _common_parameter_errors(num_iterations, min_value, max_value)
    if not _finite_nonnegative(tol):
        parameter_errors.append("tol must be a finite nonnegative number")
    if not _finite_positive(eps):
        parameter_errors.append("eps must be a finite positive number")
    if operator_norm_estimate is not None and not _finite_nonnegative(operator_norm_estimate):
        parameter_errors.append("operator_norm_estimate must be finite and nonnegative")
    if operator_norm_squared is not None and not _finite_nonnegative(operator_norm_squared):
        parameter_errors.append("operator_norm_squared must be finite and nonnegative")
    if parameter_errors:
        return _parameter_failure_result(
            "cgls", measurement, operator, x_init=x_init,
            errors=parameter_errors,
            max_iterations=num_iterations if isinstance(num_iterations, int) and not isinstance(num_iterations, bool) else None,
            parameters={
                "num_iterations": num_iterations,
                "tol": tol,
                "min_value": min_value,
                "max_value": max_value,
                "eps": eps,
                "operator_norm_estimate": operator_norm_estimate,
                "operator_norm_squared": operator_norm_squared,
            },
        )
    control = resolve_control(control, default_iterations=int(num_iterations), default_tolerance=None, callback=callback)
    limit = min(int(num_iterations), int(control.max_iterations or num_iterations))
    parameters = {
        "num_iterations": int(num_iterations),
        "tol": float(tol),
        "min_value": min_value,
        "max_value": max_value,
        "eps": float(eps),
    }
    if not _finite_tensor(measurement):
        return _numerical_result(
            "cgls", measurement, operator, x_init=x_init, max_iterations=limit,
            reason="non_finite_solver_state", parameters=parameters,
        )
    x = prepare_initial_image(measurement, operator, x_init=x_init, initial_value=0.0)
    if not _finite_tensor(x):
        return _numerical_result(
            "cgls", measurement, operator, x_init=x_init, max_iterations=limit,
            reason="non_finite_initial_reconstruction", parameters=parameters,
        )
    if _policy_active(control) or operator_norm_estimate is not None or operator_norm_squared is not None:
        try:
            operator_norm, norm_metadata = _resolve_operator_norm_estimate(
                operator,
                measurement,
                control,
                explicit=operator_norm_estimate,
                explicit_squared=operator_norm_squared,
            )
        except FloatingPointError:
            return _numerical_result(
                "cgls", measurement, operator, x_init=x_init, max_iterations=limit,
                reason="non_finite_operator_norm_estimate", parameters=parameters,
            )
        except ValueError as error:
            return _parameter_failure_result(
                "cgls", measurement, operator, x_init=x_init, errors=[str(error)],
                max_iterations=limit, parameters=parameters,
            )
    else:
        operator_norm = 1.0
        norm_metadata = {
            "operator_norm_estimate": None,
            "operator_norm_squared": None,
            "operator_norm_estimator": "not_required_for_legacy_stop",
        }
    recorder = IterationRecorder(control, measurement, operator, algorithm="cgls")
    monitor = ConsecutiveStoppingMonitor(control)
    recorder.set_initial(x)
    prediction = operator.forward(x)
    residual = prediction - measurement
    adjoint_residual = operator.adjoint(-residual)
    gamma = _sqnorm(adjoint_residual)
    initial_normal = max(_global_norm(adjoint_residual), float(eps))
    normal_tolerance = float(tol)
    if control.tolerance is not None:
        normal_tolerance = float(control.tolerance)
    if not _finite_tensor(prediction) or not _finite_tensor(residual) or not _finite_tensor(adjoint_residual):
        recorder.mark_numerical_error("non_finite_solver_state")
    actual = 0
    converged = False
    cancelled = False
    terminal_status = None
    terminal_reason = None
    if not recorder.numerical_failure and not _policy_active(control) and _global_norm(adjoint_residual) <= normal_tolerance:
        recorder.record(
            0,
            x,
            residual=residual,
            objective=_objective(residual),
            algorithm_residual=_global_norm(adjoint_residual) / max(
                operator_norm * _global_norm(residual), float(eps)
            ),
            stopping_candidate=True,
            metadata={**norm_metadata, "criterion": "absolute_normal_residual"},
        )
        converged = True
    direction = adjoint_residual.clone()
    for iteration in range(1, limit + 1):
        if recorder.numerical_failure:
            break
        if converged and control.stop_on_convergence:
            break
        q = operator.forward(direction)
        denominator = _sqnorm(q)
        if not _finite_tensor(q) or not bool(torch.isfinite(denominator).all()):
            recorder.mark_numerical_error("non_finite_krylov_direction")
            break
        if bool(torch.any(denominator <= float(eps))):
            # A zero Krylov direction at a stationary point is benign (for
            # example, a zero operator or an exact initial solution).  It is
            # not evidence of data consistency; leave policy runs as a
            # structured stall and reserve numerical_error for a material
            # normal-equation breakdown.
            normal_value = _global_norm(adjoint_residual)
            breakdown_scale = max(_global_norm(measurement), 1.0)
            if normal_value <= max(float(eps), 1e-7 * breakdown_scale):
                if not _policy_active(control) and normal_value <= normal_tolerance:
                    converged = True
                    break
                if _policy_active(control):
                    residual = prediction - measurement
                    residual_value = _normalized(_global_norm(residual), _global_norm(measurement))
                    normalized_normal = normal_value / max(
                        operator_norm * _global_norm(residual), float(eps)
                    )
                    normal_tolerance_policy = _control_threshold(
                        control, "normalized_normal_residual_tolerance"
                    )
                    iterate_tolerance_policy = _control_threshold(
                        control, "relative_iterate_tolerance"
                    )
                    criteria = {
                        "krylov_native": (
                            normalized_normal <= normal_tolerance_policy
                            or 0.0 <= iterate_tolerance_policy
                        ),
                    }
                    if control.discrepancy_target is not None:
                        criteria = {
                            "discrepancy": residual_value <= float(control.discrepancy_target),
                            **criteria,
                        }
                    decision = monitor.observe(
                        iteration,
                        criteria=criteria,
                        relative_change=0.0,
                        monitor_value=normalized_normal,
                    )
                    converged = decision.converged
                    terminal_status, terminal_reason = _policy_status(decision)
                    actual = iteration
                    if not _safe_record(
                        recorder,
                        iteration,
                        x,
                        residual=residual,
                        objective=_objective(residual),
                        algorithm_residual=normalized_normal,
                        stopping_candidate=converged,
                        consecutive_criteria_count=decision.consecutive,
                        criteria=criteria,
                        native_criterion_name="normalized_normal_residual",
                        native_criterion_value=normalized_normal,
                        native_criterion_threshold=normal_tolerance_policy,
                        metadata={
                            **norm_metadata,
                            "criterion": "scale_aware_normal_residual",
                            "normal_residual_absolute": normal_value,
                            "normal_residual_denominator": operator_norm * _global_norm(residual),
                            "breakdown": "stationary_krylov_direction",
                        },
                    ):
                        cancelled = True
                        break
                    if terminal_status or (converged and control.stop_on_convergence):
                        break
                    if control.discrepancy_target is not None and not criteria["discrepancy"]:
                        terminal_status = "stalled"
                        terminal_reason = "stalled_before_discrepancy"
                        break
                    continue
                terminal_status = "max_iterations"
                terminal_reason = "maximum_iterations_reached"
                break
            recorder.mark_numerical_error("normal_equation_breakdown")
            break
        alpha = gamma / denominator.clamp_min(float(eps))
        if not bool(torch.isfinite(alpha).all()):
            recorder.mark_numerical_error("non_finite_krylov_step")
            break
        previous = x
        x = x + alpha.reshape((alpha.shape[0],) + (1,) * (x.ndim - 1)) * direction
        x = apply_box_constraints(x, min_value=min_value, max_value=max_value)
        if not _finite_tensor(x):
            recorder.mark_numerical_error("non_finite_reconstruction")
            break
        prediction = operator.forward(x)
        residual = prediction - measurement
        next_adjoint = operator.adjoint(-residual)
        next_gamma = _sqnorm(next_adjoint)
        if not _finite_tensor(prediction) or not _finite_tensor(residual) or not _finite_tensor(next_adjoint):
            recorder.mark_numerical_error("non_finite_solver_state")
            break
        change = _relative_change(previous, x)
        normal_value = _global_norm(next_adjoint)
        residual_value = _normalized(_global_norm(residual), _global_norm(measurement))
        normalized_normal = normal_value / max(operator_norm * _global_norm(residual), float(eps))
        normal_tolerance_policy = _control_threshold(
            control, "normalized_normal_residual_tolerance"
        )
        iterate_tolerance_policy = _control_threshold(
            control, "relative_iterate_tolerance"
        )
        criteria = {
            "krylov_native": (
                normalized_normal <= normal_tolerance_policy
                or change <= iterate_tolerance_policy
            ),
        }
        if control.discrepancy_target is not None:
            criteria = {
                "discrepancy": residual_value <= float(control.discrepancy_target),
                **criteria,
            }
        decision = monitor.observe(iteration, criteria=criteria, relative_change=change, monitor_value=normalized_normal)
        converged = decision.converged if _policy_active(control) else normal_value <= normal_tolerance
        terminal_status, terminal_reason = _policy_status(decision) if _policy_active(control) else (None, None)
        actual = iteration
        if not _safe_record(
            recorder,
            actual,
            x,
            residual=residual,
            objective=_objective(residual),
            algorithm_residual=normalized_normal if _policy_active(control) else _normalized(normal_value, initial_normal),
            stopping_candidate=converged,
            consecutive_criteria_count=decision.consecutive,
            criteria=criteria,
            native_criterion_name="normalized_normal_residual",
            native_criterion_value=normalized_normal,
            native_criterion_threshold=(
                normal_tolerance_policy
                if _policy_active(control)
                else control.normalized_normal_residual_tolerance
            ),
            metadata={
                **norm_metadata,
                "criterion": "scale_aware_normal_residual",
                "normal_residual_absolute": normal_value,
                "normal_residual_denominator": operator_norm * _global_norm(residual),
            },
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
            "discrepancy_and_krylov_native_patience" if status == "converged" and _policy_active(control) else
            "normal_residual_tolerance" if status == "converged" else
            "callback_cancelled" if cancelled else
            "maximum_iterations_reached"
        ),
        final_residual=_normalized(_global_norm(final_residual_tensor), _global_norm(measurement)),
        final_objective=_objective(final_residual_tensor),
        predicted_measurement=prediction,
        metadata={
            **norm_metadata,
            "criterion": "scale_aware_normal_residual",
            "tolerance": normal_tolerance,
            "normal_residual_formula": "||A^T(Ax-b)||_2 / max(||A||_estimate * ||Ax-b||_2, eps)",
        },
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
    operator_norm_estimate: float | None = None,
    operator_norm_squared: float | None = None,
) -> SolveResult:
    require_linear_operator(operator, "lsqr")
    validate_measurement_shape(measurement, operator, "lsqr")
    parameter_errors = _common_parameter_errors(num_iterations, min_value, max_value)
    for name, value in (("damping", damping), ("atol", atol), ("btol", btol)):
        if not _finite_nonnegative(value):
            parameter_errors.append(f"{name} must be a finite nonnegative number")
    if not _finite_positive(eps):
        parameter_errors.append("eps must be a finite positive number")
    if operator_norm_estimate is not None and not _finite_nonnegative(operator_norm_estimate):
        parameter_errors.append("operator_norm_estimate must be finite and nonnegative")
    if operator_norm_squared is not None and not _finite_nonnegative(operator_norm_squared):
        parameter_errors.append("operator_norm_squared must be finite and nonnegative")
    if parameter_errors:
        return _parameter_failure_result(
            "lsqr", measurement, operator, x_init=x_init,
            errors=parameter_errors,
            max_iterations=num_iterations if isinstance(num_iterations, int) and not isinstance(num_iterations, bool) else None,
            parameters={
                "num_iterations": num_iterations,
                "damping": damping,
                "atol": atol,
                "btol": btol,
                "min_value": min_value,
                "max_value": max_value,
                "eps": eps,
                "operator_norm_estimate": operator_norm_estimate,
                "operator_norm_squared": operator_norm_squared,
            },
        )
    control = resolve_control(control, default_iterations=int(num_iterations), default_tolerance=None, callback=callback)
    limit = min(int(num_iterations), int(control.max_iterations or num_iterations))
    epsilon = float(eps)
    x = prepare_initial_image(measurement, operator, x_init=x_init, initial_value=0.0)
    parameters = {
        "num_iterations": int(num_iterations),
        "damping": float(damping),
        "atol": float(atol),
        "btol": float(btol),
        "min_value": min_value,
        "max_value": max_value,
        "eps": float(eps),
    }
    if not _finite_tensor(measurement):
        return _numerical_result(
            "lsqr", measurement, operator, x_init=x_init, max_iterations=limit,
            reason="non_finite_solver_state", parameters=parameters,
        )
    if not _finite_tensor(x):
        return _numerical_result(
            "lsqr", measurement, operator, x_init=x_init, max_iterations=limit,
            reason="non_finite_initial_reconstruction", parameters=parameters,
        )
    if _policy_active(control) or operator_norm_estimate is not None or operator_norm_squared is not None:
        try:
            operator_norm, norm_metadata = _resolve_operator_norm_estimate(
                operator,
                measurement,
                control,
                explicit=operator_norm_estimate,
                explicit_squared=operator_norm_squared,
            )
        except FloatingPointError:
            return _numerical_result(
                "lsqr", measurement, operator, x_init=x_init, max_iterations=limit,
                reason="non_finite_operator_norm_estimate", parameters=parameters,
            )
        except ValueError as error:
            return _parameter_failure_result(
                "lsqr", measurement, operator, x_init=x_init, errors=[str(error)],
                max_iterations=limit, parameters=parameters,
            )
    else:
        operator_norm = 1.0
        norm_metadata = {
            "operator_norm_estimate": None,
            "operator_norm_squared": None,
            "operator_norm_estimator": "not_required_for_legacy_stop",
        }
    recorder = IterationRecorder(control, measurement, operator, algorithm="lsqr")
    monitor = ConsecutiveStoppingMonitor(control)
    recorder.set_initial(x)
    rhs_norm = _global_norm(measurement)
    initial_prediction = operator.forward(x)
    u = measurement - initial_prediction
    beta = _norm(u)
    if not _finite_tensor(initial_prediction) or not _finite_tensor(u) or not bool(torch.isfinite(beta).all()):
        recorder.mark_numerical_error("non_finite_solver_state")
    u = u / beta.clamp_min(epsilon).reshape((beta.shape[0],) + (1,) * (u.ndim - 1))
    v = operator.adjoint(u)
    alpha = _norm(v)
    if not _finite_tensor(v) or not bool(torch.isfinite(alpha).all()):
        recorder.mark_numerical_error("non_finite_solver_state")
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
    if not _finite_tensor(residual_tensor):
        recorder.mark_numerical_error("non_finite_solver_state")
    if (
        not recorder.numerical_failure
        and not _policy_active(control)
        and _global_norm(residual_tensor) <= stop_tol
    ):
        converged = True
        recorder.record(
            0,
            x,
            residual=-residual_tensor,
            objective=_objective(residual_tensor),
            algorithm_residual=_normalized(_global_norm(residual_tensor), rhs_norm),
            stopping_candidate=True,
            metadata={"criterion": "atol_btol_tolerance", **norm_metadata},
        )
    # A zero initial bidiagonal coefficient means that the residual has no
    # usable Krylov direction.  It is a legitimate exact fit when the native
    # tolerance already accepts the residual; otherwise it is a stationary
    # breakdown and must not be advertised as convergence.
    if (
        not recorder.numerical_failure
        and not converged
        and bool(torch.any(alpha <= epsilon))
    ):
        discrepancy_ok = (
            control.discrepancy_target is None
            or _normalized(_global_norm(residual_tensor), rhs_norm)
            <= float(control.discrepancy_target)
        )
        if _policy_active(control) and discrepancy_ok:
            # The zero initial bidiagonal coefficient can also represent an
            # exact initial fit.  Let the normal monitor collect its patience
            # window instead of converting that valid stationary state into a
            # premature stall.
            pass
        else:
            terminal_status = "stalled" if _policy_active(control) else "max_iterations"
            terminal_reason = (
                "stalled_before_discrepancy"
                if _policy_active(control) else "krylov_breakdown"
            )
    for iteration in range(1, limit + 1):
        if recorder.numerical_failure:
            break
        if converged and control.stop_on_convergence:
            break
        if terminal_status:
            break
        residual_norm = _global_norm(residual_tensor)
        u = operator.forward(v) - alpha.reshape((alpha.shape[0],) + (1,) * (u.ndim - 1)) * u
        beta = _norm(u)
        if not _finite_tensor(u) or not bool(torch.isfinite(beta).all()):
            recorder.mark_numerical_error("non_finite_bidiagonal_state")
            break
        u = u / beta.clamp_min(epsilon).reshape((beta.shape[0],) + (1,) * (u.ndim - 1))
        v = operator.adjoint(u) - beta.reshape((beta.shape[0],) + (1,) * (v.ndim - 1)) * v
        alpha = _norm(v)
        if not _finite_tensor(v) or not bool(torch.isfinite(alpha).all()):
            recorder.mark_numerical_error("non_finite_bidiagonal_state")
            break
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
        if not _finite_tensor(x) or not _finite_tensor(direction):
            recorder.mark_numerical_error("non_finite_reconstruction")
            break
        actual = iteration
        residual_tensor = measurement - operator.forward(x)
        if not _finite_tensor(residual_tensor):
            recorder.mark_numerical_error("non_finite_residual")
            break
        residual_value = _normalized(_global_norm(residual_tensor), rhs_norm)
        change = _relative_change(previous, x)
        if _policy_active(control):
            try:
                _gradient, normal_value, normalized_normal, normal_denominator = _krylov_normal_metrics(
                    operator,
                    -residual_tensor,
                    operator_norm,
                    eps=epsilon,
                    x=x,
                    damping=float(damping),
                )
            except (FloatingPointError, ValueError):
                recorder.mark_numerical_error("non_finite_normal_residual")
                break
        else:
            normal_value = residual_value
            normalized_normal = residual_value
            normal_denominator = _global_norm(residual_tensor)
        normal_tolerance_policy = _control_threshold(
            control, "normalized_normal_residual_tolerance"
        )
        iterate_tolerance_policy = _control_threshold(
            control, "relative_iterate_tolerance"
        )
        criteria = {
            "krylov_native": (
                normalized_normal <= normal_tolerance_policy
                or change <= iterate_tolerance_policy
            ),
        }
        if control.discrepancy_target is not None:
            criteria = {
                "discrepancy": residual_value <= float(control.discrepancy_target),
                **criteria,
            }
        decision = monitor.observe(actual, criteria=criteria, relative_change=change, monitor_value=normalized_normal)
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
            native_criterion_threshold=(
                normal_tolerance_policy
                if _policy_active(control)
                else control.normalized_normal_residual_tolerance
            ),
            metadata={
                **norm_metadata,
                "criterion": "scale_aware_normal_residual" if _policy_active(control) else "atol_btol_tolerance",
                "normal_residual_denominator": normal_denominator,
                "damping": float(damping),
                "normal_residual_formula": (
                    "||A^T(Ax-b) + damping^2*x||_2 / "
                    "max(||A||_estimate*||Ax-b||_2 + damping^2*||x||_2, eps)"
                    if float(damping) > 0.0 else
                    "||A^T(Ax-b)||_2 / max(||A||_estimate * ||Ax-b||_2, eps)"
                ),
            },
        ):
            cancelled = True
            break
        if bool(torch.any(alpha <= epsilon)) and not converged:
            if not _policy_active(control) and _global_norm(residual_tensor) <= stop_tol:
                converged = True
            elif _policy_active(control) and (
                control.discrepancy_target is None
                or bool(criteria.get("discrepancy", False))
            ):
                # A zero Lanczos coefficient after a successful update is an
                # exact Krylov plateau, not a failure.  Keep recording the
                # stationary evidence so the configured patience window can
                # still be honored.
                pass
            else:
                terminal_status = "stalled" if _policy_active(control) else "max_iterations"
                terminal_reason = (
                    "stalled_before_discrepancy"
                    if _policy_active(control) else "krylov_breakdown"
                )
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
            "discrepancy_and_krylov_native_patience" if status == "converged" and _policy_active(control) else
            "atol_btol_tolerance" if status == "converged" else
            "callback_cancelled" if cancelled else
            "maximum_iterations_reached"
        ),
        final_residual=_normalized(_global_norm(final_residual_tensor), rhs_norm),
        final_objective=_objective(final_residual_tensor),
        predicted_measurement=prediction,
        metadata={
            **norm_metadata,
            "criterion": "scale_aware_normal_residual" if _policy_active(control) else "simplified_atol_times_measurement_norm_plus_btol",
            "atol": float(atol),
            "btol": float(btol),
            "damping": float(damping),
            "normal_residual_formula": (
                "||A^T(Ax-b) + damping^2*x||_2 / "
                "max(||A||_estimate*||Ax-b||_2 + damping^2*||x||_2, eps)"
                if float(damping) > 0.0 else
                "||A^T(Ax-b)||_2 / max(||A||_estimate * ||Ax-b||_2, eps)"
            ),
        },
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


def _normalized_poisson_deviance(
    prediction: torch.Tensor,
    measurement: torch.Tensor,
    eps: float,
) -> float:
    """Return dimensionless Poisson deviance normalized by ``2*sum(y)``."""

    raw = _poisson_deviance(prediction, measurement, eps)
    normalizer = max(2.0 * float(measurement.detach().sum().item()), float(eps))
    return raw / normalizer


def _validate_count_data(measurement: torch.Tensor, solver: str) -> None:
    if not torch.isfinite(measurement).all():
        raise ValueError(f"{solver} requires finite observations")
    if bool(torch.any(measurement < 0.0)):
        raise ValueError(
            f"{solver} requires nonnegative emission/count observations; "
            "log-domain or signed line-integral data are incompatible"
        )


def _statistical_parameter_errors(
    num_iterations: Any,
    initial_value: Any,
    min_value: Any,
    max_value: Any,
    eps: Any,
    *,
    block_size: Any = None,
    order_strategy: Any = "ordered",
    seed: Any = None,
) -> list[str]:
    errors = _common_parameter_errors(num_iterations, min_value, max_value)
    if not _finite_positive(initial_value):
        errors.append("initial_value must be a finite positive number")
    if min_value is not None and _finite_scalar(min_value) and float(min_value) < 0.0:
        errors.append("min_value must be nonnegative")
    if max_value is not None and _finite_scalar(max_value) and float(max_value) < 0.0:
        errors.append("max_value must be nonnegative")
    if not _finite_positive(eps):
        errors.append("eps must be a finite positive number")
    if block_size is not None and (
        isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0
    ):
        errors.append("block_size must be a positive integer or None")
    if not isinstance(order_strategy, str) or order_strategy not in {"ordered", "random"}:
        errors.append("order_strategy must be 'ordered' or 'random'")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        errors.append("seed must be an integer or None")
    return errors


def _is_valid_em_prediction(
    value: Any,
    expected_shape: tuple[int, ...],
) -> bool:
    return (
        isinstance(value, torch.Tensor)
        and tuple(value.shape) == expected_shape
        and _finite_tensor(value)
        and not bool(torch.any(value < 0.0))
    )


def _statistical_tolerance(control: SolveControl, name: str) -> float:
    explicit = getattr(control, name, None)
    if explicit is not None:
        return float(explicit)
    if control.tolerance is not None:
        return float(control.tolerance)
    return 1e-5


def _solve_em_detailed(
    algorithm: str,
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
    """Shared MLEM/OSEM loop with complete-epoch evidence only."""

    parameters: dict[str, Any] = {
        "num_iterations": num_iterations,
        "initial_value": initial_value,
        "min_value": min_value,
        "max_value": max_value,
        "eps": eps,
    }
    if algorithm == "osem":
        parameters.update({
            "block_size": block_size,
            "order_strategy": order_strategy,
            "seed": seed,
        })
    try:
        require_linear_operator(operator, algorithm)
        validate_measurement_shape(measurement, operator, algorithm)
    except (TypeError, ValueError) as error:
        return _parameter_failure_result(
            algorithm, measurement, operator, x_init=x_init, errors=[str(error)],
            max_iterations=None, parameters=parameters,
        )

    subset_values: list[Sequence[int]] | None = None
    if algorithm == "osem" and subset_indices is not None:
        try:
            subset_values = list(subset_indices)
        except (TypeError, ValueError) as error:
            return _parameter_failure_result(
                algorithm, measurement, operator, x_init=x_init, errors=[str(error)],
                max_iterations=None, parameters=parameters,
            )
        try:
            parameters["subset_indices"] = [
                tuple(int(value) for value in item) for item in subset_values
            ]
        except (TypeError, ValueError, OverflowError) as error:
            return _parameter_failure_result(
                algorithm, measurement, operator, x_init=x_init, errors=[str(error)],
                max_iterations=None, parameters=parameters,
            )
    parameter_errors = _statistical_parameter_errors(
        num_iterations, initial_value, min_value, max_value, eps,
        block_size=block_size, order_strategy=order_strategy, seed=seed,
    )
    if parameter_errors:
        return _parameter_failure_result(
            algorithm, measurement, operator, x_init=x_init, errors=parameter_errors,
            max_iterations=None, parameters=parameters,
        )
    if not _finite_tensor(measurement):
        return _numerical_result(
            algorithm, measurement, operator, x_init=x_init, max_iterations=None,
            reason="non_finite_measurement", parameters=parameters,
        )
    if bool(torch.any(measurement < 0.0)):
        return _parameter_failure_result(
            algorithm, measurement, operator, x_init=x_init,
            errors=[
                f"{algorithm} requires nonnegative emission/count observations; "
                "log-domain or signed line-integral data are incompatible"
            ],
            max_iterations=None, parameters=parameters,
        )
    try:
        control = resolve_control(
            control,
            default_iterations=int(num_iterations),
            default_tolerance=1e-5,
            callback=callback,
        )
    except (TypeError, ValueError) as error:
        return _parameter_failure_result(
            algorithm, measurement, operator, x_init=x_init, errors=[str(error)],
            max_iterations=None, parameters=parameters,
        )
    limit = min(int(num_iterations), int(control.max_iterations or num_iterations))
    parameters["num_iterations"] = int(num_iterations)
    parameters["max_iterations"] = limit
    try:
        x = prepare_initial_image(
            measurement, operator, x_init=x_init, initial_value=float(initial_value)
        )
        x = apply_box_constraints(x, min_value=min_value, max_value=max_value)
    except (TypeError, ValueError) as error:
        return _parameter_failure_result(
            algorithm, measurement, operator, x_init=x_init, errors=[str(error)],
            max_iterations=limit, parameters=parameters,
        )
    if not _finite_tensor(x) or bool(torch.any(x < 0.0)):
        return _numerical_result(
            algorithm, measurement, operator, x_init=x_init, max_iterations=limit,
            reason="non_finite_initial_reconstruction", parameters=parameters,
        )

    recorder = IterationRecorder(control, measurement, operator, algorithm=algorithm)
    recorder.set_initial(x)
    actual = 0

    def numerical_result(reason: str, predicted: torch.Tensor | None = None) -> SolveResult:
        recorder.mark_numerical_error(reason)
        return recorder.finish(
            x,
            actual_iterations=actual,
            status="numerical_error",
            stopping_reason=reason,
            predicted_measurement=predicted if _finite_tensor(predicted) else None,
            metadata={
                "criterion": "normalized_poisson_deviance_and_complete_epoch_iterate_change",
                "iteration_unit": "epochs",
                "native_termination_managed": True,
                **parameters,
            },
        )

    subsets: list[torch.Tensor] = []
    subset_operators: list[LinearOperator] = []
    sensitivities: list[torch.Tensor] = []
    sensitivity_masks: list[torch.Tensor] = []
    if algorithm == "osem":
        if block_size is None and subset_values is None:
            block_size = max(int(operator.range_shape[-2]) // 10, 1)
            parameters["block_size"] = block_size
        try:
            subsets = make_angle_subsets(
                num_angles=int(operator.range_shape[-2]),
                block_size=block_size,
                subset_indices=subset_values,
                order_strategy=order_strategy,
                seed=seed,
                device=measurement.device,
            )
            subset_operators = [make_subset_operator(operator, indices) for indices in subsets]
        except (NotImplementedError, TypeError, ValueError) as error:
            return _parameter_failure_result(
                algorithm, measurement, operator, x_init=x_init, errors=[str(error)],
                max_iterations=limit, parameters=parameters,
            )
        parameters["subset_sizes"] = [int(indices.numel()) for indices in subsets]
        for sub_operator, indices in zip(subset_operators, subsets):
            y_sub = select_measurement_subset(measurement, indices)
            try:
                sensitivity = sub_operator.adjoint(torch.ones_like(y_sub))
            except Exception as error:
                return numerical_result("subset_sensitivity_evaluation_failed")
            expected_domain = (measurement.shape[0], *tuple(operator.domain_shape))
            if (
                not isinstance(sensitivity, torch.Tensor)
                or tuple(sensitivity.shape) != expected_domain
                or not _finite_tensor(sensitivity)
                or bool(torch.any(sensitivity < 0.0))
                or bool(torch.all(sensitivity <= 0.0))
            ):
                return numerical_result("non_positive_subset_sensitivity")
            sensitivities.append(sensitivity.clamp_min(float(eps)))
            sensitivity_masks.append(sensitivity > float(eps))
    else:
        try:
            sensitivity = operator.adjoint(torch.ones_like(measurement))
        except Exception as error:
            return numerical_result("sensitivity_evaluation_failed")
        expected_domain = (measurement.shape[0], *tuple(operator.domain_shape))
        if (
            not isinstance(sensitivity, torch.Tensor)
            or tuple(sensitivity.shape) != expected_domain
            or not _finite_tensor(sensitivity)
            or bool(torch.any(sensitivity < 0.0))
            or bool(torch.all(sensitivity <= 0.0))
        ):
            return numerical_result("non_positive_sensitivity")
        sensitivities = [sensitivity.clamp_min(float(eps))]
        sensitivity_masks = [sensitivity > float(eps)]

    try:
        prediction = _finish_prediction(operator, x)
    except Exception as error:
        return numerical_result("initial_prediction_failed")
    expected_range = (measurement.shape[0], *tuple(operator.range_shape))
    if not _is_valid_em_prediction(prediction, expected_range):
        return numerical_result("invalid_initial_prediction", prediction)
    previous_deviance = _normalized_poisson_deviance(prediction, measurement, eps)
    if not _finite_scalar(previous_deviance):
        return numerical_result("non_finite_initial_poisson_deviance", prediction)

    measurement_norm = _global_norm(measurement)
    monitor = ConsecutiveStoppingMonitor(control)
    actual = 0
    converged = False
    cancelled = False
    terminal_status: str | None = None
    terminal_reason: str | None = None
    plateau_run = 0
    cycle_run = 0
    cycle_amplitude = 0.0
    cycle_patience = int((control.metadata or {}).get("osem_cycle_patience", control.stall_patience))
    cycle_tolerance = float(
        (control.metadata or {}).get(
            "osem_cycle_tolerance",
            max(control.stall_relative_iterate_tolerance, 1e-8),
        )
    )
    epoch_states: list[torch.Tensor] = []
    objective_tolerance = _statistical_tolerance(control, "relative_objective_tolerance")
    iterate_tolerance = _statistical_tolerance(control, "relative_iterate_tolerance")

    for epoch in range(1, limit + 1):
        previous = x
        subset_amplitude = 0.0
        try:
            if algorithm == "mlem":
                ratio = measurement / prediction.clamp_min(float(eps))
                correction = operator.adjoint(ratio)
                if (
                    tuple(correction.shape) != tuple(x.shape)
                    or not _finite_tensor(ratio)
                    or not _finite_tensor(correction)
                    or bool(torch.any(correction < 0.0))
                ):
                    return numerical_result("invalid_multiplicative_update", prediction)
                update_factor = correction / sensitivities[0]
                x = x * torch.where(
                    sensitivity_masks[0], update_factor, torch.ones_like(update_factor)
                )
                x = apply_box_constraints(x, min_value=min_value, max_value=max_value)
            else:
                for subset_index, (indices, sub_operator, sensitivity, sensitivity_mask) in enumerate(
                    zip(subsets, subset_operators, sensitivities, sensitivity_masks)
                ):
                    y_sub = select_measurement_subset(measurement, indices)
                    prediction_sub = sub_operator.forward(x)
                    expected_subset = (measurement.shape[0], *tuple(sub_operator.range_shape))
                    if not _is_valid_em_prediction(prediction_sub, expected_subset):
                        return numerical_result("invalid_subset_prediction", prediction)
                    ratio = y_sub / prediction_sub.clamp_min(float(eps))
                    correction = sub_operator.adjoint(ratio)
                    if (
                        tuple(correction.shape) != tuple(x.shape)
                        or not _finite_tensor(ratio)
                        or not _finite_tensor(correction)
                        or bool(torch.any(correction < 0.0))
                    ):
                        return numerical_result("invalid_subset_multiplicative_update", prediction)
                    before_subset = x
                    update_factor = correction / sensitivity
                    x = x * torch.where(
                        sensitivity_mask, update_factor, torch.ones_like(update_factor)
                    )
                    x = apply_box_constraints(x, min_value=min_value, max_value=max_value)
                    if not _finite_tensor(x) or bool(torch.any(x < 0.0)):
                        return numerical_result("non_finite_multiplicative_update", prediction)
                    subset_amplitude = max(subset_amplitude, _relative_change(before_subset, x))
        except Exception as error:
            return numerical_result("em_update_failed", prediction)
        if not _finite_tensor(x) or bool(torch.any(x < 0.0)):
            return numerical_result("non_finite_multiplicative_update", prediction)
        try:
            prediction = _finish_prediction(operator, x)
        except Exception as error:
            return numerical_result("prediction_evaluation_failed")
        if not _is_valid_em_prediction(prediction, expected_range):
            return numerical_result("invalid_prediction", prediction)
        change = _relative_change(previous, x)
        raw_deviance = _poisson_deviance(prediction, measurement, eps)
        normalized_deviance = _normalized_poisson_deviance(prediction, measurement, eps)
        deviance_change = abs(normalized_deviance - previous_deviance) / max(
            abs(previous_deviance), 1e-12
        )
        residual = prediction - measurement
        if not all(
            _finite_scalar(value)
            for value in (change, raw_deviance, normalized_deviance, deviance_change)
        ) or not _finite_tensor(residual):
            return numerical_result("non_finite_statistical_diagnostic", prediction)

        cycle_detected = False
        period_two_distance = None
        adjacent_distance = None
        if algorithm == "osem":
            if len(epoch_states) >= 2:
                period_two_distance = _relative_change(epoch_states[-2], x)
                adjacent_distance = _relative_change(epoch_states[-1], x)
                cycle_detected = (
                    period_two_distance <= cycle_tolerance
                    and adjacent_distance > max(5.0 * cycle_tolerance, 2.0 * iterate_tolerance, 1e-12)
                )
            fixed_point_with_subset_motion = (
                change <= cycle_tolerance
                and subset_amplitude > max(10.0 * cycle_tolerance, 10.0 * iterate_tolerance, 1e-12)
            )
            cycle_detected = cycle_detected or fixed_point_with_subset_motion
            cycle_run = cycle_run + 1 if cycle_detected else 0
            cycle_amplitude = max(subset_amplitude, float(adjacent_distance or 0.0))
            epoch_states.append(x.detach().clone())
            if len(epoch_states) > 2:
                epoch_states.pop(0)

        discrepancy_ok = (
            control.discrepancy_target is None
            or normalized_deviance <= float(control.discrepancy_target)
        )
        native_ok = (
            deviance_change <= objective_tolerance
            and change <= iterate_tolerance
        )
        criteria: dict[str, bool] = {
            "normalized_poisson_deviance_change": deviance_change <= objective_tolerance,
            "relative_iterate_change": change <= iterate_tolerance,
        }
        if control.discrepancy_target is not None:
            criteria = {"discrepancy": discrepancy_ok, **criteria}
        if algorithm == "osem" and _policy_active(control):
            criteria["osem_cycle_free"] = not cycle_detected
        decision = monitor.observe(
            epoch,
            criteria=criteria,
            relative_change=change,
            monitor_value=normalized_deviance,
        )
        converged = decision.converged
        if _policy_active(control):
            if decision.diverged:
                terminal_status = "diverged"
                terminal_reason = "persistent_poisson_deviance_increase"
            elif (
                decision.checked
                and control.stall_enabled
                and control.discrepancy_target is not None
                and not discrepancy_ok
                and native_ok
            ):
                plateau_run += 1
                if plateau_run >= control.stall_patience:
                    terminal_status = "stalled"
                    terminal_reason = "poisson_deviance_plateau_before_discrepancy"
            else:
                plateau_run = 0
            if (
                terminal_status is None
                and algorithm == "osem"
                and control.stall_enabled
                and decision.checked
                and cycle_run >= cycle_patience
            ):
                terminal_status = "stalled"
                terminal_reason = "osem_subset_cycle_detected"
        actual = epoch
        metadata = {
            "checked": decision.checked,
            "complete_epoch": True,
            "epoch_boundary": True,
            "poisson_deviance": raw_deviance,
            "normalized_poisson_deviance": normalized_deviance,
            "previous_normalized_poisson_deviance": previous_deviance,
            "normalized_poisson_deviance_change": deviance_change,
            "poisson_deviance_normalization": "2*sum_observed",
            "relative_epoch_change": change,
            "normalized_poisson_deviance_tolerance": objective_tolerance,
            "relative_iterate_tolerance": iterate_tolerance,
            "discrepancy_metric": "normalized_poisson_deviance",
            "discrepancy_satisfied": discrepancy_ok,
            "plateau_run": plateau_run,
            "cycle_detected": cycle_detected,
            "cycle_free": not cycle_detected,
            "cycle_run": cycle_run,
            "cycle_amplitude": cycle_amplitude,
            "subset_amplitude": subset_amplitude,
            "native_termination_managed": True,
        }
        if not _safe_record(
            recorder,
            epoch,
            x,
            residual=residual,
            objective=raw_deviance,
            algorithm_residual=normalized_deviance,
            stopping_candidate=bool(decision.checked or converged or terminal_status),
            consecutive_criteria_count=decision.consecutive,
            criteria=criteria,
            native_criterion_name="normalized_poisson_deviance_change",
            native_criterion_value=deviance_change,
            native_criterion_threshold=objective_tolerance,
            epoch=epoch,
            subset_count=len(subsets) if algorithm == "osem" else None,
            metadata=metadata,
        ):
            cancelled = True
            break
        previous_deviance = normalized_deviance
        if terminal_status or (converged and control.stop_on_convergence):
            break

    final_residual_tensor = prediction - measurement
    final_residual = _normalized(_global_norm(final_residual_tensor), measurement_norm)
    status = terminal_status or (
        "cancelled" if cancelled else _finish_status(
            actual=actual,
            limit=limit,
            converged=converged,
            cancelled=cancelled,
            numerical_failure=recorder.numerical_failure,
        )
    )
    converged_reason = (
        "poisson_deviance_and_relative_epoch_change_patience"
        if algorithm == "osem" else
        "poisson_deviance_and_relative_iterate_change_patience"
    )
    stopping_reason = (
        terminal_reason
        if terminal_reason else
        converged_reason if converged and actual < limit else
        "callback_cancelled" if cancelled else
        "maximum_epochs_reached"
    )
    return recorder.finish(
        x,
        actual_iterations=actual,
        status=status,
        stopping_reason=stopping_reason,
        final_residual=final_residual,
        final_objective=_poisson_deviance(prediction, measurement, eps),
        predicted_measurement=prediction,
        metadata={
            "likelihood": "poisson_emission_style",
            "criterion": "normalized_poisson_deviance_and_complete_epoch_iterate_change",
            "poisson_deviance_normalization": "2*sum_observed",
            "complete_epoch_count": actual,
            "subset_count": len(subsets) if algorithm == "osem" else None,
            "normalized_poisson_deviance": previous_deviance,
            "normalized_poisson_deviance_tolerance": objective_tolerance,
            "relative_iterate_tolerance": iterate_tolerance,
            "cycle_detected": bool(algorithm == "osem" and cycle_run >= cycle_patience),
            "cycle_run": cycle_run,
            "cycle_amplitude": cycle_amplitude,
            "native_termination_managed": True,
            "max_iterations": limit,
            **parameters,
        },
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
    return _solve_em_detailed(
        "mlem", operator, measurement,
        num_iterations=num_iterations,
        x_init=x_init,
        initial_value=initial_value,
        min_value=min_value,
        max_value=max_value,
        eps=eps,
        control=control,
        callback=callback,
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
    return _solve_em_detailed(
        "osem", operator, measurement,
        num_iterations=num_iterations,
        block_size=block_size,
        subset_indices=subset_indices,
        order_strategy=order_strategy,
        seed=seed,
        x_init=x_init,
        initial_value=initial_value,
        min_value=min_value,
        max_value=max_value,
        eps=eps,
        control=control,
        callback=callback,
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
    parameter_errors = _common_parameter_errors(num_iterations, min_value, max_value)
    if not _finite_nonnegative(reg_strength):
        parameter_errors.append("reg_strength must be a finite nonnegative number")
    if not _finite_nonnegative(tolerance):
        parameter_errors.append("tolerance must be a finite nonnegative number")
    if not _finite_positive(eps):
        parameter_errors.append("eps must be a finite positive number")
    if regularization_operator is not None and not isinstance(regularization_operator, LinearOperator):
        parameter_errors.append("regularization_operator must be a LinearOperator or None")
    elif (
        regularization_operator is not None
        and tuple(regularization_operator.domain_shape) != tuple(operator.domain_shape)
    ):
        parameter_errors.append(
            "regularization_operator.domain_shape must match operator.domain_shape"
        )
    if parameter_errors:
        return _parameter_failure_result(
            "tikhonov", measurement, operator, x_init=x_init,
            errors=parameter_errors,
            max_iterations=num_iterations if isinstance(num_iterations, int) and not isinstance(num_iterations, bool) else None,
            parameters={
                "num_iterations": num_iterations,
                "reg_strength": reg_strength,
                "tolerance": tolerance,
                "min_value": min_value,
                "max_value": max_value,
                "eps": eps,
            },
        )
    control = resolve_control(control, default_iterations=int(num_iterations), default_tolerance=float(tolerance), callback=callback)
    limit = min(int(num_iterations), int(control.max_iterations or num_iterations))
    regularizer = TikhonovRegularizer(regularization_operator)
    strength = float(reg_strength)
    parameters = {
        "num_iterations": int(num_iterations),
        "reg_strength": strength,
        "tolerance": float(tolerance),
        "min_value": min_value,
        "max_value": max_value,
        "eps": float(eps),
    }
    if not _finite_tensor(measurement):
        return _numerical_result(
            "tikhonov", measurement, operator, x_init=x_init, max_iterations=limit,
            reason="non_finite_measurement", parameters=parameters,
        )
    x = prepare_initial_image(measurement, operator, x_init=x_init, initial_value=0.0)
    if not _finite_tensor(x):
        return _numerical_result(
            "tikhonov", measurement, operator, x_init=x_init, max_iterations=limit,
            reason="non_finite_initial_reconstruction", parameters=parameters,
        )
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
    if not (
        _finite_tensor(prediction)
        and _finite_tensor(rhs)
        and _finite_tensor(normal_residual)
    ):
        recorder.mark_numerical_error("non_finite_solver_state")
    direction = normal_residual.clone()
    residual_sq = _sqnorm(normal_residual)
    rhs_norm = max(_global_norm(rhs), 1.0)
    converged = False
    cancelled = False
    terminal_status = None
    terminal_reason = None
    for iteration in range(1, limit + 1):
        if recorder.numerical_failure:
            break
        if converged and control.stop_on_convergence:
            break
        normal_value = _global_norm(normal_residual)
        if not _policy_active(control) and float(control.tolerance or 0.0) > 0.0 and _normalized(normal_value, rhs_norm) <= float(control.tolerance):
            converged = True
            break
        direction_prediction = operator.forward(direction)
        normal_direction = operator.adjoint(direction_prediction) + strength * regularizer.gradient(direction)
        denominator = (direction * normal_direction).reshape(direction.shape[0], -1).sum(dim=1)
        if (
            not _finite_tensor(direction_prediction)
            or not _finite_tensor(normal_direction)
            or not bool(torch.isfinite(denominator).all())
        ):
            recorder.mark_numerical_error("non_finite_normal_equation")
            break
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
                # A stationary normal equation with no usable CG direction
                # is benign.  It may still miss the discrepancy target (for
                # instance, an inconsistent or zero operator), so do not
                # promote it to convergence.
                if _policy_active(control):
                    residual = prediction - measurement
                    data_relative = _normalized(_global_norm(residual), _global_norm(measurement))
                    regularization_gradient = regularizer.gradient(x)
                    reg_denominator = _global_norm(rhs) + strength * _global_norm(regularization_gradient)
                    reg_normalized = normal_value / max(reg_denominator, float(eps))
                    normal_tolerance_policy = _control_threshold(
                        control, "normalized_normal_residual_tolerance"
                    )
                    iterate_tolerance_policy = _control_threshold(
                        control, "relative_iterate_tolerance"
                    )
                    criteria = {
                        "regularized_normal_residual": reg_normalized <= normal_tolerance_policy,
                        "relative_iterate_change": 0.0 <= iterate_tolerance_policy,
                    }
                    if control.discrepancy_target is not None:
                        criteria = {
                            "discrepancy": data_relative <= float(control.discrepancy_target),
                            **criteria,
                        }
                    objective = _objective(residual) + strength * float(regularizer.value(x).sum().item())
                    decision = monitor.observe(
                        iteration,
                        criteria=criteria,
                        relative_change=0.0,
                        monitor_value=objective,
                    )
                    converged = decision.converged
                    terminal_status, terminal_reason = _policy_status(decision)
                    actual = iteration
                    if not _safe_record(
                        recorder,
                        iteration,
                        x,
                        residual=residual,
                        objective=objective,
                        algorithm_residual=reg_normalized,
                        stopping_candidate=converged,
                        consecutive_criteria_count=decision.consecutive,
                        criteria=criteria,
                        native_criterion_name="normalized_regularized_normal_residual",
                        native_criterion_value=reg_normalized,
                        native_criterion_threshold=normal_tolerance_policy,
                        metadata={
                            "normal_residual": normal_value,
                            "normal_residual_denominator": reg_denominator,
                            "complete_objective": True,
                            "regularization_operator": "identity" if regularization_operator is None else "linear_operator",
                            "regularized_normal_residual_formula": (
                                "||A^T(Ax-b) + lambda L^T Lx||_2 / "
                                "max(||A^T b||_2 + lambda ||L^T Lx||_2, eps)"
                            ),
                            "breakdown": "stationary_normal_equation",
                        },
                    ):
                        cancelled = True
                        break
                    if terminal_status or (converged and control.stop_on_convergence):
                        break
                    if control.discrepancy_target is not None and not criteria["discrepancy"]:
                        terminal_status = "stalled"
                        terminal_reason = "stalled_before_discrepancy"
                        break
                    continue
                # Preserve the fixed-compute legacy accounting contract for
                # exact/stationary plateaus: each counted iteration still
                # produces a diagnostic row and consumes its native operator
                # calls.  This keeps the detailed wrapper comparable with
                # the original solver even though no further CG step exists.
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
            recorder.mark_numerical_error("normal_equation_breakdown")
            break
        alpha = residual_sq / denominator
        if not bool(torch.isfinite(alpha).all()):
            recorder.mark_numerical_error("non_finite_normal_equation_step")
            break
        previous = x
        x = x + alpha.reshape((alpha.shape[0],) + (1,) * (x.ndim - 1)) * direction
        x = apply_box_constraints(x, min_value=min_value, max_value=max_value)
        if cache_prediction:
            next_normal_residual = normal_residual - alpha.reshape((alpha.shape[0],) + (1,) * (normal_residual.ndim - 1)) * normal_direction
            next_residual_sq = _sqnorm(next_normal_residual)
            beta = next_residual_sq / residual_sq.clamp_min(float(eps))
            direction = next_normal_residual + beta.reshape((beta.shape[0],) + (1,) * (direction.ndim - 1)) * direction
            normal_residual = next_normal_residual
            residual_sq = next_residual_sq
            prediction = prediction + alpha.reshape((alpha.shape[0],) + (1,) * (prediction.ndim - 1)) * direction_prediction
            # The unconstrained CG recurrence is exact for this linear
            # normal equation.  With a supplied box constraint the projection
            # destroys that recurrence; recompute the native state instead.
        else:
            prediction = operator.forward(x)
            exact_normal_residual = rhs - (
                operator.adjoint(prediction) + strength * regularizer.gradient(x)
            )
            exact_residual_sq = _sqnorm(exact_normal_residual)
            if not _finite_tensor(exact_normal_residual) or not bool(torch.isfinite(exact_residual_sq).all()):
                recorder.mark_numerical_error("non_finite_normal_equation")
                break
            beta = exact_residual_sq / residual_sq.clamp_min(float(eps))
            direction = exact_normal_residual + beta.reshape((beta.shape[0],) + (1,) * (direction.ndim - 1)) * direction
            if not _finite_tensor(direction) or not bool(torch.isfinite(beta).all()):
                recorder.mark_numerical_error("non_finite_normal_equation_step")
                break
            normal_residual = exact_normal_residual
            residual_sq = exact_residual_sq
        residual = prediction - measurement
        objective = _objective(residual) + strength * float(regularizer.value(x).sum().item())
        normal_value = _global_norm(normal_residual)
        normal_relative = _normalized(normal_value, rhs_norm)
        change = _relative_change(previous, x)
        data_relative = _normalized(_global_norm(residual), _global_norm(measurement))
        regularization_gradient = regularizer.gradient(x)
        reg_denominator = _global_norm(rhs) + strength * _global_norm(regularization_gradient)
        reg_normalized = normal_value / max(reg_denominator, float(eps))
        normal_tolerance_policy = _control_threshold(
            control, "normalized_normal_residual_tolerance"
        )
        iterate_tolerance_policy = _control_threshold(
            control, "relative_iterate_tolerance"
        )
        criteria = {
            "regularized_normal_residual": reg_normalized <= normal_tolerance_policy,
            "relative_iterate_change": change <= iterate_tolerance_policy,
        }
        if control.discrepancy_target is not None:
            criteria = {
                "discrepancy": data_relative <= float(control.discrepancy_target),
                **criteria,
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
            native_criterion_threshold=(
                normal_tolerance_policy
                if _policy_active(control)
                else control.normalized_normal_residual_tolerance
            ),
            metadata={
                "normal_residual": normal_value,
                "normal_residual_denominator": reg_denominator,
                "complete_objective": True,
                "regularization_operator": "identity" if regularization_operator is None else "linear_operator",
                "regularized_normal_residual_formula": (
                    "||A^T(Ax-b) + lambda L^T Lx||_2 / "
                    "max(||A^T b||_2 + lambda ||L^T Lx||_2, eps)"
                ),
            },
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
            "discrepancy_regularized_normal_and_iterate_patience" if status == "converged" and _policy_active(control) else
            "normal_residual_and_iterate_tolerance" if status == "converged" else
            "callback_cancelled" if cancelled else
            "maximum_iterations_reached"
        ),
        final_residual=_normalized(_global_norm(residual), _global_norm(measurement)),
        final_objective=_objective(residual) + strength * float(regularizer.value(x).sum().item()),
        predicted_measurement=prediction,
        metadata={
            "criterion": "normalized_regularized_normal_residual_and_relative_iterate_change",
            "regularization_strength": strength,
            "regularization_operator": "identity" if regularization_operator is None else "linear_operator",
            "regularized_normal_residual_formula": (
                "||A^T(Ax-b) + lambda L^T Lx||_2 / "
                "max(||A^T b||_2 + lambda ||L^T Lx||_2, eps)"
            ),
        },
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
    parameter_errors = _common_parameter_errors(num_iterations, min_value, max_value)
    if not _finite_nonnegative(reg_strength):
        parameter_errors.append("reg_strength must be a finite nonnegative number")
    if not _finite_nonnegative(tolerance):
        parameter_errors.append("tolerance must be a finite nonnegative number")
    if isinstance(power_iterations, bool) or not isinstance(power_iterations, int) or power_iterations <= 0:
        parameter_errors.append("power_iterations must be a positive integer")
    if len(tuple(operator.domain_shape)) < 2:
        parameter_errors.append("tv_fista requires at least two spatial domain dimensions")
    if step_size is not None and not _finite_positive(step_size):
        parameter_errors.append("step_size must be a finite positive number")
    if parameter_errors:
        return _parameter_failure_result(
            "tv_fista", measurement, operator, x_init=x_init,
            errors=parameter_errors,
            max_iterations=num_iterations if isinstance(num_iterations, int) and not isinstance(num_iterations, bool) else None,
            parameters={
                "num_iterations": num_iterations,
                "reg_strength": reg_strength,
                "step_size": step_size,
                "tolerance": tolerance,
                "power_iterations": power_iterations,
                "min_value": min_value,
                "max_value": max_value,
            },
        )
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
    parameters = {
        "num_iterations": int(num_iterations),
        "reg_strength": float(reg_strength),
        "step_size": step_size,
        "tolerance": float(tolerance),
        "power_iterations": int(power_iterations),
        "min_value": min_value,
        "max_value": max_value,
        "tv_mode": tv.mode,
        "tv_num_iterations": int(tv.num_iterations),
        "tv_tolerance": float(tv.tolerance),
    }
    if not _finite_tensor(measurement):
        return _numerical_result(
            "tv_fista", measurement, operator, x_init=x_init, max_iterations=limit,
            reason="non_finite_measurement", parameters=parameters,
        )
    x = prepare_initial_image(measurement, operator, x_init=x_init, initial_value=0.0)
    if not _finite_tensor(x):
        return _numerical_result(
            "tv_fista", measurement, operator, x_init=x_init, max_iterations=limit,
            reason="non_finite_initial_reconstruction", parameters=parameters,
        )
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

    if _policy_active(control):
        # The policy path records the state that is actually returned by the
        # proximal update.  That requires one additional A/A^T pair per
        # outer iteration: a forward/adjoint pair at the accelerated point
        # drives the update, and a second pair evaluates the returned state.
        # Keeping these calls explicit is preferable to labelling a momentum
        # state as the reconstruction and makes endpoint confirmation exact.
        current_prediction = operator.forward(momentum)
        current_residual = current_prediction - measurement
        current_gradient = operator.adjoint(current_residual)
        if not (
            _finite_tensor(current_prediction)
            and _finite_tensor(current_residual)
            and _finite_tensor(current_gradient)
        ):
            recorder.mark_numerical_error("non_finite_initial_solver_state")
        if not recorder.numerical_failure:
            previous_objective = _objective(current_residual) + float(reg_strength) * float(tv.value(x).sum().item())
        native_plateau_run = 0
        for iteration in range(1, limit + 1):
            if recorder.numerical_failure:
                break
            if converged and control.stop_on_convergence:
                break
            candidate = momentum - step * current_gradient
            next_x = tv.proximal(candidate, step * float(reg_strength))
            next_x = apply_box_constraints(next_x, min_value=min_value, max_value=max_value)
            next_acceleration = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * acceleration * acceleration))
            momentum_scale = (acceleration - 1.0) / next_acceleration
            next_momentum = next_x + momentum_scale * (next_x - x)
            next_prediction = operator.forward(next_x)
            next_residual = next_prediction - measurement
            next_gradient = operator.adjoint(next_residual)
            if not (
                _finite_tensor(next_x)
                and _finite_tensor(next_prediction)
                and _finite_tensor(next_residual)
                and _finite_tensor(next_gradient)
            ):
                recorder.mark_numerical_error("non_finite_returned_state")
                break
            next_objective = _objective(next_residual) + float(reg_strength) * float(tv.value(next_x).sum().item())
            change = _relative_change(x, next_x)
            data_relative = _normalized(_global_norm(next_residual), measurement_norm)
            prox_point = tv.proximal(
                next_x - step * next_gradient,
                step * float(reg_strength),
            )
            mapping = (next_x - prox_point) / step
            mapping_norm = _global_norm(mapping)
            mapping_denominator = max(_global_norm(next_x) / step, 1e-12)
            normalized_mapping = mapping_norm / mapping_denominator
            objective_change = abs(next_objective - float(previous_objective)) / max(
                abs(float(previous_objective)), 1e-12
            )
            mapping_tolerance_policy = _control_threshold(
                control, "prox_gradient_mapping_tolerance"
            )
            objective_tolerance_policy = _control_threshold(
                control, "relative_objective_tolerance"
            )
            criteria = {
                "prox_gradient_mapping": normalized_mapping <= mapping_tolerance_policy,
                "relative_composite_objective_change": objective_change <= objective_tolerance_policy,
            }
            if control.discrepancy_target is not None:
                criteria = {
                    "discrepancy": data_relative <= float(control.discrepancy_target),
                    **criteria,
                }
            decision = monitor.observe(
                iteration,
                criteria=criteria,
                relative_change=change,
                monitor_value=next_objective,
            )
            # A stationary TV fixed point can satisfy both native optimality
            # tests forever while an incompatible discrepancy target remains
            # unmet.  The shared monitor intentionally requires an unmet
            # native criterion for its generic stall path, which is not true
            # for this case.  Count this more specific, checked-only plateau
            # locally so one admissible non-monotone objective increase does
            # not get reclassified as divergence.
            tv_native_plateau = (
                decision.checked
                and control.stall_enabled
                and control.discrepancy_target is not None
                and not bool(criteria.get("discrepancy", True))
                and bool(criteria.get("prox_gradient_mapping", False))
                and bool(criteria.get("relative_composite_objective_change", False))
                and change <= float(control.stall_relative_iterate_tolerance)
            )
            if decision.checked:
                native_plateau_run = native_plateau_run + 1 if tv_native_plateau else 0
            converged = decision.converged
            terminal_status, terminal_reason = _policy_status(decision)
            if terminal_status is None and native_plateau_run >= control.stall_patience:
                terminal_status = "stalled"
                terminal_reason = "stalled_before_discrepancy"
            actual = iteration
            if not _safe_record(
                recorder,
                iteration,
                next_x,
                residual=next_residual,
                objective=next_objective,
                algorithm_residual=normalized_mapping,
                step_size=step,
                stopping_candidate=converged,
                consecutive_criteria_count=decision.consecutive,
                criteria=criteria,
                native_criterion_name="normalized_prox_gradient_mapping",
                native_criterion_value=normalized_mapping,
                native_criterion_threshold=(
                    mapping_tolerance_policy
                    if _policy_active(control)
                    else control.prox_gradient_mapping_tolerance
                ),
                metadata={
                    "complete_objective": True,
                    "objective_state": "returned",
                    "mapping_state": "returned",
                    "relative_composite_objective_change": objective_change,
                    "normalized_prox_gradient_mapping": normalized_mapping,
                    "prox_gradient_mapping_denominator": mapping_denominator,
                    "native_plateau": tv_native_plateau,
                    "native_plateau_consecutive": native_plateau_run,
                    "fista_momentum": float(momentum_scale),
                },
            ):
                cancelled = True
            x = next_x
            previous_objective = next_objective
            if (
                cancelled
                or terminal_status
                or (converged and control.stop_on_convergence)
                or iteration >= limit
            ):
                break
            momentum = next_momentum
            acceleration = next_acceleration
            current_prediction = operator.forward(momentum)
            current_residual = current_prediction - measurement
            current_gradient = operator.adjoint(current_residual)
            if not (
                _finite_tensor(current_prediction)
                and _finite_tensor(current_residual)
                and _finite_tensor(current_gradient)
            ):
                recorder.mark_numerical_error("non_finite_momentum_state")
                break
        prediction = _finish_prediction(operator, x)
        residual = prediction - measurement
        resources = {"prox_iterations": int(actual) * int(tv.num_iterations), "prox_configured_iterations": int(tv.num_iterations)}
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
                "discrepancy_prox_gradient_and_objective_patience" if status == "converged" else
                "callback_cancelled" if cancelled else
                "maximum_iterations_reached"
            ),
            final_residual=_normalized(_global_norm(residual), measurement_norm),
            final_objective=_objective(residual) + float(reg_strength) * float(tv.value(x).sum().item()),
            predicted_measurement=prediction,
            resources=resources,
            metadata={
                "criterion": "returned_state_prox_gradient_mapping_and_composite_objective",
                "step_size": step,
                "operator_norm_squared": lipschitz,
                "power_iterations": int(power_iterations) if step_size is None else 0,
                "tv_mode": tv.mode,
                "tv_tolerance": tv.tolerance,
                "prox_gradient_mapping_formula": (
                    "G_t(x) = (x - prox_(t*lambda*TV)(x - t*A^T(Ax-b))) / t"
                ),
                "native_plateau_stall": {
                    "enabled": bool(control.stall_enabled),
                    "patience": int(control.stall_patience),
                    "relative_iterate_tolerance": float(control.stall_relative_iterate_tolerance),
                    "criteria": (
                        "discrepancy_unmet_and_prox_gradient_mapping_and_"
                        "relative_composite_objective_change_and_relative_iterate_change"
                    ),
                    "consecutive": int(native_plateau_run),
                },
            },
        )

    # Preserve the established one-gradient-forward legacy detailed path when
    # no stopping policy is requested.  It remains API-compatible and keeps
    # its historical operator-call profile; policy runs above provide the
    # returned-state evidence required for strict convergence.
    for iteration in range(1, limit + 1):
        if recorder.numerical_failure:
            break
        if converged and control.stop_on_convergence:
            break
        residual_at_momentum = operator.forward(momentum) - measurement
        gradient = operator.adjoint(residual_at_momentum)
        candidate = momentum - step * gradient
        next_x = tv.proximal(candidate, step * float(reg_strength))
        next_x = apply_box_constraints(next_x, min_value=min_value, max_value=max_value)
        if not (
            _finite_tensor(residual_at_momentum)
            and _finite_tensor(gradient)
            and _finite_tensor(next_x)
        ):
            recorder.mark_numerical_error("non_finite_solver_state")
            break
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
            "discrepancy": (
                control.discrepancy_target is not None
                and data_relative <= float(control.discrepancy_target)
            ),
            "prox_gradient_mapping": normalized_mapping <= _control_threshold(
                control, "prox_gradient_mapping_tolerance"
            ),
            "relative_composite_objective_change": objective_change is not None and objective_change <= _control_threshold(
                control, "relative_objective_tolerance"
            ),
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
            "discrepancy_prox_gradient_and_objective_patience" if status == "converged" and _policy_active(control) else
            "complete_objective_and_iterate_tolerance" if status == "converged" else
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

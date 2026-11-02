"""Shared iteration diagnostics, stopping and numerical failure handling."""
from __future__ import annotations
from typing import Any, Callable, Iterable, Mapping
from inv_framework.core.operator import LinearOperator
from inv_framework.solvers.base import IterationRecorder, IterationRecord, SolveControl, SolveResult
from inv_framework.solvers._utils import prepare_initial_image
import math
import torch


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
        resources={**dict(resources or {})},
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
        resources={},
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

"""Canonical convergence reports and post-run validation for CT reconstructions.

The runtime historically exposed several names for equivalent outcomes.  This
module is the compatibility boundary: legacy inputs remain parseable, while
core CT reports use one canonical status vocabulary and retain the evidence
needed to audit a terminal decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any, Iterable, Mapping

import torch

from .instrumentation import OperatorBudgetExceeded


CONVERGENCE_SCHEMA_VERSION = "ct.convergence.v1"

# This is a status-normalization contract, not a second registry.  Keeping it
# local avoids making the low-level diagnostics module depend on solver specs.
CORE_CT_ALGORITHMS = frozenset({
    "fbp", "sirt", "landweber", "cgls", "lsqr", "sart", "os_sart",
    "mlem", "osem", "tikhonov", "tv_fista", "fdk",
})


class ConvergenceStatus(str, Enum):
    CONVERGED = "converged"
    STALLED = "stalled"
    DIVERGED = "diverged"
    MAX_ITERATIONS = "max_iterations"
    NUMERICAL_ERROR = "numerical_error"
    INVALID_PARAMETERS = "invalid_parameters"
    COMPLETED_VALID = "completed_valid"
    UNAVAILABLE = "unavailable"
    CANCELLED = "cancelled"
    RESOURCE_EXHAUSTED = "resource_exhausted"

    # Migration-only values.  They remain parseable for old callers but are
    # normalized before a core CT report is serialized.
    PARTIAL = "partial"
    NUMERICAL_FAILURE = "numerical_failure"
    NON_ITERATIVE_COMPLETED = "non_iterative_completed"
    NOT_APPLICABLE = "not_applicable"


CANONICAL_CONVERGENCE_STATUSES = (
    "converged", "stalled", "diverged", "max_iterations", "numerical_error",
    "invalid_parameters", "completed_valid", "unavailable", "cancelled",
    "resource_exhausted",
)

LEGACY_STATUS_ALIASES = {
    "numerical_failure": "numerical_error",
    "non_iterative_completed": "completed_valid",
    "not_applicable": "completed_valid",
}

STATUS_REASON_CLASSES = {
    "converged": "convergence",
    "stalled": "stagnation",
    "diverged": "divergence",
    "max_iterations": "iteration_budget",
    "numerical_error": "numerical",
    "invalid_parameters": "invalid_parameters",
    "completed_valid": "completed_valid",
    "unavailable": "capability",
    "cancelled": "cancelled",
    "resource_exhausted": "resource_budget",
    "partial": "insufficient_evidence",
}

DEFAULT_REASON_CODES = {
    "converged": "criterion_satisfied",
    "stalled": "stalled_before_convergence",
    "diverged": "persistent_worsening",
    "max_iterations": "maximum_iterations_reached",
    "numerical_error": "numerical_error",
    "invalid_parameters": "parameter_validation_failed",
    "completed_valid": "direct_reconstruction_valid",
    "unavailable": "backend_unavailable",
    "cancelled": "cancelled_by_callback",
    "resource_exhausted": "resource_budget_exhausted",
    "partial": "insufficient_termination_evidence",
}


def is_core_algorithm(algorithm: str | None) -> bool:
    return algorithm is not None and str(algorithm) in CORE_CT_ALGORITHMS


def status_reason_class(status: ConvergenceStatus | str) -> str:
    """Return the single reason class assigned to a canonical status."""

    value = status.value if isinstance(status, ConvergenceStatus) else str(status)
    value = LEGACY_STATUS_ALIASES.get(value, value)
    try:
        return STATUS_REASON_CLASSES[value]
    except KeyError as error:
        raise ValueError(f"unknown convergence status {status!r}") from error


def normalize_status(
    status: ConvergenceStatus | str,
    *,
    algorithm: str | None = None,
    iterations: int | None = None,
    max_iterations: int | None = None,
    direct: bool = False,
) -> ConvergenceStatus:
    """Normalize one status without promoting a budget stop to convergence."""

    raw = status.value if isinstance(status, ConvergenceStatus) else str(status)
    value = LEGACY_STATUS_ALIASES.get(raw, raw)
    if raw == "converged" and max_iterations is not None and iterations is not None and int(iterations) >= int(max_iterations):
        value = "max_iterations"
    elif raw == "partial" and direct:
        value = "completed_valid"
    elif raw == "partial" and is_core_algorithm(algorithm):
        if max_iterations is not None and iterations is not None and int(iterations) >= int(max_iterations):
            value = "max_iterations"
        else:
            value = "numerical_error"
    try:
        return ConvergenceStatus(value)
    except ValueError as error:
        raise ValueError(f"unknown convergence status {status!r}") from error


canonicalize_status = normalize_status
normalize_convergence_status = normalize_status


def _json_safe(value: Any) -> Any:
    """Make diagnostic payloads JSON-compatible without NaN/Inf literals."""

    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, float):
        return value if isfinite(value) else None
    if hasattr(value, "item") and value.__class__.__module__.startswith("numpy"):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _evidence_rows(rows: list[Mapping[str, Any]], patience: int) -> tuple[dict[str, Any], ...]:
    if not rows:
        return ()
    return tuple(_json_safe(dict(row)) for row in rows[-max(1, int(patience)):])


def _report_evidence(rows: list[Mapping[str, Any]], patience: int) -> dict[str, Any]:
    return {
        "terminal": _json_safe(dict(rows[-1])) if rows else None,
        "patience_window": list(_evidence_rows(rows, patience)),
    }


@dataclass(frozen=True)
class ConvergenceReport:
    """Portable convergence state independent of a solver implementation."""

    status: ConvergenceStatus | str
    stopping_reason: str = ""
    iterations: int = 0
    max_iterations: int | None = None
    final_residual: float | None = None
    final_objective: float | None = None
    relative_iterate_change: float | None = None
    trajectory: tuple[dict[str, Any], ...] = ()
    warnings: tuple[str, ...] = ()
    failure_reason: str | None = None
    schema_version: str = CONVERGENCE_SCHEMA_VERSION
    algorithm: str | None = None
    reason_code: str | None = None
    reason_class: str | None = None
    legacy_status: str | None = None
    terminal_evidence: Mapping[str, Any] = field(default_factory=dict)
    endpoint_confirmation: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        raw = self.status.value if isinstance(self.status, ConvergenceStatus) else str(self.status)
        normalized = normalize_status(
            raw,
            algorithm=self.algorithm,
            iterations=self.iterations,
            max_iterations=self.max_iterations,
            direct=bool(self.algorithm in {"fbp", "fdk"}),
        )
        object.__setattr__(self, "status", normalized)
        if raw != normalized.value and raw in (*LEGACY_STATUS_ALIASES, "partial"):
            object.__setattr__(self, "legacy_status", raw)
        reason = str(self.stopping_reason or DEFAULT_REASON_CODES[normalized.value])
        if normalized == ConvergenceStatus.MAX_ITERATIONS and raw != normalized.value:
            reason = DEFAULT_REASON_CODES[normalized.value]
        object.__setattr__(self, "stopping_reason", reason)
        object.__setattr__(self, "reason_code", str(self.reason_code or reason))
        object.__setattr__(self, "reason_class", status_reason_class(normalized))
        object.__setattr__(self, "iterations", max(0, int(self.iterations)))

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "stopping_reason": self.stopping_reason,
            "reason_code": self.reason_code,
            "reason_class": self.reason_class,
            "legacy_status": self.legacy_status,
            "algorithm": self.algorithm,
            "iterations": int(self.iterations),
            "max_iterations": self.max_iterations,
            "final_residual": self.final_residual,
            "final_objective": self.final_objective,
            "relative_iterate_change": self.relative_iterate_change,
            "trajectory": [dict(item) for item in self.trajectory],
            "warnings": list(self.warnings),
            "failure_reason": self.failure_reason,
            "terminal_evidence": dict(self.terminal_evidence),
            "endpoint_confirmation": dict(self.endpoint_confirmation),
        }
        return _json_safe(payload)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConvergenceReport":
        """Parse current reports and legacy status payloads."""

        value = dict(payload)
        allowed = {
            "schema_version", "status", "stopping_reason", "reason_code",
            "reason_class", "legacy_status", "algorithm", "iterations",
            "max_iterations", "final_residual", "final_objective",
            "relative_iterate_change", "trajectory", "warnings",
            "failure_reason", "terminal_evidence", "endpoint_confirmation",
        }
        return cls(**{key: value[key] for key in allowed if key in value})

    @classmethod
    def invalid_parameters(
        cls,
        errors: Iterable[str],
        *,
        algorithm: str | None = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> "ConvergenceReport":
        messages = tuple(str(error) for error in errors)
        return cls(
            status=ConvergenceStatus.INVALID_PARAMETERS,
            stopping_reason="parameter_validation_failed",
            algorithm=algorithm,
            terminal_evidence={"validation_errors": list(messages), "parameters": dict(parameters or {})},
            warnings=messages,
            failure_reason="; ".join(messages),
        )

    @classmethod
    def unavailable(cls, reason: str, *, algorithm: str | None = None) -> "ConvergenceReport":
        return cls(
            status=ConvergenceStatus.UNAVAILABLE,
            stopping_reason="backend_unavailable",
            algorithm=algorithm,
            failure_reason=str(reason),
        )

    @classmethod
    def resource_exhausted(cls, reason: str, *, algorithm: str | None = None) -> "ConvergenceReport":
        return cls(
            status=ConvergenceStatus.RESOURCE_EXHAUSTED,
            stopping_reason="resource_budget_exhausted",
            algorithm=algorithm,
            failure_reason=str(reason),
        )


def sample_trajectory(
    trajectory: Iterable[Mapping[str, Any]],
    *,
    max_points: int = 100,
    patience: int = 0,
) -> tuple[dict[str, Any], ...]:
    """Keep a bounded sample while retaining the terminal patience window."""

    rows = [dict(row) for row in trajectory]
    limit = max(1, int(max_points))
    # A complete terminal window is stronger evidence than an arbitrary
    # uniform sample.  Grow the effective bound only when the requested bound
    # is too small to retain that configured window.
    limit = max(limit, max(1, int(patience)) if rows else 1)
    if len(rows) <= limit:
        return tuple(_json_safe(row) for row in rows)
    positions = {
        round(index * (len(rows) - 1) / (limit - 1)) if limit > 1 else 0
        for index in range(limit)
    }
    positions.update(range(max(0, len(rows) - max(1, int(patience))), len(rows)))
    return tuple(_json_safe(rows[index]) for index in sorted(positions))


def _metric(row: Mapping[str, Any], names: tuple[str, ...]) -> float | None:
    for name in names:
        value = row.get(name)
        if value is not None:
            try:
                result = float(value)
            except (TypeError, ValueError):
                return float("nan")
            return result
    return None


def _relative_measurement_residual(
    measurement: torch.Tensor,
    predicted_measurement: torch.Tensor,
    valid_measurement_mask: torch.Tensor | None = None,
) -> float:
    difference = predicted_measurement - measurement
    if valid_measurement_mask is not None:
        mask = valid_measurement_mask.to(device=measurement.device, dtype=torch.bool)
        if mask.ndim == measurement.ndim - 1:
            mask = mask.unsqueeze(0)
        if tuple(mask.shape) == tuple(measurement.shape):
            difference = difference.masked_fill(~mask, 0.0)
            measurement = measurement.masked_fill(~mask, 0.0)
    numerator = difference.reshape(difference.shape[0], -1).norm(dim=1)
    denominator = measurement.reshape(measurement.shape[0], -1).norm(dim=1).clamp_min(1e-12)
    return float((numerator / denominator).mean().item())


def classify_trajectory(
    trajectory: Iterable[Mapping[str, Any]],
    *,
    max_iterations: int | None = None,
    tolerance: float = 1e-5,
    patience: int = 5,
    stall_tolerance: float | None = None,
    algorithm: str | None = None,
) -> ConvergenceReport:
    """Classify a trajectory without treating a fixed iteration count as success."""

    rows = list(trajectory)
    sampled = sample_trajectory(rows, patience=patience)
    if not rows:
        empty_status = ConvergenceStatus.NUMERICAL_ERROR if is_core_algorithm(algorithm) else ConvergenceStatus.PARTIAL
        return ConvergenceReport(
            status=empty_status,
            stopping_reason="no_iteration_trajectory",
            max_iterations=max_iterations,
            warnings=("solver did not expose an iteration trajectory",),
            algorithm=algorithm,
            terminal_evidence=_report_evidence(rows, patience),
        )

    residuals = [_metric(row, ("normalized_residual", "data_residual", "residual", "normal_residual", "objective")) for row in rows]
    relative_changes = [_metric(row, ("relative_iterate_change", "iterate_change", "relative_change")) for row in rows]
    objective = [_metric(row, ("objective",)) for row in rows]
    if any(value is not None and not isfinite(value) for value in [*residuals, *relative_changes, *objective]):
        return ConvergenceReport(
            status=ConvergenceStatus.NUMERICAL_ERROR,
            stopping_reason="non_finite_trajectory_value",
            iterations=len(rows),
            max_iterations=max_iterations,
            trajectory=sampled,
            failure_reason="trajectory contains NaN or Inf",
            algorithm=algorithm,
            terminal_evidence=_report_evidence(rows, patience),
        )

    explicit = [
        normalize_status(
            row["status"],
            algorithm=algorithm,
            iterations=len(rows),
            max_iterations=max_iterations,
        ).value
        for row in rows if row.get("status")
    ]
    if any(status == ConvergenceStatus.NUMERICAL_ERROR.value for status in explicit):
        return ConvergenceReport(
            status=ConvergenceStatus.NUMERICAL_ERROR,
            stopping_reason="solver_reported_numerical_error",
            iterations=len(rows),
            max_iterations=max_iterations,
            final_residual=residuals[-1],
            final_objective=objective[-1],
            relative_iterate_change=relative_changes[-1],
            trajectory=sampled,
            failure_reason="solver reported a numerical error",
            algorithm=algorithm,
            terminal_evidence=_report_evidence(rows, patience),
        )
    if any(status == ConvergenceStatus.DIVERGED.value for status in explicit):
        return ConvergenceReport(
            status=ConvergenceStatus.DIVERGED,
            stopping_reason="solver_reported_divergence",
            iterations=len(rows),
            max_iterations=max_iterations,
            final_residual=residuals[-1],
            final_objective=objective[-1],
            relative_iterate_change=relative_changes[-1],
            trajectory=sampled,
            failure_reason="solver reported divergence",
            algorithm=algorithm,
            terminal_evidence=_report_evidence(rows, patience),
        )
    if any(status == ConvergenceStatus.RESOURCE_EXHAUSTED.value for status in explicit):
        return ConvergenceReport(
            status=ConvergenceStatus.RESOURCE_EXHAUSTED,
            stopping_reason="resource_budget_exhausted",
            iterations=len(rows), max_iterations=max_iterations,
            final_residual=residuals[-1], final_objective=objective[-1],
            relative_iterate_change=relative_changes[-1], trajectory=sampled,
            algorithm=algorithm, terminal_evidence=_report_evidence(rows, patience),
        )
    if any(status == ConvergenceStatus.CANCELLED.value for status in explicit):
        return ConvergenceReport(
            status=ConvergenceStatus.CANCELLED,
            stopping_reason="cancelled_by_callback",
            iterations=len(rows), max_iterations=max_iterations,
            final_residual=residuals[-1], final_objective=objective[-1],
            relative_iterate_change=relative_changes[-1], trajectory=sampled,
            algorithm=algorithm, terminal_evidence=_report_evidence(rows, patience),
        )
    if any(status == ConvergenceStatus.MAX_ITERATIONS.value for status in explicit):
        return ConvergenceReport(
            status=ConvergenceStatus.MAX_ITERATIONS,
            stopping_reason="maximum_iterations_reached",
            iterations=len(rows), max_iterations=max_iterations,
            final_residual=residuals[-1], final_objective=objective[-1],
            relative_iterate_change=relative_changes[-1], trajectory=sampled,
            algorithm=algorithm, terminal_evidence=_report_evidence(rows, patience),
        )

    bad_run = 0
    for previous, current in zip(residuals, residuals[1:]):
        if previous is not None and current is not None and current > previous * (1.0 + 1e-6):
            bad_run += 1
        else:
            bad_run = 0
        if bad_run >= max(2, int(patience)):
            return ConvergenceReport(
                status=ConvergenceStatus.DIVERGED,
                stopping_reason="residual_or_objective_increasing",
                iterations=len(rows),
                max_iterations=max_iterations,
                final_residual=residuals[-1],
                final_objective=objective[-1],
                relative_iterate_change=relative_changes[-1],
                trajectory=sampled,
                failure_reason="residual/objective increased for the patience window",
                algorithm=algorithm,
                terminal_evidence=_report_evidence(rows, patience),
            )

    # Reaching the configured boundary is a budget outcome.  It must not be
    # promoted to convergence merely because the last sampled row happens to
    # satisfy a residual/change threshold.
    if max_iterations is not None and len(rows) >= int(max_iterations):
        return ConvergenceReport(
            status=ConvergenceStatus.MAX_ITERATIONS,
            stopping_reason="maximum_iterations_reached",
            iterations=len(rows),
            max_iterations=max_iterations,
            final_residual=residuals[-1],
            final_objective=objective[-1],
            relative_iterate_change=relative_changes[-1],
            trajectory=sampled,
            algorithm=algorithm,
            terminal_evidence=_report_evidence(rows, patience),
        )

    final_residual = residuals[-1]
    final_change = relative_changes[-1]
    residual_ok = final_residual is not None and final_residual <= float(tolerance)
    # A tiny iterate change while the data residual is still large is
    # stagnation, not convergence.  Change-only convergence is accepted when
    # no residual was exposed (or when the residual is already near tolerance).
    change_ok = (
        final_change is not None
        and final_change <= float(tolerance)
        and (final_residual is None or final_residual <= max(float(tolerance) * 10.0, 1e-8))
    )
    if residual_ok or change_ok:
        return ConvergenceReport(
            status=ConvergenceStatus.CONVERGED,
            stopping_reason="residual_or_iterate_tolerance",
            iterations=len(rows),
            max_iterations=max_iterations,
            final_residual=final_residual,
            final_objective=objective[-1],
            relative_iterate_change=final_change,
            trajectory=sampled,
            algorithm=algorithm,
            terminal_evidence=_report_evidence(rows, patience),
        )

    stall_limit = float(stall_tolerance if stall_tolerance is not None else max(float(tolerance) * 0.1, 1e-12))
    recent_changes = [value for value in relative_changes[-max(1, int(patience)):] if value is not None]
    if len(recent_changes) >= max(1, int(patience)) and all(value <= stall_limit for value in recent_changes):
        return ConvergenceReport(
            status=ConvergenceStatus.STALLED,
            stopping_reason="iterate_change_below_progress_threshold",
            iterations=len(rows),
            max_iterations=max_iterations,
            final_residual=final_residual,
            final_objective=objective[-1],
            relative_iterate_change=final_change,
            trajectory=sampled,
            algorithm=algorithm,
            terminal_evidence=_report_evidence(rows, patience),
        )

    if max_iterations is not None and len(rows) >= int(max_iterations):
        return ConvergenceReport(
            status=ConvergenceStatus.MAX_ITERATIONS,
            stopping_reason="maximum_iterations_reached",
            iterations=len(rows),
            max_iterations=max_iterations,
            final_residual=final_residual,
            final_objective=objective[-1],
            relative_iterate_change=final_change,
            trajectory=sampled,
            algorithm=algorithm,
            terminal_evidence=_report_evidence(rows, patience),
        )

    incomplete_status = ConvergenceStatus.NUMERICAL_ERROR if is_core_algorithm(algorithm) else ConvergenceStatus.PARTIAL
    return ConvergenceReport(
        status=incomplete_status,
        stopping_reason="trajectory_ended_before_convergence",
        iterations=len(rows),
        max_iterations=max_iterations,
        final_residual=final_residual,
        final_objective=objective[-1],
        relative_iterate_change=final_change,
        trajectory=sampled,
        warnings=(f"algorithm={algorithm}",) if algorithm else (),
        algorithm=algorithm,
        terminal_evidence=_report_evidence(rows, patience),
    )


def post_run_validation(
    reconstruction: torch.Tensor,
    *,
    measurement: torch.Tensor | None = None,
    predicted_measurement: torch.Tensor | None = None,
    valid_measurement_mask: torch.Tensor | None = None,
    operator: Any | None = None,
    trajectory: Iterable[Mapping[str, Any]] | None = None,
    iterations: int = 0,
    max_iterations: int | None = None,
    tolerance: float = 1e-5,
    patience: int = 5,
    non_iterative: bool = False,
    algorithm: str | None = None,
) -> ConvergenceReport:
    """Validate final tensor/shape and classify available optimization evidence."""

    if not isinstance(reconstruction, torch.Tensor):
        return ConvergenceReport(
            status=ConvergenceStatus.NUMERICAL_ERROR,
            stopping_reason="solver_returned_non_tensor",
            failure_reason=f"got {type(reconstruction).__name__}",
            algorithm=algorithm,
        )
    if not torch.isfinite(reconstruction).all():
        return ConvergenceReport(
            status=ConvergenceStatus.NUMERICAL_ERROR,
            stopping_reason="non_finite_reconstruction",
            iterations=iterations,
            max_iterations=max_iterations,
            failure_reason="solver returned NaN or Inf",
            algorithm=algorithm,
        )
    if measurement is not None and not torch.isfinite(measurement).all():
        return ConvergenceReport(
            status=ConvergenceStatus.NUMERICAL_ERROR,
            stopping_reason="non_finite_measurement",
            iterations=iterations,
            max_iterations=max_iterations,
            failure_reason="measurement contains NaN or Inf",
            algorithm=algorithm,
        )
    if predicted_measurement is not None and not torch.isfinite(predicted_measurement).all():
        return ConvergenceReport(
            status=ConvergenceStatus.NUMERICAL_ERROR,
            stopping_reason="non_finite_prediction",
            iterations=iterations,
            max_iterations=max_iterations,
            failure_reason="predicted measurement contains NaN or Inf",
            algorithm=algorithm,
        )
    if (
        measurement is not None
        and predicted_measurement is not None
        and tuple(measurement.shape) != tuple(predicted_measurement.shape)
    ):
        return ConvergenceReport(
            status=ConvergenceStatus.NUMERICAL_ERROR,
            stopping_reason="measurement_shape_mismatch",
            iterations=iterations,
            max_iterations=max_iterations,
            failure_reason=(
                f"measurement shape {tuple(measurement.shape)} does not match "
                f"prediction shape {tuple(predicted_measurement.shape)}"
            ),
            algorithm=algorithm,
        )
    if operator is not None and tuple(reconstruction.shape[1:]) != tuple(operator.domain_shape):
        return ConvergenceReport(
            status=ConvergenceStatus.NUMERICAL_ERROR,
            stopping_reason="reconstruction_shape_mismatch",
            iterations=iterations,
            max_iterations=max_iterations,
            failure_reason=f"expected (B, {tuple(operator.domain_shape)}), got {tuple(reconstruction.shape)}",
            algorithm=algorithm,
        )

    if predicted_measurement is None and operator is not None:
        predicted_measurement = operator.forward(reconstruction)
    final_residual = None
    if measurement is not None and predicted_measurement is not None:
        final_residual = _relative_measurement_residual(
            measurement,
            predicted_measurement,
            valid_measurement_mask,
        )
    if non_iterative:
        return ConvergenceReport(
            status=ConvergenceStatus.COMPLETED_VALID,
            stopping_reason="direct_solver_completed",
            iterations=0,
            max_iterations=max_iterations,
            final_residual=final_residual,
            algorithm=algorithm,
            terminal_evidence={"terminal": {"iterations": 0}},
        )

    if trajectory is not None:
        report = classify_trajectory(
            trajectory,
            max_iterations=max_iterations,
            tolerance=tolerance,
            patience=patience,
            algorithm=algorithm,
        )
        if report.final_residual is None and final_residual is not None:
            return ConvergenceReport(
                status=report.status,
                stopping_reason=report.stopping_reason,
                iterations=report.iterations,
                max_iterations=report.max_iterations,
                final_residual=final_residual,
                final_objective=report.final_objective,
                relative_iterate_change=report.relative_iterate_change,
                trajectory=report.trajectory,
                warnings=report.warnings,
                failure_reason=report.failure_reason,
                schema_version=report.schema_version,
                algorithm=report.algorithm,
                reason_code=report.reason_code,
                reason_class=report.reason_class,
                legacy_status=report.legacy_status,
                terminal_evidence=report.terminal_evidence,
                endpoint_confirmation=report.endpoint_confirmation,
            )
        return report

    # A residual-only post-run observation is not enough to claim convergence:
    # no trajectory means no native criterion or patience window was observed.
    if max_iterations is not None and int(iterations) >= int(max_iterations):
        status = ConvergenceStatus.MAX_ITERATIONS
        reason = "maximum_iterations_reached_without_trajectory"
    elif is_core_algorithm(algorithm):
        status = ConvergenceStatus.NUMERICAL_ERROR
        reason = "no_iteration_trajectory"
    else:
        status = ConvergenceStatus.PARTIAL
        reason = "solver_did_not_expose_trajectory"
    return ConvergenceReport(
        status=status,
        stopping_reason=reason,
        iterations=int(iterations),
        max_iterations=max_iterations,
        final_residual=final_residual,
        warnings=("solver did not expose an iteration trajectory",),
        algorithm=algorithm,
        terminal_evidence={"terminal": {"iterations": int(iterations), "residual": final_residual}},
    )


# Explicit aliases make the small public interface easy to discover.
assess_trajectory = classify_trajectory
validate_post_run = post_run_validation


def _endpoint_operator_stats(operator: Any) -> dict[str, Any]:
    stats = getattr(operator, "stats", None)
    if not callable(stats):
        return {}
    return dict(stats() or {})


def _endpoint_call_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, int]:
    delta: dict[str, int] = {}
    for name in ("forward_calls", "adjoint_calls", "total_operator_calls"):
        if before.get(name) is not None and after.get(name) is not None:
            delta[name] = int(after[name]) - int(before[name])
    return delta


def _endpoint_finite_tensor(value: Any) -> bool:
    return isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all().item())


def _endpoint_relative_change(previous: torch.Tensor, current: torch.Tensor) -> float:
    numerator = float((current - previous).detach().reshape(-1).norm().item())
    denominator = max(float(previous.detach().reshape(-1).norm().item()), 1e-12)
    return numerator / denominator


def _endpoint_box_constraints(
    value: torch.Tensor,
    parameters: Mapping[str, Any],
) -> torch.Tensor:
    minimum = parameters.get("min_value")
    maximum = parameters.get("max_value")
    if minimum is None and maximum is None:
        return value
    return value.clamp(minimum, maximum)


def _balanced_endpoint_subsets(num_angles: int, count: int) -> list[tuple[int, ...]]:
    count = int(count)
    if count <= 0 or count > int(num_angles):
        raise ValueError("subset_count must be between 1 and the number of views")
    base, remainder = divmod(int(num_angles), count)
    subsets: list[tuple[int, ...]] = []
    start = 0
    for index in range(count):
        size = base + (1 if index < remainder else 0)
        subsets.append(tuple(range(start, start + size)))
        start += size
    return subsets


def _row_action_endpoint_native(
    algorithm: str,
    *,
    reconstruction: torch.Tensor,
    measurement: torch.Tensor,
    prediction: torch.Tensor,
    operator: Any,
    parameters: Mapping[str, Any],
) -> tuple[str, float, dict[str, Any]]:
    """Recompute a fixed-point relative change at the final reconstruction.

    The solver trajectory stores the change made by the last update.  The
    endpoint has no previous tensor, so the independent evaluator applies one
    solver-native update to the final reconstruction and measures its relative
    fixed-point change.  This is a state-based check and does not trust the
    scalar copied from the solver trajectory.
    """

    from .solvers._utils import (
        apply_box_constraints,
        make_angle_subsets,
        make_subset_operator,
        select_measurement_subset,
    )

    residual = prediction - measurement
    if not _endpoint_finite_tensor(residual):
        raise ValueError("endpoint residual is non-finite")
    if algorithm == "landweber":
        step = parameters.get("step_size", 1e-3)
        if isinstance(step, bool):
            raise ValueError("step_size is not a finite positive number")
        step = float(step)
        if not isfinite(step) or step <= 0.0:
            raise ValueError("step_size is not a finite positive number")
        candidate = reconstruction - step * operator.adjoint(residual)
        candidate = apply_box_constraints(
            candidate,
            min_value=parameters.get("min_value"),
            max_value=parameters.get("max_value"),
        )
        if not _endpoint_finite_tensor(candidate):
            raise ValueError("endpoint Landweber update is non-finite")
        return (
            "relative_iterate_change",
            _endpoint_relative_change(reconstruction, candidate),
            {"update": "landweber", "step_size": step},
        )

    if algorithm == "sirt":
        batch = int(measurement.shape[0])
        domain = (batch, *tuple(operator.domain_shape))
        range_ = (batch, *tuple(operator.range_shape))
        dtype = reconstruction.dtype
        device = reconstruction.device
        row_weight = operator.forward(torch.ones(domain, device=device, dtype=dtype))
        row_weight = torch.where(
            row_weight < 1e-8,
            torch.full_like(row_weight, float("inf")),
            row_weight,
        ).reciprocal()
        column_weight = operator.adjoint(torch.ones(range_, device=device, dtype=dtype))
        column_weight = torch.where(
            column_weight < 1e-8,
            torch.full_like(column_weight, float("inf")),
            column_weight,
        ).reciprocal()
        if not _endpoint_finite_tensor(row_weight) or not _endpoint_finite_tensor(column_weight):
            raise ValueError("endpoint SIRT normalization is non-finite")
        candidate = reconstruction - column_weight * operator.adjoint(row_weight * residual)
        candidate = apply_box_constraints(
            candidate,
            min_value=parameters.get("min_value"),
            max_value=parameters.get("max_value"),
        )
        if not _endpoint_finite_tensor(candidate):
            raise ValueError("endpoint SIRT update is non-finite")
        return (
            "relative_iterate_change",
            _endpoint_relative_change(reconstruction, candidate),
            {"update": "sirt"},
        )

    if algorithm not in {"sart", "os_sart"}:
        raise ValueError(f"unsupported row-action endpoint algorithm {algorithm!r}")

    num_angles = int(operator.range_shape[-2])
    explicit_subsets = parameters.get("subset_indices")
    if explicit_subsets is not None:
        subset_spec = [tuple(int(value) for value in indices) for indices in explicit_subsets]
    elif algorithm == "os_sart" and parameters.get("subset_count") is not None:
        subset_spec = _balanced_endpoint_subsets(num_angles, int(parameters["subset_count"]))
    else:
        block_size = parameters.get("block_size")
        if algorithm == "os_sart" and block_size is None:
            block_size = max(num_angles // 10, 1)
        subsets = make_angle_subsets(
            num_angles=num_angles,
            block_size=block_size,
            order_strategy=str(parameters.get("order_strategy", "ordered")),
            seed=parameters.get("seed"),
            device=measurement.device,
        )
        subset_spec = [tuple(int(value) for value in item.detach().cpu().tolist()) for item in subsets]

    subsets = make_angle_subsets(
        num_angles=num_angles,
        subset_indices=subset_spec,
        device=measurement.device,
    )
    batch = int(measurement.shape[0])
    domain = (batch, *tuple(operator.domain_shape))
    candidate = reconstruction.clone()
    eps = float(parameters.get("eps", 1e-8))
    relaxation = float(parameters.get("relaxation", 1.0))
    for indices in subsets:
        sub_operator = make_subset_operator(operator, indices)
        y_sub = select_measurement_subset(measurement, indices)
        ones_range = torch.ones(
            (batch, *tuple(sub_operator.range_shape)),
            device=measurement.device,
            dtype=reconstruction.dtype,
        )
        row_weight = sub_operator.forward(
            torch.ones(domain, device=measurement.device, dtype=reconstruction.dtype)
        )
        row_weight = torch.where(
            row_weight.abs() < eps,
            torch.full_like(row_weight, float("inf")),
            row_weight,
        ).reciprocal()
        column_weight = sub_operator.adjoint(ones_range)
        column_weight = torch.where(
            column_weight.abs() < eps,
            torch.full_like(column_weight, float("inf")),
            column_weight,
        ).reciprocal()
        subset_residual = sub_operator.forward(candidate) - y_sub
        candidate = candidate - relaxation * column_weight * sub_operator.adjoint(row_weight * subset_residual)
        candidate = apply_box_constraints(
            candidate,
            min_value=parameters.get("min_value"),
            max_value=parameters.get("max_value"),
        )
        if not (
            _endpoint_finite_tensor(row_weight)
            and _endpoint_finite_tensor(column_weight)
            and _endpoint_finite_tensor(subset_residual)
            and _endpoint_finite_tensor(candidate)
        ):
            raise ValueError("endpoint subset update is non-finite")
    return (
        "relative_epoch_change",
        _endpoint_relative_change(reconstruction, candidate),
        {"update": algorithm, "complete_epoch": True, "subset_count": len(subsets)},
    )


def confirm_endpoint(
    *,
    algorithm: str,
    reconstruction: torch.Tensor,
    measurement: torch.Tensor,
    operator: Any,
    predicted_measurement: torch.Tensor | None = None,
    valid_measurement_mask: torch.Tensor | None = None,
    policy: Mapping[str, Any] | None,
    trajectory: Iterable[Mapping[str, Any]],
    solver_status: str,
    solver_stopping_reason: str,
    iterations: int,
    max_iterations: int | None,
    parameters: Mapping[str, Any],
    operator_norm_estimate: float,
) -> dict[str, Any]:
    """Independently recompute final evidence and audit the patience window."""

    rows = [dict(row) for row in trajectory]
    endpoint_counter_before = _endpoint_operator_stats(operator)
    solver_status_value = normalize_status(
        solver_status,
        algorithm=algorithm,
        iterations=iterations,
        max_iterations=max_iterations,
        direct=algorithm in {"fbp", "fdk"},
    ).value
    prediction = operator.forward(reconstruction) if predicted_measurement is None else predicted_measurement
    finite_inputs = (
        isinstance(reconstruction, torch.Tensor)
        and isinstance(measurement, torch.Tensor)
        and isinstance(prediction, torch.Tensor)
        and bool(torch.isfinite(reconstruction).all())
        and bool(torch.isfinite(measurement).all())
        and bool(torch.isfinite(prediction).all())
    )
    residual = prediction - measurement if finite_inputs else None
    native_residual = residual
    native_measurement = measurement
    if residual is not None and valid_measurement_mask is not None:
        mask = valid_measurement_mask.to(device=measurement.device, dtype=torch.bool)
        if mask.ndim == measurement.ndim - 1:
            mask = mask.unsqueeze(0)
        if tuple(mask.shape) == tuple(measurement.shape):
            native_residual = residual.masked_fill(~mask, 0.0)
            native_measurement = measurement.masked_fill(~mask, 0.0)
    endpoint_data_residual = (
        _relative_measurement_residual(measurement, prediction, valid_measurement_mask)
        if finite_inputs else None
    )
    finite = finite_inputs
    reconstruction_min = float(reconstruction.min().item()) if finite else None
    reconstruction_max = float(reconstruction.max().item()) if finite else None
    result: dict[str, Any] = {
        "schema_version": "ct.endpoint_confirmation.v1",
        "status": "not_applicable" if algorithm in {"fbp", "fdk"} else "not_requested",
        "passed": algorithm in {"fbp", "fdk"},
        "finite": finite,
        "reconstruction_min": reconstruction_min,
        "reconstruction_max": reconstruction_max,
        "normalized_data_residual": endpoint_data_residual,
        "solver_status": solver_status_value,
        "solver_stopping_reason": solver_stopping_reason,
        "solver_reason_class": status_reason_class(solver_status_value),
        "reasons": [],
        "operator_calls": {},
    }
    if algorithm in {"fbp", "fdk"}:
        result["status"] = "passed" if finite else "failed"
        result["passed"] = finite
        result["operator_calls"] = _endpoint_call_delta(
            endpoint_counter_before, _endpoint_operator_stats(operator)
        )
        return result
    if policy is None:
        result["operator_calls"] = _endpoint_call_delta(
            endpoint_counter_before, _endpoint_operator_stats(operator)
        )
        return result
    raw_discrepancy_target = (policy.get("effective", {}) or {}).get(
        "discrepancy_target", policy.get("discrepancy_target", float("inf"))
    )
    discrepancy_available = raw_discrepancy_target is not None
    discrepancy_target = (
        float(raw_discrepancy_target) if discrepancy_available else None
    )
    normal_tol = float(policy.get("normalized_normal_residual_tolerance", 1e-4))
    iterate_tol = float(policy.get("relative_iterate_tolerance", 1e-4))
    prox_tol = float(policy.get("prox_gradient_mapping_tolerance", 1e-4))
    objective_tol = float(policy.get("relative_objective_tolerance", 1e-5))
    endpoint_cfg = dict(policy.get("endpoint_confirmation", {}) or {})
    absolute_tolerance = float(endpoint_cfg.get("absolute_tolerance", 1e-7))
    relative_tolerance = float(endpoint_cfg.get("relative_tolerance", 1e-4))
    last_row = rows[-1] if rows else {}
    native_name = "relative_iterate_change"
    native_trajectory_value = _metric(
        last_row,
        ("native_criterion_value", "relative_iterate_change", "relative_epoch_change"),
    )
    native_value = native_trajectory_value
    native_ok = native_value is not None and isfinite(float(native_value)) and float(native_value) <= iterate_tol
    native_details: dict[str, Any] = {}
    native_recomputation_error: str | None = None
    native_endpoint_consistent: bool | None = None
    if algorithm in {"sirt", "landweber", "sart", "os_sart"} and finite_inputs:
        try:
            native_name, native_value, native_details = _row_action_endpoint_native(
                algorithm,
                reconstruction=reconstruction,
                measurement=measurement,
                prediction=prediction,
                operator=operator,
                parameters=parameters,
            )
            native_ok = isfinite(float(native_value)) and float(native_value) <= iterate_tol
        except OperatorBudgetExceeded:
            raise
        except Exception as error:
            native_recomputation_error = str(error)
            native_ok = False
    if (
        algorithm in {"sirt", "landweber", "sart", "os_sart"}
        and native_trajectory_value is not None
        and native_value is not None
        and isfinite(float(native_trajectory_value))
        and isfinite(float(native_value))
    ):
        native_endpoint_consistent = abs(float(native_trajectory_value) - float(native_value)) <= (
            absolute_tolerance
            + relative_tolerance * max(abs(float(native_trajectory_value)), abs(float(native_value)))
        )
    if algorithm in {"cgls", "lsqr"}:
        gradient = operator.adjoint(native_residual) if native_residual is not None else torch.full_like(reconstruction, float("nan"))
        native_name = "normalized_normal_residual"
        native_value = float(gradient.norm().item()) / max(float(operator_norm_estimate) * float(native_residual.norm().item()), 1e-12)
        change = rows[-1].get("relative_iterate_change") if rows else None
        native_ok = native_value <= normal_tol or (change is not None and float(change) <= iterate_tol)
    elif algorithm == "tikhonov":
        strength = float(parameters.get("reg_strength", 0.0))
        gradient = (
            operator.adjoint(native_residual) + strength * reconstruction
            if native_residual is not None else torch.full_like(reconstruction, float("nan"))
        )
        rhs = operator.adjoint(native_measurement)
        native_name = "normalized_regularized_normal_residual"
        native_value = float(gradient.norm().item()) / max(float(rhs.norm().item()) + strength * float(reconstruction.norm().item()), 1e-12)
        change = rows[-1].get("relative_iterate_change") if rows else None
        native_ok = native_value <= normal_tol and change is not None and float(change) <= iterate_tol
    elif algorithm == "tv_fista":
        from .regularizers import TVRegularizer
        tv = TVRegularizer(
            mode=str(parameters.get("tv_mode", "isotropic")),
            num_iterations=int(parameters.get("tv_num_iterations", 50)),
            tolerance=float(parameters.get("tv_tolerance", 1e-5)),
        )
        step = float((rows[-1].get("step_size") if rows else None) or parameters.get("step_size") or 1.0)
        gradient = operator.adjoint(native_residual) if native_residual is not None else torch.full_like(reconstruction, float("nan"))
        prox = tv.proximal(reconstruction - step * gradient, step * float(parameters.get("reg_strength", 0.0)))
        native_name = "normalized_prox_gradient_mapping"
        native_value = float(((reconstruction - prox) / step).norm().item()) / max(float(reconstruction.norm().item()) / step, 1e-12)
        objective_change = ((rows[-1].get("metadata") or {}).get("relative_composite_objective_change") if rows else None)
        native_ok = native_value <= prox_tol and objective_change is not None and float(objective_change) <= objective_tol
        result["relative_composite_objective_change"] = objective_change
        result["composite_objective"] = (
            float(0.5 * residual.square().sum().item() + float(parameters.get("reg_strength", 0.0)) * tv.value(reconstruction).sum().item())
            if residual is not None else None
        )
    if native_value is not None and not isfinite(float(native_value)):
        finite = False
        result["reasons"].append("non_finite_native_criterion")
    if native_recomputation_error is not None:
        result["native_recomputation_error"] = native_recomputation_error
    if native_details:
        result["native_recomputation"] = native_details
    discrepancy_ok = (
        not discrepancy_available
        or bool(
            endpoint_data_residual is not None
            and endpoint_data_residual <= float(discrepancy_target)
        )
    )
    result.update({
        "discrepancy_target": discrepancy_target,
        "discrepancy_available": discrepancy_available,
        "discrepancy_satisfied": discrepancy_ok,
        "native_criterion_name": native_name,
        "native_criterion_value": native_value,
        "native_criterion_satisfied": native_ok,
        "trajectory_native_criterion_value": native_trajectory_value,
        "trajectory_endpoint_native_consistent": native_endpoint_consistent,
    })
    patience = int(policy.get("patience", 5))
    checked = [
        row for row in rows
        if row.get("criteria") and bool((row.get("metadata") or {}).get("checked", True))
    ]
    tail = checked[-patience:]
    check_every = max(1, int(policy.get("check_every", 1)))
    check_spacing_ok = all(
        int(current.get("iteration", -1)) - int(previous.get("iteration", -1)) == check_every
        for previous, current in zip(tail, tail[1:])
    )
    min_iterations = int(policy.get("min_iterations", 1))
    min_iteration_ok = bool(tail) and int(tail[0].get("iteration", -1)) >= min_iterations
    monotonic = all(int(a.get("iteration", -1)) < int(b.get("iteration", -1)) for a, b in zip(rows, rows[1:]))
    consecutive = (
        len(tail) == patience
        and check_spacing_ok
        and min_iteration_ok
        and all(all(bool(v) for v in (row.get("criteria") or {}).values()) for row in tail)
    )
    counter_ok = bool(tail)
    expected_consecutive = 0
    for row in checked:
        expected_consecutive = expected_consecutive + 1 if all(
            bool(value) for value in (row.get("criteria") or {}).values()
        ) else 0
        reported = row.get("consecutive_criteria_count")
        if reported is not None and int(reported) != expected_consecutive:
            counter_ok = False
    counter_ok = counter_ok and len(tail) == patience and int(tail[-1].get("consecutive_criteria_count", 0)) >= patience
    epoch_boundary = True
    if algorithm in {"sart", "os_sart"}:
        epoch_boundary = bool(rows) and all(
            row.get("subset") is None
            and (
                row.get("epoch") is not None
                or bool((row.get("metadata") or {}).get("complete_sweep", False))
            )
            for row in rows
        )
    result.update({
        "trajectory_monotonic": monotonic,
        "check_spacing_consistent": check_spacing_ok,
        "min_iteration_respected": min_iteration_ok,
        "patience_window_passed": consecutive,
        "consecutive_counter_consistent": counter_ok,
        "complete_epoch_boundaries": epoch_boundary,
    })
    last_data = _metric(
        last_row,
        ("normalized_data_residual", "normalized_residual", "data_residual", "residual"),
    )
    trajectory_endpoint_consistent = bool(
        last_data is not None
        and endpoint_data_residual is not None
        and abs(float(last_data) - endpoint_data_residual)
        <= absolute_tolerance
        + relative_tolerance * max(abs(float(last_data)), abs(endpoint_data_residual))
    )
    result["trajectory_endpoint_consistent"] = trajectory_endpoint_consistent
    if not finite:
        result["reasons"].append("non_finite_endpoint")
    if solver_status_value == "converged":
        if not discrepancy_ok: result["reasons"].append("endpoint_discrepancy_unmet")
        if not native_ok: result["reasons"].append("endpoint_native_criterion_unmet")
        if not monotonic: result["reasons"].append("trajectory_not_monotonic")
        if not consecutive: result["reasons"].append("patience_window_incomplete")
        if not counter_ok: result["reasons"].append("consecutive_counter_mismatch")
        if not epoch_boundary: result["reasons"].append("subset_level_termination")
        if native_recomputation_error is not None:
            result["reasons"].append("endpoint_native_recomputation_failed")
        if native_endpoint_consistent is False:
            result["reasons"].append("trajectory_endpoint_native_metric_mismatch")
        if endpoint_cfg.get("require_trajectory_consistency", True) and not trajectory_endpoint_consistent:
            result["reasons"].append("trajectory_endpoint_metric_mismatch")
    allowed = {
        "sirt": "discrepancy_and_relative_iterate_change_patience",
        "landweber": "discrepancy_and_relative_iterate_change_patience",
        "sart": "discrepancy_and_relative_epoch_change_patience",
        "os_sart": "discrepancy_and_relative_epoch_change_patience",
        "cgls": "discrepancy_and_krylov_native_patience",
        "lsqr": "discrepancy_and_krylov_native_patience",
        "tikhonov": "discrepancy_regularized_normal_and_iterate_patience",
        "tv_fista": "discrepancy_prox_gradient_and_objective_patience",
    }
    if solver_status_value == "converged" and solver_stopping_reason != allowed.get(algorithm):
        result["reasons"].append("stopping_reason_not_allowed")
    result["converged_at_budget_boundary"] = bool(solver_status_value == "converged" and max_iterations is not None and iterations == max_iterations)
    result["operator_calls"] = _endpoint_call_delta(
        endpoint_counter_before, _endpoint_operator_stats(operator)
    )
    result["passed"] = finite and (solver_status_value != "converged" or not result["reasons"])
    result["status"] = "passed" if result["passed"] else "failed"
    return result

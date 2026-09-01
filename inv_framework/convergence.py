"""Convergence reports and post-run validation for CT reconstructions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any, Iterable, Mapping

import torch


class ConvergenceStatus(str, Enum):
    CONVERGED = "converged"
    PARTIAL = "partial"
    STALLED = "stalled"
    DIVERGED = "diverged"
    MAX_ITERATIONS = "max_iterations"
    NON_ITERATIVE_COMPLETED = "non_iterative_completed"
    INVALID_PARAMETERS = "invalid_parameters"
    NUMERICAL_FAILURE = "numerical_failure"


@dataclass(frozen=True)
class ConvergenceReport:
    """Portable convergence state independent of a solver implementation."""

    status: ConvergenceStatus | str
    stopping_reason: str
    iterations: int = 0
    max_iterations: int | None = None
    final_residual: float | None = None
    final_objective: float | None = None
    relative_iterate_change: float | None = None
    trajectory: tuple[dict[str, Any], ...] = ()
    warnings: tuple[str, ...] = ()
    failure_reason: str | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.status, str):
            try:
                object.__setattr__(self, "status", ConvergenceStatus(self.status))
            except ValueError:
                raise ValueError(f"unknown convergence status {self.status!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "stopping_reason": self.stopping_reason,
            "iterations": int(self.iterations),
            "max_iterations": self.max_iterations,
            "final_residual": self.final_residual,
            "final_objective": self.final_objective,
            "relative_iterate_change": self.relative_iterate_change,
            "trajectory": [dict(item) for item in self.trajectory],
            "warnings": list(self.warnings),
            "failure_reason": self.failure_reason,
        }

    @classmethod
    def invalid_parameters(cls, errors: Iterable[str]) -> "ConvergenceReport":
        messages = tuple(str(error) for error in errors)
        return cls(
            status=ConvergenceStatus.INVALID_PARAMETERS,
            stopping_reason="parameter_validation_failed",
            warnings=messages,
            failure_reason="; ".join(messages),
        )


def sample_trajectory(
    trajectory: Iterable[Mapping[str, Any]],
    *,
    max_points: int = 100,
) -> tuple[dict[str, Any], ...]:
    """Keep a deterministic, evenly spaced subset of a long trajectory."""

    rows = [dict(row) for row in trajectory]
    limit = max(1, int(max_points))
    if len(rows) <= limit:
        return tuple(rows)
    positions = {
        round(index * (len(rows) - 1) / (limit - 1)) if limit > 1 else 0
        for index in range(limit)
    }
    return tuple(rows[index] for index in sorted(positions))


def _metric(row: Mapping[str, Any], names: tuple[str, ...]) -> float | None:
    for name in names:
        value = row.get(name)
        if value is not None:
            try:
                result = float(value)
            except (TypeError, ValueError):
                return None
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
    sampled = sample_trajectory(rows)
    if not rows:
        return ConvergenceReport(
            status=ConvergenceStatus.PARTIAL,
            stopping_reason="no_iteration_trajectory",
            max_iterations=max_iterations,
            warnings=("solver did not expose an iteration trajectory",),
        )

    residuals = [_metric(row, ("normalized_residual", "data_residual", "residual", "normal_residual", "objective")) for row in rows]
    relative_changes = [_metric(row, ("relative_iterate_change", "iterate_change", "relative_change")) for row in rows]
    objective = [_metric(row, ("objective",)) for row in rows]
    if any(value is not None and not isfinite(value) for value in [*residuals, *relative_changes, *objective]):
        return ConvergenceReport(
            status=ConvergenceStatus.NUMERICAL_FAILURE,
            stopping_reason="non_finite_trajectory_value",
            iterations=len(rows),
            max_iterations=max_iterations,
            trajectory=sampled,
            failure_reason="trajectory contains NaN or Inf",
        )

    explicit = [str(row.get("status")) for row in rows if row.get("status")]
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
        )

    return ConvergenceReport(
        status=ConvergenceStatus.PARTIAL,
        stopping_reason="trajectory_ended_before_convergence",
        iterations=len(rows),
        max_iterations=max_iterations,
        final_residual=final_residual,
        final_objective=objective[-1],
        relative_iterate_change=final_change,
        trajectory=sampled,
        warnings=(f"algorithm={algorithm}",) if algorithm else (),
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
            status=ConvergenceStatus.NUMERICAL_FAILURE,
            stopping_reason="solver_returned_non_tensor",
            failure_reason=f"got {type(reconstruction).__name__}",
        )
    if not torch.isfinite(reconstruction).all():
        return ConvergenceReport(
            status=ConvergenceStatus.NUMERICAL_FAILURE,
            stopping_reason="non_finite_reconstruction",
            iterations=iterations,
            max_iterations=max_iterations,
            failure_reason="solver returned NaN or Inf",
        )
    if operator is not None and tuple(reconstruction.shape[1:]) != tuple(operator.domain_shape):
        return ConvergenceReport(
            status=ConvergenceStatus.NUMERICAL_FAILURE,
            stopping_reason="reconstruction_shape_mismatch",
            iterations=iterations,
            max_iterations=max_iterations,
            failure_reason=f"expected (B, {tuple(operator.domain_shape)}), got {tuple(reconstruction.shape)}",
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
            status=ConvergenceStatus.NON_ITERATIVE_COMPLETED,
            stopping_reason="direct_solver_completed",
            iterations=0,
            max_iterations=max_iterations,
            final_residual=final_residual,
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
            )
        return report

    if final_residual is not None and final_residual <= float(tolerance):
        status = ConvergenceStatus.CONVERGED
        reason = "post_run_residual_tolerance"
    elif max_iterations is not None and int(iterations) >= int(max_iterations):
        status = ConvergenceStatus.MAX_ITERATIONS
        reason = "maximum_iterations_reached_without_trajectory"
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
    )


# Explicit aliases make the small public interface easy to discover.
assess_trajectory = classify_trajectory
validate_post_run = post_run_validation

"""Common solver contracts and optional detailed execution diagnostics.

The original public solver API intentionally stays small:
``solve(measurement, operator) -> Tensor``.  Research orchestration needs a
little more evidence than that return value can provide, however.  The
dataclasses in this module are the backwards-compatible seam for that
evidence.  A solver may implement :meth:`solve_detailed`; callers that only
know the old API continue to use :meth:`solve` unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
import math
import time
from typing import Any, Callable, Iterable, Mapping

import torch

from ..convergence import (
    ConvergenceStatus,
    _json_safe,
    canonicalize_status,
    sample_trajectory,
    status_reason_class,
)
from ..operators.base import ForwardOperator


@dataclass(frozen=True)
class SolveControl:
    """Controls optional detailed solver execution.

    ``max_iterations`` is an upper bound, never evidence of convergence.
    ``tolerance`` is interpreted by the individual solver's native criterion;
    algorithm-specific tolerances remain available through the solver's
    constructor.  The callback is invoked for each recorded checkpoint and
    may return ``False`` to request cancellation.
    """

    max_iterations: int | None = None
    tolerance: float | None = None
    min_iterations: int = 1
    check_every: int = 1
    record_every: int = 1
    max_trajectory_points: int = 200
    patience: int = 5
    stop_on_convergence: bool = True
    check_finite: bool = True
    relative_iterate_tolerance: float | None = None
    normalized_normal_residual_tolerance: float | None = None
    relative_objective_tolerance: float | None = None
    prox_gradient_mapping_tolerance: float | None = None
    discrepancy_target: float | None = None
    stall_relative_iterate_tolerance: float = 1e-8
    stall_patience: int = 5
    stall_enabled: bool = False
    divergence_relative_increase_tolerance: float = 1e-4
    divergence_patience: int = 5
    divergence_enabled: bool = False
    callback: Callable[["IterationRecord"], bool | None] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_iterations is not None and (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, int)
            or self.max_iterations <= 0
        ):
            raise ValueError("max_iterations must be a positive integer or None")
        if self.tolerance is not None and (
            isinstance(self.tolerance, bool)
            or not math.isfinite(float(self.tolerance))
            or float(self.tolerance) < 0.0
        ):
            raise ValueError("tolerance must be a finite nonnegative number or None")
        for name in ("min_iterations", "check_every", "record_every", "max_trajectory_points", "patience", "stall_patience", "divergence_patience"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "relative_iterate_tolerance", "normalized_normal_residual_tolerance",
            "relative_objective_tolerance", "prox_gradient_mapping_tolerance",
            "discrepancy_target", "stall_relative_iterate_tolerance",
            "divergence_relative_increase_tolerance",
        ):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(float(value)) or float(value) < 0.0):
                raise ValueError(f"{name} must be a finite nonnegative number or None")


@dataclass(frozen=True)
class IterationRecord:
    """One solver-native diagnostic checkpoint.

    The fields are deliberately scalar/JSON-friendly except for ``metadata``;
    no tensors are retained in a trajectory.  ``data_residual`` is the
    normalized 2-norm whenever a residual tensor is available.
    """

    iteration: int
    epoch: int | None = None
    data_residual: float | None = None
    normalized_data_residual: float | None = None
    objective: float | None = None
    relative_iterate_change: float | None = None
    algorithm_residual: float | None = None
    forward_calls: int | None = None
    adjoint_calls: int | None = None
    elapsed_seconds: float = 0.0
    step_size: float | None = None
    relaxation: float | None = None
    finite: bool = True
    stopping_candidate: bool = False
    consecutive_criteria_count: int = 0
    criteria: Mapping[str, bool] = field(default_factory=dict)
    native_criterion_name: str | None = None
    native_criterion_value: float | None = None
    native_criterion_threshold: float | None = None
    status: str | None = None
    subset: int | None = None
    subset_count: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def normal_residual(self) -> float | None:
        """Compatibility alias used by CGLS/Tikhonov diagnostics."""

        return self.algorithm_residual

    def to_dict(self) -> dict[str, Any]:
        return _json_safe({
            "iteration": int(self.iteration),
            "epoch": self.epoch,
            "data_residual": self.data_residual,
            "normalized_data_residual": self.normalized_data_residual,
            "objective": self.objective,
            "relative_iterate_change": self.relative_iterate_change,
            "algorithm_residual": self.algorithm_residual,
            "normal_residual": self.algorithm_residual,
            "forward_calls": self.forward_calls,
            "adjoint_calls": self.adjoint_calls,
            "elapsed_seconds": float(self.elapsed_seconds),
            "step_size": self.step_size,
            "relaxation": self.relaxation,
            "finite": bool(self.finite),
            "stopping_candidate": bool(self.stopping_candidate),
            "consecutive_criteria_count": int(self.consecutive_criteria_count),
            "criteria": dict(self.criteria),
            "native_criterion_name": self.native_criterion_name,
            "native_criterion_value": self.native_criterion_value,
            "native_criterion_threshold": self.native_criterion_threshold,
            "status": self.status,
            "subset": self.subset,
            "subset_count": self.subset_count,
            "metadata": dict(self.metadata),
        })


@dataclass(frozen=True)
class SolveResult:
    """Detailed, backwards-compatible result for one solver invocation."""

    reconstruction: torch.Tensor
    actual_iterations: int
    status: str
    stopping_reason: str
    trajectory: tuple[IterationRecord, ...] = ()
    resources: Mapping[str, Any] = field(default_factory=dict)
    final_residual: float | None = None
    final_objective: float | None = None
    relative_iterate_change: float | None = None
    predicted_measurement: torch.Tensor | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        metadata = dict(self.metadata)
        algorithm = metadata.get("algorithm")
        raw = self.status.value if isinstance(self.status, ConvergenceStatus) else str(self.status)
        normalized = canonicalize_status(
            raw,
            algorithm=str(algorithm) if algorithm is not None else None,
            iterations=self.actual_iterations,
            max_iterations=self.metadata.get("max_iterations"),
            direct=bool(metadata.get("direct", False) or algorithm in {"fbp", "fdk"}),
        )
        stopping_reason = str(self.stopping_reason)
        if normalized.value == "max_iterations" and raw != normalized.value:
            stopping_reason = "maximum_iterations_reached"
        if raw != normalized.value and raw in {"numerical_failure", "non_iterative_completed", "not_applicable", "partial"}:
            metadata.setdefault("legacy_status", raw)
        metadata["reason_code"] = stopping_reason
        metadata["reason_class"] = status_reason_class(normalized)
        object.__setattr__(self, "status", normalized.value)
        object.__setattr__(self, "stopping_reason", stopping_reason)
        object.__setattr__(self, "metadata", metadata)

    @property
    def iterations(self) -> int:
        """Short alias used by result consumers."""

        return int(self.actual_iterations)

    @property
    def trajectory_available(self) -> bool:
        return bool(self.trajectory)

    def to_dict(self) -> dict[str, Any]:
        """Serialize diagnostics without serializing tensors."""

        return _json_safe({
            "schema_version": "ct.solve_result.v1",
            "actual_iterations": int(self.actual_iterations),
            "iterations": int(self.actual_iterations),
            "status": self.status,
            "convergence_status": self.status,
            "stopping_reason": self.stopping_reason,
            "reason_code": self.metadata.get("reason_code", self.stopping_reason),
            "reason_class": self.metadata.get("reason_class"),
            "legacy_status": self.metadata.get("legacy_status"),
            "trajectory_available": self.trajectory_available,
            "trajectory": [record.to_dict() for record in self.trajectory],
            "terminal_evidence": {
                "terminal": self.metadata.get("terminal_record"),
                "patience_window": list(self.metadata.get("patience_window", ())),
            },
            "resources": dict(self.resources),
            "final_residual": self.final_residual,
            "final_objective": self.final_objective,
            "relative_iterate_change": self.relative_iterate_change,
            "metadata": dict(self.metadata),
        })


def _tensor_norm(value: torch.Tensor | None) -> float | None:
    if value is None:
        return None
    return float(value.detach().reshape(-1).norm().item())


def _sample_records(
    records: Iterable[IterationRecord],
    limit: int,
    *,
    patience: int = 0,
) -> tuple[IterationRecord, ...]:
    rows = list(records)
    limit = max(1, int(limit), max(1, int(patience)) if rows else 1)
    if len(rows) <= limit:
        return tuple(rows)
    if limit <= 1:
        return (rows[-1],)
    positions = {
        round(index * (len(rows) - 1) / (limit - 1))
        for index in range(limit)
    }
    positions.update(range(max(0, len(rows) - max(1, int(patience))), len(rows)))
    return tuple(rows[index] for index in sorted(positions))


class IterationRecorder:
    """Create bounded scalar records while sharing operator counters."""

    def __init__(
        self,
        control: SolveControl,
        measurement: torch.Tensor,
        operator: Any,
        *,
        algorithm: str,
        callback: Callable[[IterationRecord], bool | None] | None = None,
    ) -> None:
        self.control = control
        self.measurement = measurement
        self.operator = operator
        self.algorithm = algorithm
        self.callback = callback if callback is not None else control.callback
        self.started = time.perf_counter()
        self.records: list[IterationRecord] = []
        self.previous: torch.Tensor | None = None
        # Keep the old attribute as a compatibility alias.  New result
        # payloads use the canonical ``numerical_error`` status.
        self.numerical_failure = False
        self.numerical_error_reason: str | None = None
        measurement_norm = measurement.detach().reshape(-1).norm().item()
        self.measurement_norm = max(float(measurement_norm), 1e-12)

    def set_initial(self, value: torch.Tensor) -> None:
        self.previous = value.detach().clone()

    def mark_numerical_error(self, reason: str) -> None:
        """Record the first field-specific numerical failure reason.

        ``record`` keeps its historical generic finite check for legacy
        callers.  Native detailed loops can opt into a more useful reason
        without changing the public result shape or the old solver paths.
        """

        self.numerical_failure = True
        if self.numerical_error_reason is None:
            self.numerical_error_reason = str(reason)

    def _counter_values(self) -> tuple[int | None, int | None]:
        stats = getattr(self.operator, "stats", None)
        if not callable(stats):
            return None, None
        payload = stats() or {}
        return payload.get("forward_calls"), payload.get("adjoint_calls")

    def record(
        self,
        iteration: int,
        value: torch.Tensor,
        *,
        residual: torch.Tensor | None = None,
        objective: float | None = None,
        algorithm_residual: float | None = None,
        step_size: float | None = None,
        relaxation: float | None = None,
        stopping_candidate: bool = False,
        consecutive_criteria_count: int = 0,
        criteria: Mapping[str, bool] | None = None,
        native_criterion_name: str | None = None,
        native_criterion_value: float | None = None,
        native_criterion_threshold: float | None = None,
        status: str | None = None,
        epoch: int | None = None,
        subset: int | None = None,
        subset_count: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> bool:
        # Finite checks are performed even for down-sampled trajectories.  A
        # skipped checkpoint must not hide a NaN/Inf that later contaminates
        # the reconstruction.
        finite = bool(torch.isfinite(value).all())
        if residual is not None:
            finite = finite and bool(torch.isfinite(residual).all())
        self.numerical_failure = self.numerical_failure or (self.control.check_finite and not finite)
        if int(iteration) % self.control.record_every != 0 and not stopping_candidate:
            self.previous = value.detach().clone()
            return True
        residual_norm = _tensor_norm(residual)
        change = None
        if self.previous is not None:
            # A unit floor avoids an artificial 1e12 relative change on the
            # first update from a zero initialization while preserving the
            # usual relative norm for non-small iterates.
            denominator = max(_tensor_norm(self.previous) or 0.0, 1e-12)
            change = (_tensor_norm(value - self.previous) or 0.0) / denominator
        normalized = None if residual_norm is None else residual_norm / self.measurement_norm
        forward_calls, adjoint_calls = self._counter_values()
        row = IterationRecord(
            iteration=int(iteration),
            epoch=epoch,
            data_residual=residual_norm,
            normalized_data_residual=normalized,
            objective=None if objective is None else float(objective),
            relative_iterate_change=change,
            algorithm_residual=None if algorithm_residual is None else float(algorithm_residual),
            forward_calls=forward_calls,
            adjoint_calls=adjoint_calls,
            elapsed_seconds=time.perf_counter() - self.started,
            step_size=step_size,
            relaxation=relaxation,
            finite=finite,
            stopping_candidate=bool(stopping_candidate),
            consecutive_criteria_count=int(consecutive_criteria_count),
            criteria=dict(criteria or {}),
            native_criterion_name=native_criterion_name,
            native_criterion_value=native_criterion_value,
            native_criterion_threshold=native_criterion_threshold,
            status=status,
            subset=subset,
            subset_count=subset_count,
            metadata={"algorithm": self.algorithm, **dict(metadata or {})},
        )
        self.records.append(row)
        self.previous = value.detach().clone()
        if self.callback is not None:
            decision = self.callback(row)
            if decision is False:
                return False
        return True

    def finish(
        self,
        reconstruction: torch.Tensor,
        *,
        actual_iterations: int,
        status: str,
        stopping_reason: str,
        residual: torch.Tensor | None = None,
        final_residual: float | None = None,
        final_objective: float | None = None,
        predicted_measurement: torch.Tensor | None = None,
        resources: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SolveResult:
        if final_residual is None and residual is not None:
            norm = _tensor_norm(residual)
            final_residual = None if norm is None else norm / self.measurement_norm
        last_change = self.records[-1].relative_iterate_change if self.records else None
        stats = {}
        stats_method = getattr(self.operator, "stats", None)
        if callable(stats_method):
            stats.update(stats_method() or {})
        stats["runtime_seconds"] = time.perf_counter() - self.started
        stats.update(dict(resources or {}))
        if self.control.check_finite:
            finite_result = isinstance(reconstruction, torch.Tensor) and bool(torch.isfinite(reconstruction).all())
            if predicted_measurement is not None:
                finite_result = finite_result and bool(torch.isfinite(predicted_measurement).all())
            for scalar in (final_residual, final_objective):
                if scalar is not None:
                    finite_result = finite_result and math.isfinite(float(scalar))
            self.numerical_failure = self.numerical_failure or not finite_result
        effective_status = str(status)
        effective_reason = str(stopping_reason)
        if self.numerical_failure:
            effective_status = "numerical_error"
            effective_reason = self.numerical_error_reason or "non_finite_solver_state"
        else:
            effective_status = canonicalize_status(
                effective_status,
                algorithm=self.algorithm,
                iterations=int(actual_iterations),
                max_iterations=self.control.max_iterations,
                direct=self.algorithm in {"fbp", "fdk"},
            ).value
        evidence: dict[str, Any] = {}
        # A native loop can terminate at its configured limit without having
        # met a stopping criterion.  Use the recorded scalar trajectory only
        # to detect clear divergence/stagnation; never turn a budget stop
        # into a convergence claim.
        if self.records and effective_status in {"max_iterations", "partial"}:
            try:
                from ..convergence import classify_trajectory

                rows = [
                    {
                        "iteration": record.iteration,
                        "normalized_residual": record.normalized_data_residual,
                        "objective": record.objective,
                        "relative_iterate_change": record.relative_iterate_change,
                        "status": record.status,
                    }
                    for record in self.records
                ]
                classification = classify_trajectory(
                    rows,
                    max_iterations=self.control.max_iterations,
                    tolerance=float(self.control.tolerance or 0.0),
                    patience=self.control.patience,
                    algorithm=self.algorithm,
                )
                evidence["trajectory_classification"] = classification.status.value
                if classification.status.value in {"diverged", "stalled", "numerical_error"}:
                    effective_status = classification.status.value
                    effective_reason = classification.stopping_reason
            except (ImportError, AttributeError, ValueError, TypeError):
                # Diagnostics must not make a numerically valid reconstruction
                # fail merely because an optional classifier is unavailable.
                evidence["trajectory_classification"] = "unavailable"
        return SolveResult(
            reconstruction=reconstruction,
            actual_iterations=int(actual_iterations),
            status=effective_status,
            stopping_reason=effective_reason,
            trajectory=_sample_records(
                self.records,
                self.control.max_trajectory_points,
                patience=max(self.control.patience, self.control.stall_patience, self.control.divergence_patience),
            ),
            resources=stats,
            final_residual=final_residual,
            final_objective=final_objective,
            relative_iterate_change=last_change,
            predicted_measurement=predicted_measurement,
            metadata={
                "algorithm": self.algorithm,
                "max_iterations": self.control.max_iterations,
                "terminal_record": self.records[-1].to_dict() if self.records else None,
                "patience_window": [
                    record.to_dict()
                    for record in self.records[-max(self.control.patience, self.control.stall_patience, self.control.divergence_patience):]
                ],
                **dict(self.control.metadata),
                **dict(metadata or {}),
                **evidence,
            },
        )


@dataclass(frozen=True)
class StoppingDecision:
    checked: bool
    converged: bool
    stalled: bool
    diverged: bool
    consecutive: int
    criteria: Mapping[str, bool]


class ConsecutiveStoppingMonitor:
    """Shared min/check/patience state machine for solver-native criteria."""

    def __init__(self, control: SolveControl) -> None:
        self.control = control
        self.consecutive = 0
        self.stall_run = 0
        self.divergence_run = 0
        self.previous_monitor: float | None = None

    def observe(self, iteration: int, *, criteria: Mapping[str, bool], relative_change: float | None, monitor_value: float | None) -> StoppingDecision:
        checked = iteration >= self.control.min_iterations and iteration % self.control.check_every == 0
        if not checked:
            return StoppingDecision(False, False, False, False, self.consecutive, dict(criteria))
        satisfied = bool(criteria) and all(bool(value) for value in criteria.values())
        self.consecutive = self.consecutive + 1 if satisfied else 0
        discrepancy_unmet = not bool(criteria.get("discrepancy", True))
        native_items = [(k, v) for k, v in criteria.items() if k != "discrepancy"]
        native_values = [v for _k, v in native_items]
        native_unmet = bool(native_values) and not all(native_values)
        # For row-action policies the only native progress statistic is the
        # same iterate/epoch change used by the convergence criterion.  A
        # fixed point with an unmet discrepancy is therefore a stall even
        # though that native change criterion is technically satisfied.  For
        # solvers with a distinct native optimality metric, retain the
        # stricter unmet-native-evidence requirement.
        native_is_change_only = bool(native_items) and all(
            name in {"relative_iterate_change", "relative_epoch_change"}
            for name, _value in native_items
        )
        stall_native_ok = native_unmet or native_is_change_only
        if self.control.stall_enabled and relative_change is not None and relative_change <= self.control.stall_relative_iterate_tolerance and discrepancy_unmet and stall_native_ok:
            self.stall_run += 1
        else:
            self.stall_run = 0
        if self.control.divergence_enabled and monitor_value is not None and self.previous_monitor is not None:
            self.divergence_run = self.divergence_run + 1 if monitor_value > self.previous_monitor * (1.0 + self.control.divergence_relative_increase_tolerance) else 0
        self.previous_monitor = monitor_value
        return StoppingDecision(
            True,
            self.consecutive >= self.control.patience,
            self.stall_run >= self.control.stall_patience,
            self.divergence_run >= self.control.divergence_patience,
            self.consecutive,
            dict(criteria),
        )


def resolve_control(
    control: SolveControl | None,
    *,
    default_iterations: int,
    default_tolerance: float | None = None,
    callback: Callable[[IterationRecord], bool | None] | None = None,
) -> SolveControl:
    """Fill a detailed-control object with solver-specific defaults."""

    if control is None:
        return SolveControl(
            max_iterations=int(default_iterations),
            tolerance=default_tolerance,
            callback=callback,
        )
    return replace(
        control,
        max_iterations=(
            int(default_iterations)
            if control.max_iterations is None
            else control.max_iterations
        ),
        tolerance=(
            default_tolerance
            if control.tolerance is None
            else control.tolerance
        ),
        callback=callback if callback is not None else control.callback,
    )


class InverseProblemSolver(ABC):
    """Given a measurement y and operator A, return an estimate of x.

    All solvers in this framework expose the same `solve` signature so that
    classical, model-based (DIP, INR), and diffusion methods are
    interchangeable in benchmarks. Solvers that require linearity (e.g.
    FBP, SIRT, PiGDM) take a `LinearOperator` and assert that internally;
    everything else accepts any `ForwardOperator`.
    """

    @abstractmethod
    def solve(self,
              measurement: torch.Tensor,
              operator: ForwardOperator,
              **kwargs) -> torch.Tensor:
        ...

    def solve_detailed(
        self,
        measurement: torch.Tensor,
        operator: ForwardOperator,
        *,
        control: SolveControl | None = None,
        callback: Callable[[IterationRecord], bool | None] | None = None,
        **kwargs: Any,
    ) -> SolveResult:
        """Fallback for legacy solvers that do not expose native checkpoints.

        The fallback is intentionally conservative: it reports ``partial``
        and does not fabricate an iteration count or convergence claim.
        Core CT solvers override this method with actual loop diagnostics.
        """

        reconstruction = self.solve(measurement, operator, **kwargs)
        return SolveResult(
            reconstruction=reconstruction,
            actual_iterations=0,
            status="partial",
            stopping_reason="solver_did_not_expose_trajectory",
            metadata={"trajectory_available": False},
        )

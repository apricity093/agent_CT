"""Fair, multi-axis CT benchmark records and budget helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from math import isfinite
from typing import Any, Iterable, Mapping, Sequence


FIXED_DEFAULTS = "fixed_defaults"
EQUAL_OPERATOR_CALLS = "equal_operator_calls"
ORACLE_CALIBRATION = "oracle_calibration"
BENCHMARK_PROTOCOLS = (FIXED_DEFAULTS, EQUAL_OPERATOR_CALLS, ORACLE_CALIBRATION)


@dataclass(frozen=True)
class ProjectionSplit:
    """Deterministic angle-fold split for measurement-only validation.

    Indices refer to the original view axis.  Every fold is a validation
    subset and its complement is the corresponding training subset; no new
    angles are synthesized, so a limited-angle case keeps its original
    angular support.
    """

    case_id: str
    protocol_version: str
    validation_folds: tuple[tuple[int, ...], ...]
    sorted_view_indices: tuple[int, ...]
    split_sha256: str

    @property
    def num_views(self) -> int:
        return len(self.sorted_view_indices)

    @property
    def fold_count(self) -> int:
        return len(self.validation_folds)

    def training_indices(self, fold: int) -> tuple[int, ...]:
        if fold < 0 or fold >= self.fold_count:
            raise IndexError(f"fold must be between 0 and {self.fold_count - 1}")
        held_out = set(self.validation_folds[fold])
        return tuple(index for index in range(self.num_views) if index not in held_out)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "protocol_version": self.protocol_version,
            "num_views": self.num_views,
            "fold_count": self.fold_count,
            "sorted_view_indices": list(self.sorted_view_indices),
            "validation_folds": [list(fold) for fold in self.validation_folds],
            "training_folds": [list(self.training_indices(index)) for index in range(self.fold_count)],
            "split_sha256": self.split_sha256,
        }


def make_heldout_projection_split(
    case_id: str,
    angles_or_num_views: Sequence[float] | int,
    *,
    folds: int = 3,
    protocol_version: str = "heldout_projection_cv/v1",
) -> ProjectionSplit:
    """Create a stable, interleaved angle split shared by all candidates.

    Angles are sorted modulo ``pi`` before assigning validation folds in
    round-robin order.  Passing an integer is a convenience for synthetic
    tests and uses the original index order.
    """

    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("case_id must be a non-empty string")
    if isinstance(angles_or_num_views, bool):
        raise ValueError("angles_or_num_views must be a sequence or integer")
    if isinstance(angles_or_num_views, int):
        if angles_or_num_views <= 0:
            raise ValueError("num_views must be positive")
        pairs = [(float(index), index) for index in range(angles_or_num_views)]
    else:
        pairs = []
        try:
            for index, angle in enumerate(angles_or_num_views):
                value = float(angle)
                if not isfinite(value):
                    raise ValueError("angles must be finite")
                pairs.append((value % 3.141592653589793, index))
        except TypeError as error:
            raise ValueError("angles_or_num_views must be a sequence or integer") from error
    if not pairs:
        raise ValueError("at least one view is required")
    fold_count = int(folds)
    if isinstance(folds, bool) or fold_count < 2 or fold_count > len(pairs):
        raise ValueError("folds must be at least 2 and no greater than num_views")
    ordered = tuple(index for _angle, index in sorted(pairs, key=lambda item: (item[0], item[1])))
    validation = tuple(
        tuple(ordered[position] for position in range(fold_index, len(ordered), fold_count))
        for fold_index in range(fold_count)
    )
    canonical = {
        "case_id": case_id,
        "protocol_version": protocol_version,
        "sorted_view_indices": list(ordered),
        "validation_folds": [list(fold) for fold in validation],
    }
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return ProjectionSplit(
        case_id=case_id,
        protocol_version=str(protocol_version),
        validation_folds=validation,
        sorted_view_indices=ordered,
        split_sha256=digest,
    )


make_projection_split = make_heldout_projection_split


@dataclass(frozen=True)
class BenchmarkBudget:
    """A comparable tuning/execution budget for one benchmark group."""

    max_forward_calls: int | None = None
    max_adjoint_calls: int | None = None
    max_runtime_seconds: float | None = None
    tuning_trials: int = 0
    tuning_runtime_seconds: float | None = None
    tuning_max_forward_calls: int | None = None
    tuning_max_adjoint_calls: int | None = None
    protocol: str = FIXED_DEFAULTS

    def validate(self) -> None:
        if self.protocol not in BENCHMARK_PROTOCOLS:
            raise ValueError(f"unknown benchmark protocol {self.protocol!r}")
        for name in (
            "max_forward_calls",
            "max_adjoint_calls",
            "tuning_trials",
            "tuning_max_forward_calls",
            "tuning_max_adjoint_calls",
        ):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("max_runtime_seconds", "tuning_runtime_seconds"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isfinite(float(value)) or float(value) < 0):
                raise ValueError(f"{name} must be a finite nonnegative number")
        if self.protocol == EQUAL_OPERATOR_CALLS and self.max_forward_calls is None and self.max_adjoint_calls is None:
            raise ValueError("equal_operator_calls requires a forward or adjoint call budget")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_forward_calls": self.max_forward_calls,
            "max_adjoint_calls": self.max_adjoint_calls,
            "max_runtime_seconds": self.max_runtime_seconds,
            "tuning_trials": self.tuning_trials,
            "tuning_runtime_seconds": self.tuning_runtime_seconds,
            "tuning_max_forward_calls": self.tuning_max_forward_calls,
            "tuning_max_adjoint_calls": self.tuning_max_adjoint_calls,
            "protocol": self.protocol,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | "BenchmarkBudget") -> "BenchmarkBudget":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise ValueError("benchmark budget must be a mapping")
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{name: value[name] for name in fields if name in value})


@dataclass(frozen=True)
class BenchmarkResult:
    """One result with quality, consistency, optimization and resource axes."""

    algorithm: str
    solver: str
    case_id: str
    geometry: str | None = None
    observation_domain: str | None = None
    regularizer: str | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    parameter_source: str = "config"
    parameter_sources: Mapping[str, str] = field(default_factory=dict)
    tuning_protocol: str = FIXED_DEFAULTS
    convergence_status: str | None = None
    stopping_reason: str | None = None
    iterations: int | None = None
    runtime_seconds: float | None = None
    forward_calls: int | None = None
    adjoint_calls: int | None = None
    peak_memory_mb: float | None = None
    objective: float | None = None
    residual: float | None = None
    psnr: float | None = None
    ssim: float | None = None
    rmse: float | None = None
    failure_reason: str | None = None
    status: str = "success"
    budget: Mapping[str, Any] = field(default_factory=dict)
    device: str | None = None
    dtype: str | None = None
    initialization: str | None = None
    preprocessing: str | None = None
    mask_id: str | None = None

    @property
    def total_operator_calls(self) -> int | None:
        if self.forward_calls is None and self.adjoint_calls is None:
            return None
        return int(self.forward_calls or 0) + int(self.adjoint_calls or 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "solver": self.solver,
            "case_id": self.case_id,
            "geometry": self.geometry,
            "observation_domain": self.observation_domain,
            "regularizer": self.regularizer,
            "parameters": dict(self.parameters),
            "parameter_source": self.parameter_source,
            "parameter_sources": dict(self.parameter_sources),
            "tuning_protocol": self.tuning_protocol,
            "convergence_status": self.convergence_status,
            "stopping_reason": self.stopping_reason,
            "iterations": self.iterations,
            "runtime_seconds": self.runtime_seconds,
            "forward_calls": self.forward_calls,
            "adjoint_calls": self.adjoint_calls,
            "total_operator_calls": self.total_operator_calls,
            "peak_memory_mb": self.peak_memory_mb,
            "objective": self.objective,
            "residual": self.residual,
            "psnr": self.psnr,
            "ssim": self.ssim,
            "rmse": self.rmse,
            "failure_reason": self.failure_reason,
            "status": self.status,
            "budget": dict(self.budget),
            "device": self.device,
            "dtype": self.dtype,
            "initialization": self.initialization,
            "preprocessing": self.preprocessing,
            "mask_id": self.mask_id,
        }

    @classmethod
    def from_mapping(cls, record: Mapping[str, Any]) -> "BenchmarkResult":
        values = dict(record)
        values.setdefault("algorithm", values.get("solver", "unknown"))
        values.setdefault("solver", values["algorithm"])
        values.setdefault("case_id", "unknown")
        values.pop("total_operator_calls", None)
        return cls(**{key: values[key] for key in cls.__dataclass_fields__ if key in values})


def check_fairness(
    results: Iterable[BenchmarkResult | Mapping[str, Any]],
    *,
    shared_fields: tuple[str, ...] = (
        "case_id",
        "geometry",
        "observation_domain",
        "parameter_source",
        "tuning_protocol",
        "budget",
        "device",
        "dtype",
        "initialization",
        "preprocessing",
        "mask_id",
    ),
) -> dict[str, Any]:
    """Check that records in one comparison group share benchmark conditions."""

    rows = [row if isinstance(row, BenchmarkResult) else BenchmarkResult.from_mapping(row) for row in results]
    issues: list[str] = []
    if not rows:
        issues.append("comparison group is empty")
    for field_name in shared_fields:
        values = {repr(getattr(row, field_name)) for row in rows}
        if len(values) > 1:
            issues.append(f"shared field {field_name} differs across candidates")
    protocols = {row.tuning_protocol for row in rows}
    if len(protocols) > 1:
        issues.append("candidates use different tuning protocols")
    for row in rows:
        try:
            BenchmarkBudget.from_mapping(row.budget).validate()
        except ValueError as error:
            issues.append(f"{row.algorithm}: {error}")
    return {
        "schema_version": "inv_framework.ct_benchmark.v1",
        "fair": not issues,
        "issues": issues,
        "record_count": len(rows),
    }


def pareto_front(
    results: Iterable[BenchmarkResult | Mapping[str, Any]],
    *,
    minimize: tuple[str, ...] = ("residual", "runtime_seconds", "total_operator_calls"),
    maximize: tuple[str, ...] = ("psnr", "ssim"),
) -> list[BenchmarkResult]:
    """Return nondominated records; missing metrics never dominate a record."""

    rows = [row if isinstance(row, BenchmarkResult) else BenchmarkResult.from_mapping(row) for row in results]

    def value(row: BenchmarkResult, name: str) -> float | None:
        if name == "total_operator_calls":
            raw = row.total_operator_calls
        else:
            raw = getattr(row, name, None)
        try:
            return None if raw is None else float(raw)
        except (TypeError, ValueError):
            return None

    def dominates(left: BenchmarkResult, right: BenchmarkResult) -> bool:
        compared = False
        strictly_better = False
        for name in minimize:
            a, b = value(left, name), value(right, name)
            if a is None or b is None:
                continue
            compared = True
            if a > b:
                return False
            strictly_better |= a < b
        for name in maximize:
            a, b = value(left, name), value(right, name)
            if a is None or b is None:
                continue
            compared = True
            if a < b:
                return False
            strictly_better |= a > b
        return compared and strictly_better

    return [row for row in rows if not any(dominates(other, row) for other in rows if other is not row)]


def rank_by_group(
    results: Iterable[BenchmarkResult | Mapping[str, Any]],
    *,
    group_by: tuple[str, ...] = ("case_id",),
) -> dict[tuple[Any, ...], list[BenchmarkResult]]:
    """Group records without collapsing quality and resource axes into one score."""

    groups: dict[tuple[Any, ...], list[BenchmarkResult]] = {}
    for row in results:
        result = row if isinstance(row, BenchmarkResult) else BenchmarkResult.from_mapping(row)
        key = tuple(getattr(result, field_name, None) for field_name in group_by)
        groups.setdefault(key, []).append(result)
    for rows in groups.values():
        rows.sort(key=lambda item: (item.residual is None, item.residual if item.residual is not None else float("inf")))
    return groups


def enforce_budget(
    *,
    forward_calls: int,
    adjoint_calls: int,
    budget: BenchmarkBudget,
    runtime_seconds: float | None = None,
    tuning_trials: int | None = None,
    tuning_runtime_seconds: float | None = None,
) -> None:
    budget.validate()
    if budget.max_forward_calls is not None and forward_calls > budget.max_forward_calls:
        raise RuntimeError(f"forward-call budget exceeded: {forward_calls} > {budget.max_forward_calls}")
    if budget.max_adjoint_calls is not None and adjoint_calls > budget.max_adjoint_calls:
        raise RuntimeError(f"adjoint-call budget exceeded: {adjoint_calls} > {budget.max_adjoint_calls}")
    if runtime_seconds is not None and budget.max_runtime_seconds is not None and runtime_seconds > budget.max_runtime_seconds:
        raise RuntimeError(f"runtime budget exceeded: {runtime_seconds} > {budget.max_runtime_seconds}")
    if tuning_trials is not None and tuning_trials > budget.tuning_trials:
        raise RuntimeError(f"tuning-trial budget exceeded: {tuning_trials} > {budget.tuning_trials}")
    if tuning_runtime_seconds is not None and budget.tuning_runtime_seconds is not None and tuning_runtime_seconds > budget.tuning_runtime_seconds:
        raise RuntimeError(f"tuning runtime budget exceeded: {tuning_runtime_seconds} > {budget.tuning_runtime_seconds}")

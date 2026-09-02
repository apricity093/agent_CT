"""Fair, multi-axis CT benchmark records and budget helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from math import isfinite
import platform
from typing import Any, Iterable, Mapping, Sequence


FIXED_DEFAULTS = "fixed_defaults/v1"
EQUAL_TRIALS = "equal_trials/v1"
EQUAL_TUNING_TIME = "equal_tuning_time/v1"
EQUAL_OPERATOR_CALLS = "equal_operator_calls/v1"
COMMON_VALIDATION = "common_validation/v1"
ORACLE_UPPER_BOUND = "oracle_upper_bound/v1"
ORACLE_CALIBRATION = ORACLE_UPPER_BOUND
BENCHMARK_PROTOCOLS = (
    FIXED_DEFAULTS, EQUAL_TRIALS, EQUAL_TUNING_TIME, EQUAL_OPERATOR_CALLS,
    COMMON_VALIDATION, ORACLE_UPPER_BOUND,
)
OBSERVATION_STRATA = ("transmission", "emission_count", "fdk_backend")
CONTEXT_SCHEMA_VERSION = "ct.fair_comparison_context.v1"
PROTOCOL_SCHEMA_VERSION = "ct.fair_benchmark_protocol.v1"
_LEGACY_PROTOCOLS = {
    "fixed_defaults": FIXED_DEFAULTS,
    "equal_trials": EQUAL_TRIALS,
    "equal_tuning_time": EQUAL_TUNING_TIME,
    "equal_operator_calls": EQUAL_OPERATOR_CALLS,
    "equal_calls": EQUAL_OPERATOR_CALLS,
    "common_validation": COMMON_VALIDATION,
    "oracle_calibration": ORACLE_UPPER_BOUND,
    "oracle_upper_bound": ORACLE_UPPER_BOUND,
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def canonical_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def canonical_protocol(value: str | None) -> str:
    name = _LEGACY_PROTOCOLS.get(str(value or "fixed_defaults").strip(), str(value or "fixed_defaults").strip())
    if name not in BENCHMARK_PROTOCOLS:
        raise ValueError(f"unknown benchmark protocol {value!r}; expected one of {list(BENCHMARK_PROTOCOLS)}")
    return name


def environment_snapshot(*, dependencies: Mapping[str, Any] | None = None,
                         hardware: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "dependencies": _jsonable(dependencies or {}),
        "hardware": _jsonable(hardware or {}),
    }
    return {**payload, "digest": canonical_digest(payload)}


def _serialize_exception(value: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve historical partial exceptions; fairness validation still rejects them."""

    try:
        return FairnessException.from_mapping(value).to_dict()
    except ValueError:
        return dict(value)


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
class FairComparisonContext:
    """Every value that must be identical inside one ranked comparison."""

    input_data_id: str
    projection_data_sha256: str
    geometry: Mapping[str, Any]
    preprocessing: Mapping[str, Any]
    normalization: Mapping[str, Any]
    reconstruction_resolution: Sequence[int]
    mask_id: str | None
    initialization: Mapping[str, Any]
    seed: int
    device: str
    dtype: str
    precision: str
    warmup_policy: Mapping[str, Any]
    timing_policy: Mapping[str, Any]
    environment: Mapping[str, Any]
    tuning_budget: Mapping[str, Any]
    observation_stratum: str
    observation_model: str
    validation: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = CONTEXT_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != CONTEXT_SCHEMA_VERSION:
            raise ValueError(f"context schema_version must be {CONTEXT_SCHEMA_VERSION}")
        if self.observation_stratum not in OBSERVATION_STRATA:
            raise ValueError(f"observation_stratum must be one of {list(OBSERVATION_STRATA)}")
        if not self.input_data_id or not self.projection_data_sha256:
            raise ValueError("input_data_id and projection_data_sha256 are required")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if not self.reconstruction_resolution or any(
            isinstance(value, bool) or int(value) <= 0 for value in self.reconstruction_resolution
        ):
            raise ValueError("reconstruction_resolution must contain positive integers")
        environment = dict(self.environment)
        supplied = environment.pop("digest", None)
        if supplied is not None and supplied != canonical_digest(environment):
            raise ValueError("environment digest does not match the frozen environment")

    def _payload(self) -> dict[str, Any]:
        return {name: _jsonable(getattr(self, name)) for name in self.__dataclass_fields__}

    @property
    def digest(self) -> str:
        return canonical_digest(self._payload())

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {**self._payload(), "context_digest": self.digest}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FairComparisonContext":
        if not isinstance(value, Mapping):
            raise ValueError("shared comparison context must be a mapping")
        data = dict(value)
        supplied = data.pop("context_digest", None)
        try:
            context = cls(**{
                name: data[name] for name in cls.__dataclass_fields__ if name in data
            })
        except TypeError as error:
            raise ValueError(f"invalid shared comparison context: {error}") from error
        context.validate()
        if supplied is not None and supplied != context.digest:
            raise ValueError("context_digest does not match the frozen comparison context")
        return context


@dataclass(frozen=True)
class FairnessException:
    code: str
    reason: str
    basis: str
    affected_fields: tuple[str, ...]
    fairness_impact: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FairnessException":
        if not isinstance(value, Mapping):
            raise ValueError("fairness exception must be a mapping")
        required = {"code", "reason", "basis", "affected_fields", "fairness_impact"}
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(f"fairness exception missing required fields: {', '.join(missing)}")
        affected = value["affected_fields"]
        if not isinstance(affected, (list, tuple)) or not affected or not all(
            isinstance(item, str) and item for item in affected
        ):
            raise ValueError("fairness exception affected_fields must be a non-empty string list")
        strings = {name: str(value[name]).strip() for name in required - {"affected_fields"}}
        if not all(strings.values()):
            raise ValueError("fairness exception text fields must be non-empty")
        return cls(affected_fields=tuple(affected), **strings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code, "reason": self.reason, "basis": self.basis,
            "affected_fields": list(self.affected_fields),
            "fairness_impact": self.fairness_impact,
        }


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
        protocol = canonical_protocol(self.protocol)
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
        if protocol == FIXED_DEFAULTS and self.tuning_trials != 0:
            raise ValueError("fixed_defaults/v1 forbids tuning trials")
        if protocol == EQUAL_TRIALS and self.tuning_trials <= 0:
            raise ValueError("equal_trials/v1 requires tuning_trials > 0")
        if protocol == EQUAL_TUNING_TIME and self.tuning_runtime_seconds is None:
            raise ValueError("equal_tuning_time/v1 requires tuning_runtime_seconds")
        if protocol == EQUAL_OPERATOR_CALLS and all(value is None for value in (
            self.max_forward_calls, self.max_adjoint_calls,
            self.tuning_max_forward_calls, self.tuning_max_adjoint_calls,
        )):
            raise ValueError("equal_operator_calls/v1 requires a final or tuning operator-call ceiling")

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
        data = {name: value[name] for name in fields if name in value}
        canonical_protocol(data.get("protocol"))
        return cls(**data)


@dataclass(frozen=True)
class ComparisonProtocol:
    protocol_id: str
    budget: BenchmarkBudget
    validation: Mapping[str, Any] = field(default_factory=dict)
    agent_available: bool = True
    include_in_normal_ranking: bool = True
    schema_version: str = PROTOCOL_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != PROTOCOL_SCHEMA_VERSION:
            raise ValueError(f"protocol schema_version must be {PROTOCOL_SCHEMA_VERSION}")
        protocol = canonical_protocol(self.protocol_id)
        if canonical_protocol(self.budget.protocol) != protocol:
            raise ValueError("protocol_id and budget.protocol disagree")
        self.budget.validate()
        if protocol == COMMON_VALIDATION:
            if not self.validation.get("split_sha256") or isinstance(self.validation.get("folds"), bool) or not isinstance(self.validation.get("folds"), int) or self.validation["folds"] < 2:
                raise ValueError("common_validation/v1 requires validation split_sha256 and folds >= 2")
        if protocol == ORACLE_UPPER_BOUND and (self.agent_available or self.include_in_normal_ranking):
            raise ValueError("oracle_upper_bound/v1 must be unavailable to the Agent and excluded from normal rankings")

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "protocol_id": canonical_protocol(self.protocol_id),
            "budget": self.budget.to_dict(), "validation": _jsonable(self.validation),
            "agent_available": self.agent_available,
            "include_in_normal_ranking": self.include_in_normal_ranking,
        }
        return {**payload, "protocol_digest": canonical_digest(payload)} if include_digest else payload

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_dict(include_digest=False))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ComparisonProtocol":
        if not isinstance(value, Mapping):
            raise ValueError("comparison protocol must be a mapping")
        protocol_id = canonical_protocol(value.get("protocol_id") or value.get("name"))
        budget_raw = value.get("budget", {})
        if not isinstance(budget_raw, Mapping):
            raise ValueError("protocol budget must be a mapping")
        protocol = cls(
            protocol_id=protocol_id,
            budget=BenchmarkBudget.from_mapping({**dict(budget_raw), "protocol": protocol_id}),
            validation=dict(value.get("validation", {}) or {}),
            agent_available=bool(value.get("agent_available", protocol_id != ORACLE_UPPER_BOUND)),
            include_in_normal_ranking=bool(value.get("include_in_normal_ranking", protocol_id != ORACLE_UPPER_BOUND)),
            schema_version=str(value.get("schema_version", PROTOCOL_SCHEMA_VERSION)),
        )
        protocol.validate()
        if value.get("protocol_digest") not in {None, protocol.digest}:
            raise ValueError("protocol_digest does not match protocol configuration")
        return protocol


@dataclass(frozen=True)
class BenchmarkResult:
    """One result with quality, consistency, optimization and resource axes."""

    algorithm: str
    solver: str
    case_id: str
    geometry: str | None = None
    observation_domain: str | None = None
    observation_stratum: str | None = None
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
    relative_l2_error: float | None = None
    normalized_residual: float | None = None
    data_fidelity: float | None = None
    failure_reason: str | None = None
    status: str = "success"
    budget: Mapping[str, Any] = field(default_factory=dict)
    device: str | None = None
    dtype: str | None = None
    initialization: str | None = None
    preprocessing: str | None = None
    mask_id: str | None = None
    shared_context: Mapping[str, Any] = field(default_factory=dict)
    context_digest: str | None = None
    protocol_digest: str | None = None
    environment_digest: str | None = None
    resources: Mapping[str, Any] = field(default_factory=dict)
    tuning_usage: Mapping[str, Any] = field(default_factory=dict)
    tuning_provenance: Mapping[str, Any] = field(default_factory=dict)
    optimization_trajectories: Mapping[str, Any] = field(default_factory=dict)
    robustness: Mapping[str, Any] = field(default_factory=dict)
    fairness_exceptions: tuple[Mapping[str, Any], ...] = ()
    schema_version: str = "ct.benchmark_record.v2"

    @property
    def total_operator_calls(self) -> int | None:
        if self.forward_calls is None and self.adjoint_calls is None:
            return None
        return int(self.forward_calls or 0) + int(self.adjoint_calls or 0)

    def inferred_stratum(self) -> str:
        if self.observation_stratum:
            return self.observation_stratum
        domain = str(self.observation_domain or "").lower()
        if self.algorithm == "fdk" or "cone" in str(self.geometry).lower():
            return "fdk_backend"
        if domain in {"emission", "counts", "count", "emission_counts", "poisson_counts"} or self.algorithm in {"mlem", "osem"}:
            return "emission_count"
        return "transmission"

    def axes(self) -> dict[str, Any]:
        return {
            "reconstruction_quality": {
                "psnr": self.psnr, "ssim": self.ssim, "rmse": self.rmse,
                "relative_l2_error": self.relative_l2_error,
            },
            "data_consistency": {
                "projection_residual": self.residual,
                "normalized_residual": self.normalized_residual,
                "data_fidelity": self.data_fidelity,
            },
            "optimization_behavior": {
                "convergence_status": self.convergence_status,
                "stopping_reason": self.stopping_reason,
                "iterations": self.iterations, "objective": self.objective,
                "trajectories": _jsonable(self.optimization_trajectories),
            },
            "computational_efficiency": {
                **_jsonable(self.resources), "runtime_seconds": self.runtime_seconds,
                "forward_calls": self.forward_calls, "adjoint_calls": self.adjoint_calls,
                "total_operator_calls": self.total_operator_calls,
                "peak_memory_mb": self.peak_memory_mb,
            },
            "robustness": _jsonable(self.robustness),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "solver": self.solver,
            "case_id": self.case_id,
            "geometry": self.geometry,
            "observation_domain": self.observation_domain,
            "observation_stratum": self.inferred_stratum(),
            "regularizer": self.regularizer,
            "parameters": dict(self.parameters),
            "parameter_source": self.parameter_source,
            "parameter_sources": dict(self.parameter_sources),
            "tuning_protocol": canonical_protocol(self.tuning_protocol),
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
            "relative_l2_error": self.relative_l2_error,
            "normalized_residual": self.normalized_residual,
            "data_fidelity": self.data_fidelity,
            "failure_reason": self.failure_reason,
            "status": self.status,
            "budget": dict(self.budget),
            "device": self.device,
            "dtype": self.dtype,
            "initialization": self.initialization,
            "preprocessing": self.preprocessing,
            "mask_id": self.mask_id,
            "shared_context": _jsonable(self.shared_context),
            "context_digest": self.context_digest,
            "protocol_digest": self.protocol_digest,
            "environment_digest": self.environment_digest,
            "resources": _jsonable(self.resources),
            "tuning_usage": _jsonable(self.tuning_usage),
            "tuning_provenance": _jsonable(self.tuning_provenance),
            "optimization_trajectories": _jsonable(self.optimization_trajectories),
            "robustness": _jsonable(self.robustness),
            "fairness_exceptions": [_serialize_exception(value) for value in self.fairness_exceptions],
            "axes": self.axes(),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_mapping(cls, record: Mapping[str, Any]) -> "BenchmarkResult":
        values = dict(record)
        axes = values.pop("axes", {}) or {}
        quality = axes.get("reconstruction_quality", {}) if isinstance(axes, Mapping) else {}
        consistency = axes.get("data_consistency", {}) if isinstance(axes, Mapping) else {}
        values.setdefault("algorithm", values.get("solver", "unknown"))
        values.setdefault("solver", values["algorithm"])
        values.setdefault("case_id", "unknown")
        for name in ("psnr", "ssim", "rmse", "relative_l2_error"):
            values.setdefault(name, quality.get(name))
        values.setdefault("residual", consistency.get("projection_residual"))
        values.setdefault("normalized_residual", consistency.get("normalized_residual"))
        values.setdefault("data_fidelity", consistency.get("data_fidelity"))
        values.pop("total_operator_calls", None)
        values["tuning_protocol"] = canonical_protocol(values.get("tuning_protocol"))
        if values.get("shared_context"):
            context = FairComparisonContext.from_mapping(values["shared_context"])
            if values.get("context_digest") not in {None, context.digest}:
                raise ValueError("record context_digest does not match shared_context")
            values["context_digest"] = context.digest
        return cls(**{key: values[key] for key in cls.__dataclass_fields__ if key in values})


def check_fairness(
    results: Iterable[BenchmarkResult | Mapping[str, Any]],
    *,
    shared_fields: tuple[str, ...] = ("case_id", "context_digest", "tuning_protocol", "protocol_digest", "budget"),
) -> dict[str, Any]:
    """Check that records in one comparison group share benchmark conditions."""

    rows = [row if isinstance(row, BenchmarkResult) else BenchmarkResult.from_mapping(row) for row in results]
    issues: list[str] = []
    if not rows:
        issues.append("comparison group is empty")
    strata = {row.inferred_stratum() for row in rows}
    if len(strata) > 1:
        issues.append(f"incompatible observation strata cannot share a ranking: {sorted(strata)}")
    for field_name in shared_fields:
        values = {
            json.dumps(_jsonable(getattr(row, field_name)), sort_keys=True, separators=(",", ":"), allow_nan=False)
            for row in rows
        }
        if len(values) > 1:
            issues.append(f"shared field {field_name} differs across candidates")
    protocols = {row.tuning_protocol for row in rows}
    if len(protocols) > 1:
        issues.append("candidates use different tuning protocols")
    for row in rows:
        try:
            if row.algorithm in {"mlem", "osem"} and row.inferred_stratum() != "emission_count":
                raise ValueError("count-likelihood algorithm requires the emission_count stratum")
            if row.algorithm == "fdk" and row.inferred_stratum() != "fdk_backend":
                raise ValueError("FDK requires the fdk_backend stratum")
            if row.budget.get("protocol") is not None and canonical_protocol(row.budget.get("protocol")) != canonical_protocol(row.tuning_protocol):
                raise ValueError("record tuning_protocol and budget.protocol disagree")
            budget = BenchmarkBudget.from_mapping({**dict(row.budget), "protocol": row.tuning_protocol})
            usage = row.tuning_usage
            enforce_budget(
                forward_calls=int(row.forward_calls or 0), adjoint_calls=int(row.adjoint_calls or 0),
                budget=budget, runtime_seconds=row.runtime_seconds,
                tuning_trials=usage.get("completed_trials"),
                tuning_runtime_seconds=usage.get("runtime_seconds"),
                tuning_forward_calls=usage.get("forward_calls"),
                tuning_adjoint_calls=usage.get("adjoint_calls"),
            )
            for exception in row.fairness_exceptions:
                FairnessException.from_mapping(exception)
            if row.shared_context:
                context = FairComparisonContext.from_mapping(row.shared_context)
                if row.context_digest not in {None, context.digest}:
                    raise ValueError("context digest mismatch")
                if context.observation_stratum != row.inferred_stratum():
                    raise ValueError("record and shared context observation strata disagree")
            if canonical_protocol(row.tuning_protocol) == ORACLE_UPPER_BOUND:
                issues.append(f"{row.algorithm}: oracle_upper_bound/v1 is excluded from normal rankings")
        except (ValueError, RuntimeError) as error:
            issues.append(f"{row.algorithm}: {error}")
    return {
        "schema_version": "inv_framework.ct_fairness_report.v2",
        "fair": not issues,
        "rankable": not issues,
        "issues": issues,
        "record_count": len(rows),
        "observation_stratum": next(iter(strata)) if len(strata) == 1 else None,
        "context_digest": rows[0].context_digest if rows and len({row.context_digest for row in rows}) == 1 else None,
        "protocol_digest": rows[0].protocol_digest if rows and len({row.protocol_digest for row in rows}) == 1 else None,
    }


def pareto_front(
    results: Iterable[BenchmarkResult | Mapping[str, Any]],
    *,
    minimize: tuple[str, ...] = ("residual", "runtime_seconds", "total_operator_calls"),
    maximize: tuple[str, ...] = ("psnr", "ssim"),
) -> list[BenchmarkResult]:
    """Return nondominated records; missing metrics never dominate a record."""

    rows = [row if isinstance(row, BenchmarkResult) else BenchmarkResult.from_mapping(row) for row in results]
    report = check_fairness(rows)
    if not report["rankable"]:
        raise ValueError("unrankable CT comparison: " + "; ".join(report["issues"]))

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
    group_by: tuple[str, ...] = ("case_id", "observation_stratum"),
) -> dict[tuple[Any, ...], list[BenchmarkResult]]:
    """Group records without collapsing quality and resource axes into one score."""

    groups: dict[tuple[Any, ...], list[BenchmarkResult]] = {}
    for row in results:
        result = row if isinstance(row, BenchmarkResult) else BenchmarkResult.from_mapping(row)
        key = tuple(result.inferred_stratum() if field_name == "observation_stratum" else getattr(result, field_name, None) for field_name in group_by)
        groups.setdefault(key, []).append(result)
    for rows in groups.values():
        report = check_fairness(rows)
        if not report["rankable"]:
            raise ValueError("unrankable CT comparison: " + "; ".join(report["issues"]))
        rows.sort(key=lambda item: (item.residual is None, item.residual if item.residual is not None else float("inf")))
    return groups


def per_axis_rankings(results: Iterable[BenchmarkResult | Mapping[str, Any]]) -> dict[str, list[str]]:
    """Return transparent metric-specific orderings; no ranks are aggregated."""

    rows = [row if isinstance(row, BenchmarkResult) else BenchmarkResult.from_mapping(row) for row in results]
    report = check_fairness(rows)
    if not report["rankable"]:
        raise ValueError("unrankable CT comparison: " + "; ".join(report["issues"]))

    def rank(name: str, *, reverse: bool = False) -> list[str]:
        available = []
        for row in rows:
            raw = row.total_operator_calls if name == "total_operator_calls" else getattr(row, name, None)
            if raw is not None:
                available.append((row, float(raw)))
        return [row.algorithm for row, _value in sorted(available, key=lambda item: item[1], reverse=reverse)]

    return {
        "quality.psnr": rank("psnr", reverse=True),
        "quality.ssim": rank("ssim", reverse=True),
        "quality.rmse": rank("rmse"),
        "quality.relative_l2_error": rank("relative_l2_error"),
        "consistency.projection_residual": rank("residual"),
        "consistency.normalized_residual": rank("normalized_residual"),
        "consistency.data_fidelity": rank("data_fidelity"),
        "optimization.objective": rank("objective"),
        "optimization.iterations": rank("iterations"),
        "efficiency.runtime_seconds": rank("runtime_seconds"),
        "efficiency.total_operator_calls": rank("total_operator_calls"),
        "efficiency.peak_memory_mb": rank("peak_memory_mb"),
    }


def enforce_budget(
    *,
    forward_calls: int,
    adjoint_calls: int,
    budget: BenchmarkBudget,
    runtime_seconds: float | None = None,
    tuning_trials: int | None = None,
    tuning_runtime_seconds: float | None = None,
    tuning_forward_calls: int | None = None,
    tuning_adjoint_calls: int | None = None,
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
    if tuning_forward_calls is not None and budget.tuning_max_forward_calls is not None and tuning_forward_calls > budget.tuning_max_forward_calls:
        raise RuntimeError(f"tuning forward-call budget exceeded: {tuning_forward_calls} > {budget.tuning_max_forward_calls}")
    if tuning_adjoint_calls is not None and budget.tuning_max_adjoint_calls is not None and tuning_adjoint_calls > budget.tuning_max_adjoint_calls:
        raise RuntimeError(f"tuning adjoint-call budget exceeded: {tuning_adjoint_calls} > {budget.tuning_max_adjoint_calls}")

"""Structured CT solver and regularizer metadata.

The registry in this module is the single source of truth for capabilities
that have to be visible to an orchestrator.  Numerical implementations keep
their small, stable ``solve(measurement, operator)`` interface; this module
describes the assumptions around that interface.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any, Mapping


REGISTRY_SCHEMA_VERSION = "ct.algorithm_registry.v1"

# This is the only canonical ordering exposed by the ordinary-CT registry.
# Keep the tuple stable: it is used by the runtime, configuration checks and
# the Agent-side translation as a cross-layer inventory contract.
CANONICAL_ALGORITHM_IDS: tuple[str, ...] = (
    "fbp",
    "sirt",
    "landweber",
    "cgls",
    "lsqr",
    "sart",
    "os_sart",
    "mlem",
    "osem",
    "tikhonov",
    "tv_fista",
    "fdk",
)
CANONICAL_CT_ALGORITHM_IDS = CANONICAL_ALGORITHM_IDS

# Aliases are deliberately explicit.  In particular, ``ossart`` is a Python
# implementation function name, not an additional registry algorithm.
# The mapping direction is alias -> canonical ID.
ALGORITHM_ALIASES: dict[str, str] = {}
ALGORITHM_ALIAS_MAP = ALGORITHM_ALIASES

TRANSMISSION_OBSERVATION_MODELS: tuple[str, ...] = (
    "xray_transmission",
    "transmission",
    "line_integral",
    "log_projection",
    "poisson_log",
)
EMISSION_OBSERVATION_MODELS: tuple[str, ...] = (
    "emission",
    "emission_counts",
    "poisson_emission",
)

# Stable machine-readable compatibility codes.  Human-facing text remains in
# ``validate_compatibility`` for callers that already consume its strings.
COMPATIBILITY_REASON_CODES: dict[str, dict[str, str]] = {
    "unknown_algorithm": {
        "severity": "error",
        "description": "The requested algorithm is not in the canonical CT registry.",
    },
    "dimension_unsupported": {
        "severity": "error",
        "description": "The reconstruction dimension is not supported by the algorithm.",
    },
    "geometry_unsupported": {
        "severity": "error",
        "description": "The CT geometry is not supported by the algorithm.",
    },
    "observation_domain_unsupported": {
        "severity": "error",
        "description": "The observation domain is not supported by the algorithm.",
    },
    "observation_model_unsupported": {
        "severity": "error",
        "description": "The likelihood/observation model is not supported by the algorithm.",
    },
    "observation_not_nonnegative": {
        "severity": "error",
        "description": "The algorithm requires nonnegative observations.",
    },
    "observation_not_finite": {
        "severity": "error",
        "description": "The observations contain a non-finite value.",
    },
    "emission_observation_model_required": {
        "severity": "error",
        "description": "MLEM/OSEM require an explicit Poisson emission/count observation model.",
    },
    "regularizer_not_registered": {
        "severity": "error",
        "description": "The requested regularizer is not registered for this solver.",
    },
    "fixed_regularizer_pairing": {
        "severity": "error",
        "description": "The solver only supports its registered fixed regularizer pairing.",
    },
    "backend_requirement_missing": {
        "severity": "error",
        "description": "A required CT backend capability is unavailable.",
    },
    "cuda_device_required": {
        "severity": "error",
        "description": "The algorithm requires a CUDA device.",
    },
    "astra_cuda_required": {
        "severity": "error",
        "description": "The algorithm requires ASTRA with CUDA support.",
    },
    "cubic_volume_required": {
        "severity": "error",
        "description": "The FDK volume must be cubic.",
    },
    "parameter_unknown": {
        "severity": "error",
        "description": "A parameter is not declared by the selected algorithm.",
    },
    "parameter_invalid": {
        "severity": "error",
        "description": "A parameter violates its declared type or range.",
    },
    "parameter_constraint_violation": {
        "severity": "error",
        "description": "A cross-parameter or problem constraint is violated.",
    },
    "parameter_not_applicable": {
        "severity": "info",
        "description": "The requested parameter category is not part of this solver contract.",
    },
    "free_momentum_not_applicable": {
        "severity": "info",
        "description": "Momentum is fixed by the solver sequence and is not a free parameter.",
    },
}
CT_COMPATIBILITY_REASON_CODES = COMPATIBILITY_REASON_CODES

_ITERATIVE_ALGORITHM_IDS: tuple[str, ...] = tuple(
    name for name in CANONICAL_ALGORITHM_IDS if name not in {"fbp", "fdk"}
)

# These are metadata categories, not additional runtime parameters.  The
# explicit ``not_applicable`` entries prevent an Agent from inventing ADMM,
# primal/dual, or free-momentum values for the current twelve algorithms.
PARAMETER_APPLICABILITY: dict[str, dict[str, Any]] = {
    "regularization_strength": {
        "status": "applicable",
        "parameter_names": ("reg_strength",),
        "algorithms": ("tikhonov", "tv_fista"),
        "description": "Registered regularization strength for the fixed prior pairing.",
    },
    "learning_rate": {
        "status": "alias",
        "parameter_names": ("step_size",),
        "algorithms": ("landweber",),
        "canonical_parameter": "step_size",
        "description": "Landweber step_size semantic; no separate learning_rate field.",
    },
    "step_size": {
        "status": "applicable",
        "parameter_names": ("step_size",),
        "algorithms": ("landweber", "tv_fista"),
        "description": "Theory-constrained gradient/proximal step size.",
    },
    "iteration_number": {
        "status": "applicable",
        "parameter_names": ("num_iterations",),
        "algorithms": _ITERATIVE_ALGORITHM_IDS,
        "description": "Maximum iteration/epoch count, never a convergence claim.",
    },
    "tolerance": {
        "status": "applicable",
        "parameter_names": (
            "tol", "atol", "btol", "eps", "tolerance", "tv_tolerance",
        ),
        "algorithms": _ITERATIVE_ALGORITHM_IDS,
        "description": "Solver-native tolerance or numerical safeguard.",
    },
    "stopping_threshold": {
        "status": "policy_controlled",
        "parameter_names": (),
        "algorithms": _ITERATIVE_ALGORITHM_IDS,
        "description": "Provided by the versioned stopping policy, not a free solver parameter.",
    },
    "admm_rho": {
        "status": "not_applicable",
        "parameter_names": (),
        "algorithms": (),
        "reason_code": "parameter_not_applicable",
        "description": "No current canonical CT algorithm is an ADMM solver.",
    },
    "primal_step": {
        "status": "not_applicable",
        "parameter_names": (),
        "algorithms": (),
        "reason_code": "parameter_not_applicable",
        "description": "No current canonical CT algorithm exposes a primal-dual step.",
    },
    "dual_step": {
        "status": "not_applicable",
        "parameter_names": (),
        "algorithms": (),
        "reason_code": "parameter_not_applicable",
        "description": "No current canonical CT algorithm exposes a dual step.",
    },
    "primal_step_size": {
        "status": "not_applicable",
        "parameter_names": (),
        "algorithms": (),
        "reason_code": "parameter_not_applicable",
        "description": "No current canonical CT algorithm exposes a primal step size.",
    },
    "dual_step_size": {
        "status": "not_applicable",
        "parameter_names": (),
        "algorithms": (),
        "reason_code": "parameter_not_applicable",
        "description": "No current canonical CT algorithm exposes a dual step size.",
    },
    "momentum": {
        "status": "not_applicable",
        "parameter_names": (),
        "algorithms": (),
        "internal_sequence_algorithms": ("tv_fista",),
        "reason_code": "free_momentum_not_applicable",
        "description": "TV-FISTA uses its deterministic Nesterov sequence; free momentum is not tunable.",
    },
    "free_momentum": {
        "status": "not_applicable",
        "parameter_names": (),
        "algorithms": (),
        "internal_sequence_algorithms": ("tv_fista",),
        "reason_code": "free_momentum_not_applicable",
        "description": "The current registry has no free momentum parameter.",
    },
}
NON_APPLICABLE_PARAMETER_CATEGORIES: tuple[str, ...] = (
    "admm_rho",
    "primal_step",
    "dual_step",
    "primal_step_size",
    "dual_step_size",
    "momentum",
    "free_momentum",
)
NON_APPLICABLE_PARAMETERS = NON_APPLICABLE_PARAMETER_CATEGORIES


def _jsonable(value: Any, *, _path: str = "metadata") -> Any:
    """Convert registry values to strict JSON-compatible finite values."""

    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item, _path=f"{_path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item, _path=f"{_path}[{index}]") for index, item in enumerate(value)]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{_path} contains a non-finite number")
        return value
    raise TypeError(f"{_path} contains unsupported metadata type {type(value).__name__}")


def _parameter_applicability(
    algorithm: str,
    parameter_names: tuple[str, ...],
    *,
    direct: bool,
) -> dict[str, str]:
    """Return category status without adding category names to solve APIs."""

    names = set(parameter_names)
    result: dict[str, str] = {}
    for category, descriptor in PARAMETER_APPLICABILITY.items():
        if category in NON_APPLICABLE_PARAMETER_CATEGORIES:
            result[category] = "not_applicable"
            continue
        if category == "stopping_threshold":
            result[category] = "not_applicable" if direct else "policy_controlled"
            continue
        algorithms = set(descriptor.get("algorithms", ()))
        native_names = set(descriptor.get("parameter_names", ()))
        if algorithm in algorithms and names.intersection(native_names):
            result[category] = str(descriptor.get("status", "applicable"))
        elif algorithm in algorithms and not native_names:
            result[category] = str(descriptor.get("status", "applicable"))
        else:
            result[category] = "not_applicable"
    return result


@dataclass(frozen=True)
class ParameterSpec:
    """One solver parameter and its validation contract."""

    name: str
    type: str = "float"
    default: Any = None
    required: bool = False
    minimum: float | int | None = None
    maximum: float | int | None = None
    inclusive_minimum: bool = True
    inclusive_maximum: bool = True
    nullable: bool = False
    choices: tuple[Any, ...] = ()
    tunable: bool = False
    scale: str = "linear"
    theoretical_constraint: str | None = None
    description: str = ""
    category: str = "solver_parameter"
    units: str | None = None
    default_condition: str = "always"
    theoretical_derivation: str | None = None
    conditional_constraints: tuple[str, ...] = ()
    retry_transform: str | None = None
    cost_effect: str | None = None
    sensitivity: str = "unknown"
    serialization_name: str | None = None

    @property
    def value_type(self) -> str:
        """Alias useful to callers that avoid the built-in name ``type``."""

        return self.type

    def normalize(self, value: Any) -> Any:
        if value is None:
            if self.nullable or not self.required:
                return None
            raise ValueError(f"parameter {self.name} is required")

        if self.type == "int":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"parameter {self.name} must be an integer")
            normalized = int(value)
        elif self.type == "float":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"parameter {self.name} must be numeric")
            normalized = float(value)
            if not isfinite(normalized):
                raise ValueError(f"parameter {self.name} must be finite")
        elif self.type == "bool":
            if not isinstance(value, bool):
                raise ValueError(f"parameter {self.name} must be a bool")
            normalized = value
        elif self.type == "str":
            if not isinstance(value, str):
                raise ValueError(f"parameter {self.name} must be a string")
            normalized = value
        else:  # pragma: no cover - registry authoring error
            raise ValueError(f"unsupported parameter type {self.type!r}")

        if self.choices and normalized not in self.choices:
            choices = ", ".join(repr(choice) for choice in self.choices)
            raise ValueError(f"parameter {self.name} must be one of {choices}")
        if self.minimum is not None:
            invalid = normalized < self.minimum or (
                not self.inclusive_minimum and normalized == self.minimum
            )
            if invalid:
                operator = ">=" if self.inclusive_minimum else ">"
                raise ValueError(f"parameter {self.name} must be {operator} {self.minimum}")
        if self.maximum is not None:
            invalid = normalized > self.maximum or (
                not self.inclusive_maximum and normalized == self.maximum
            )
            if invalid:
                operator = "<=" if self.inclusive_maximum else "<"
                raise ValueError(f"parameter {self.name} must be {operator} {self.maximum}")
        return normalized

    def to_dict(self) -> dict[str, Any]:
        record = _jsonable(asdict(self))
        record["canonical_name"] = self.name
        record["serialization_name"] = self.serialization_name or self.name
        record["hard_constraints"] = {
            "minimum": self.minimum,
            "maximum": self.maximum,
            "inclusive_minimum": self.inclusive_minimum,
            "inclusive_maximum": self.inclusive_maximum,
            "choices": list(self.choices),
        }
        record["validation_only_search_space"] = []
        record["provenance_vocabulary"] = [
            "registry_default",
            "repository_config@sha256",
            "problem_metadata_rule",
            "problem_constraint",
            "power_iteration",
            "validation_search",
            "user_override",
            "recovery_attempt",
        ]
        return _jsonable(record)


@dataclass(frozen=True)
class RegularizerSpec:
    """A regularization prior and the operation it exposes to solvers."""

    name: str
    display_name: str
    prior_type: str
    has_proximal: bool
    has_gradient: bool
    convex: bool
    differentiable: bool
    parameter_names: tuple[str, ...] = ()
    parameter_sensitivity: str = "unknown"
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class CompatibilityRule:
    """A structured solver/regularizer/data-domain compatibility rule."""

    solver: str
    regularizer: str | None = None
    geometry_types: tuple[str, ...] = ()
    observation_domains: tuple[str, ...] = ()
    requires_nonnegative_observation: bool = False
    allowed: bool = True
    message: str = ""
    observation_models: tuple[str, ...] = ()
    reason_code: str = "compatibility_rule"
    severity: str = "error"
    evidence_source: str = "registry"
    remediation: str = ""

    def matches(
        self,
        *,
        solver: str,
        regularizer: str | None = None,
        geometry_type: str | None = None,
        observation_domain: str | None = None,
        observation_model: str | None = None,
    ) -> bool:
        if solver != self.solver:
            return False
        if self.regularizer is not None and regularizer != self.regularizer:
            return False
        if self.geometry_types and geometry_type not in self.geometry_types:
            return False
        if self.observation_domains and observation_domain not in self.observation_domains:
            return False
        if self.observation_models and observation_model not in self.observation_models:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class CTAlgorithmSpec:
    """Complete metadata for one runnable CT algorithm."""

    name: str
    display_name: str
    dimensions: tuple[int, ...]
    geometry_types: tuple[str, ...]
    parameter_names: tuple[str, ...]
    description: str
    backend: str | None = None
    family: str = "iterative"
    objective: str = "least_squares"
    likelihood: str | None = None
    regularizers: tuple[str, ...] = ()
    observation_domains: tuple[str, ...] = ("line_integral", "log_projection")
    noise_models: tuple[str, ...] = ("none", "gaussian_relative", "poisson_log")
    sparse_view: bool = False
    limited_angle: bool = False
    convex: bool | None = True
    differentiable: bool = True
    requires_proximal: bool = False
    supports_nonnegative_observation: bool = False
    initialization: str = "zeros"
    cost: str = "operator_iterative"
    failure_modes: tuple[str, ...] = ()
    convergence_criteria: tuple[str, ...] = ()
    direct: bool = False
    parameters: tuple[ParameterSpec, ...] = ()
    compatibility_rules: tuple[CompatibilityRule, ...] = ()
    schema_version: str = REGISTRY_SCHEMA_VERSION
    aliases: tuple[str, ...] = ()
    regularizer_pairing: str | None = None
    regularizer_pairing_policy: str = "none"
    compatibility_reason_codes: tuple[str, ...] = ()
    required_metadata: tuple[str, ...] = ()
    observation_models: tuple[str, ...] = ()
    requirements: tuple[str, ...] = ()
    parameter_applicability: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def parameter(self, name: str) -> ParameterSpec:
        for parameter in self.parameters:
            if parameter.name == name:
                return parameter
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        record = _jsonable(asdict(self))
        record["schema_version"] = self.schema_version
        record["registry_schema_version"] = REGISTRY_SCHEMA_VERSION
        record["canonical_id"] = self.name
        record["aliases"] = list(self.aliases)
        record["fixed_regularizer"] = self.regularizer_pairing
        record["regularizer_pairing"] = self.regularizer_pairing
        record["regularizer_pairing_policy"] = self.regularizer_pairing_policy
        record["compatibility_reason_codes"] = list(self.compatibility_reason_codes)
        record["required_metadata"] = list(self.required_metadata)
        record["observation_models"] = list(self.observation_models)
        record["requirements"] = list(self.requirements)
        record["parameter_applicability"] = dict(self.parameter_applicability)
        record["parameter_applicability_reason_codes"] = {
            category: PARAMETER_APPLICABILITY[category]["reason_code"]
            for category, status in self.parameter_applicability.items()
            if status == "not_applicable"
            and PARAMETER_APPLICABILITY.get(category, {}).get("reason_code")
        }
        parameter_records = []
        for parameter in self.parameters:
            parameter_record = parameter.to_dict()
            parameter_record["applicable_algorithms"] = [self.name]
            parameter_records.append(parameter_record)
        record["parameter_specs"] = parameter_records
        record["compatibility_rules"] = [rule.to_dict() for rule in self.compatibility_rules]
        return record


@dataclass(frozen=True)
class ParameterValidationResult:
    """Normalized parameters plus actionable validation diagnostics."""

    algorithm: str
    parameters: dict[str, Any]
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    estimates: dict[str, Any] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)
    reason_codes: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "parameters": _jsonable(self.parameters),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "estimates": _jsonable(self.estimates),
            "sources": {str(key): str(value) for key, value in self.sources.items()},
            "reason_codes": list(self.reason_codes),
            "valid": self.valid,
        }


def _p(name: str, type: str = "float", **kwargs: Any) -> ParameterSpec:
    return ParameterSpec(name=name, type=type, **kwargs)


def _common_parameters(num_iterations: int = 100) -> tuple[ParameterSpec, ...]:
    return (
    _p("num_iterations", "int", default=num_iterations, minimum=1, tunable=True, description="Maximum outer iterations."),
    _p("min_value", default=None, nullable=True, description="Optional lower box constraint."),
    _p("max_value", default=None, nullable=True, description="Optional upper box constraint."),
)


_COMMON = _common_parameters()


REGULARIZER_SPECS: dict[str, RegularizerSpec] = {
    "tikhonov": RegularizerSpec(
        name="tikhonov",
        display_name="Quadratic Tikhonov",
        prior_type="quadratic_l2",
        has_proximal=False,
        has_gradient=True,
        convex=True,
        differentiable=True,
        parameter_names=("reg_strength",),
        parameter_sensitivity="low_to_medium",
        description="Half the squared norm of the image or a supplied linear transform.",
    ),
    "tv": RegularizerSpec(
        name="tv",
        display_name="Total variation",
        prior_type="total_variation_l1_gradient",
        has_proximal=True,
        has_gradient=False,
        convex=True,
        differentiable=False,
        parameter_names=("reg_strength", "tv_mode", "tv_num_iterations", "tv_tolerance"),
        parameter_sensitivity="medium_to_high",
        description="Convex two-dimensional TV with a FGP proximal operator.",
    ),
}


def _algorithm(
    name: str,
    display_name: str,
    dimensions: tuple[int, ...],
    geometry_types: tuple[str, ...],
    description: str,
    parameters: tuple[ParameterSpec, ...],
    *,
    backend: str | None = None,
    family: str = "iterative",
    objective: str = "least_squares",
    likelihood: str | None = None,
    regularizers: tuple[str, ...] = (),
    observation_domains: tuple[str, ...] = ("line_integral", "log_projection"),
    noise_models: tuple[str, ...] = ("none", "gaussian_relative", "poisson_log"),
    sparse_view: bool = False,
    limited_angle: bool = False,
    convex: bool | None = True,
    differentiable: bool = True,
    requires_proximal: bool = False,
    supports_nonnegative_observation: bool = False,
    initialization: str = "zeros",
    cost: str = "operator_iterative",
    failure_modes: tuple[str, ...] = (),
    convergence_criteria: tuple[str, ...] = (),
    direct: bool = False,
    compatibility_rules: tuple[CompatibilityRule, ...] = (),
    aliases: tuple[str, ...] = (),
    regularizer_pairing: str | None = None,
    compatibility_reason_codes: tuple[str, ...] = (),
    required_metadata: tuple[str, ...] = (),
    observation_models: tuple[str, ...] = (),
    requirements: tuple[str, ...] = (),
    parameter_applicability: Mapping[str, str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CTAlgorithmSpec:
    parameter_names = tuple(parameter.name for parameter in parameters)
    if regularizer_pairing is None and name in {"tikhonov", "tv_fista"} and len(regularizers) == 1:
        regularizer_pairing = regularizers[0]
    pairing_policy = "fixed" if regularizer_pairing is not None else "none"
    if not observation_models:
        observation_models = (
            EMISSION_OBSERVATION_MODELS
            if name in {"mlem", "osem"}
            else TRANSMISSION_OBSERVATION_MODELS
        )
    if not requirements:
        if backend:
            requirements = (backend, "CUDA", "ASTRA", "cone_3d", "cubic_voxels")
        else:
            requirements = ("linear_operator",)
    if not required_metadata:
        required_metadata = (
            "dimension",
            "geometry_type",
            "observation_domain",
            "noise_model",
        )
        if not direct:
            required_metadata += ("dtype", "device")
        if name in {"mlem", "osem"}:
            required_metadata += ("measurement_kind", "observation_model")
        if name == "fdk":
            required_metadata += ("backend_capabilities",)
    reason_codes = list(compatibility_reason_codes)
    for code in ("dimension_unsupported", "geometry_unsupported", "observation_domain_unsupported"):
        if code not in reason_codes:
            reason_codes.append(code)
    if observation_models:
        if "observation_model_unsupported" not in reason_codes:
            reason_codes.append("observation_model_unsupported")
    if supports_nonnegative_observation:
        for code in ("observation_not_nonnegative", "emission_observation_model_required"):
            if code not in reason_codes:
                reason_codes.append(code)
    if regularizer_pairing is not None and "fixed_regularizer_pairing" not in reason_codes:
        reason_codes.append("fixed_regularizer_pairing")
    if backend:
        for code in ("backend_requirement_missing", "cuda_device_required", "astra_cuda_required"):
            if code not in reason_codes:
                reason_codes.append(code)
    if name == "fdk" and "cubic_volume_required" not in reason_codes:
        reason_codes.append("cubic_volume_required")
    applicability = (
        dict(parameter_applicability)
        if parameter_applicability is not None
        else _parameter_applicability(name, parameter_names, direct=direct)
    )
    contract_metadata = {
        "canonical_id": name,
        "aliases": tuple(aliases),
        "fixed_regularizer": regularizer_pairing,
        "regularizer_pairing": regularizer_pairing,
        "regularizer_pairing_policy": pairing_policy,
        "compatibility_reason_codes": tuple(reason_codes),
        "required_metadata": tuple(required_metadata),
        "observation_models": tuple(observation_models),
        "requirements": tuple(requirements),
        "parameter_applicability": applicability,
        "parameter_applicability_reason_codes": {
            category: PARAMETER_APPLICABILITY[category]["reason_code"]
            for category, status in applicability.items()
            if status == "not_applicable"
            and PARAMETER_APPLICABILITY.get(category, {}).get("reason_code")
        },
        "parameter_categories_are_metadata_only": True,
        "momentum_semantics": (
            "deterministic_nesterov_sequence"
            if name == "tv_fista" else "not_applicable"
        ),
    }
    contract_metadata.update(dict(metadata or {}))
    return CTAlgorithmSpec(
        name=name,
        display_name=display_name,
        dimensions=dimensions,
        geometry_types=geometry_types,
        parameter_names=parameter_names,
        description=description,
        backend=backend,
        family=family,
        objective=objective,
        likelihood=likelihood,
        regularizers=regularizers,
        observation_domains=observation_domains,
        noise_models=noise_models,
        sparse_view=sparse_view,
        limited_angle=limited_angle,
        convex=convex,
        differentiable=differentiable,
        requires_proximal=requires_proximal,
        supports_nonnegative_observation=supports_nonnegative_observation,
        initialization=initialization,
        cost=cost,
        failure_modes=failure_modes,
        convergence_criteria=convergence_criteria,
        direct=direct,
        parameters=parameters,
        compatibility_rules=compatibility_rules,
        aliases=tuple(aliases),
        regularizer_pairing=regularizer_pairing,
        regularizer_pairing_policy=pairing_policy,
        compatibility_reason_codes=tuple(reason_codes),
        required_metadata=tuple(required_metadata),
        observation_models=tuple(observation_models),
        requirements=tuple(requirements),
        parameter_applicability=applicability,
        metadata=contract_metadata,
    )


SOLVER_SPECS: dict[str, CTAlgorithmSpec] = {
    "fbp": _algorithm(
        "fbp", "FBP", (2,), ("parallel_2d",), "Filtered backprojection.",
        (_p("scale", default=None, nullable=True, minimum=0.0, inclusive_minimum=False, description="Optional analytic scale."),),
        family="analytical", objective="filtered_backprojection", sparse_view=False,
        limited_angle=True, cost="one_adjoint_plus_filter", direct=True,
        failure_modes=("sparse_view_aliasing", "limited_angle_null_space"),
        convergence_criteria=("non_iterative_completion",),
    ),
    "sirt": _algorithm(
        "sirt", "SIRT", (2,), ("parallel_2d",), "Simultaneous iterative reconstruction.",
        _COMMON, sparse_view=True, limited_angle=True,
        failure_modes=("slow_convergence", "over_smoothing"),
        convergence_criteria=("normalized_residual", "relative_iterate_change"),
    ),
    "landweber": _algorithm(
        "landweber", "Landweber", (2,), ("parallel_2d",), "Landweber gradient iteration.",
        _COMMON + (_p("step_size", default=None, nullable=True, minimum=0.0, inclusive_minimum=False, tunable=True, theoretical_constraint="0 < alpha < 2 / ||A||^2"),),
        sparse_view=True, limited_angle=True,
        failure_modes=("step_size_instability", "slow_convergence"),
        convergence_criteria=("normalized_residual", "relative_iterate_change"),
    ),
    "cgls": _algorithm(
        "cgls", "CGLS", (2,), ("parallel_2d",), "Conjugate-gradient least squares.",
        _common_parameters(10) + (_p("tol", minimum=0.0, default=1e-6, tunable=True, description="Residual tolerance."),),
        sparse_view=True, limited_angle=True,
        failure_modes=("ill_conditioning", "normal_equation_breakdown"),
        convergence_criteria=("normalized_normal_residual", "data_residual", "relative_iterate_change"),
    ),
    "lsqr": _algorithm(
        "lsqr", "LSQR", (2,), ("parallel_2d",), "Golub-Kahan LSQR.",
        _common_parameters(10) + (
            _p("damping", minimum=0.0, default=0.0, tunable=True),
            _p("atol", minimum=0.0, default=1e-6, tunable=True),
            _p("btol", minimum=0.0, default=1e-6, tunable=True),
        ), sparse_view=True, limited_angle=True,
        failure_modes=("ill_conditioning", "premature_tolerance"),
        convergence_criteria=("normalized_normal_residual", "data_residual", "relative_iterate_change"),
    ),
    "sart": _algorithm(
        "sart", "SART", (2,), ("parallel_2d",), "Ordered row-action reconstruction.",
        _COMMON + (
            _p("block_size", "int", default=1, minimum=1, tunable=True),
            _p("order_strategy", "str", default="ordered", choices=("ordered", "random")),
            _p("seed", "int", default=None, nullable=True),
            _p("relaxation", minimum=0.0, maximum=1.0, inclusive_minimum=False, default=1.0, tunable=True, theoretical_constraint="0 < relaxation <= 1"),
            _p("eps", minimum=0.0, inclusive_minimum=False, default=1e-8),
        ), sparse_view=True, limited_angle=True,
        failure_modes=("relaxation_instability", "subset_bias"),
        convergence_criteria=("complete_sweep_residual", "complete_sweep_iterate_change"),
    ),
    "os_sart": _algorithm(
        "os_sart", "OS-SART", (2,), ("parallel_2d",), "Ordered-subsets SART.",
        _COMMON + (
            _p("block_size", "int", default=None, nullable=True, minimum=1, tunable=True),
            _p("subset_count", "int", default=None, nullable=True, minimum=1, tunable=True),
            _p("order_strategy", "str", default="ordered", choices=("ordered", "random")),
            _p("seed", "int", default=None, nullable=True),
            _p("relaxation", minimum=0.0, maximum=1.0, inclusive_minimum=False, default=1.0, tunable=True, theoretical_constraint="0 < relaxation <= 1"),
            _p("eps", minimum=0.0, inclusive_minimum=False, default=1e-8),
        ), sparse_view=True, limited_angle=True,
        failure_modes=("relaxation_instability", "subset_bias", "incomplete_sweep"),
        convergence_criteria=("complete_sweep_residual", "complete_sweep_iterate_change"),
    ),
    "mlem": _algorithm(
        "mlem", "MLEM", (2,), ("parallel_2d",), "Maximum-likelihood EM for nonnegative count/intensity data.",
        _common_parameters(50) + (
            _p("initial_value", minimum=0.0, inclusive_minimum=False, default=1e-6),
            _p("eps", minimum=0.0, inclusive_minimum=False, default=1e-8),
        ), family="statistical", objective="poisson_likelihood", likelihood="poisson_emission",
        observation_domains=("nonnegative_counts", "intensity"),
        noise_models=("poisson_counts", "poisson"), sparse_view=True,
        supports_nonnegative_observation=True, initialization="positive_constant",
        failure_modes=("log_projection_incompatible", "zero_sensitivity", "slow_convergence"),
        convergence_criteria=("poisson_deviance", "complete_sweep_iterate_change"),
        compatibility_rules=(CompatibilityRule(
            "mlem",
            observation_domains=("nonnegative_counts", "intensity"),
            observation_models=EMISSION_OBSERVATION_MODELS,
            requires_nonnegative_observation=True,
            reason_code="emission_observation_model_required",
            remediation="Provide nonnegative count/intensity data with a Poisson emission model.",
        ),),
    ),
    "osem": _algorithm(
        "osem", "OSEM", (2,), ("parallel_2d",), "Ordered-subsets EM for nonnegative count/intensity data.",
        _common_parameters(50) + (
            _p("block_size", "int", default=None, nullable=True, minimum=1, tunable=True),
            _p("subset_count", "int", default=None, nullable=True, minimum=1, tunable=True),
            _p("order_strategy", "str", default="ordered", choices=("ordered", "random")),
            _p("seed", "int", default=None, nullable=True),
            _p("initial_value", minimum=0.0, inclusive_minimum=False, default=1e-6),
            _p("eps", minimum=0.0, inclusive_minimum=False, default=1e-8),
        ), family="statistical", objective="poisson_likelihood", likelihood="poisson_emission",
        observation_domains=("nonnegative_counts", "intensity"),
        noise_models=("poisson_counts", "poisson"), sparse_view=True,
        supports_nonnegative_observation=True, initialization="positive_constant",
        failure_modes=("log_projection_incompatible", "subset_limit_cycle", "zero_sensitivity"),
        convergence_criteria=("poisson_deviance", "complete_sweep_iterate_change"),
        compatibility_rules=(CompatibilityRule(
            "osem",
            observation_domains=("nonnegative_counts", "intensity"),
            observation_models=EMISSION_OBSERVATION_MODELS,
            requires_nonnegative_observation=True,
            reason_code="emission_observation_model_required",
            remediation="Provide nonnegative count/intensity data with a Poisson emission model.",
        ),),
    ),
    "tikhonov": _algorithm(
        "tikhonov", "Tikhonov", (2,), ("parallel_2d",), "Quadratic Tikhonov reconstruction.",
        _COMMON + (
            _p("reg_strength", minimum=0.0, default=1e-2, tunable=True, scale="log"),
            _p("tolerance", minimum=0.0, default=1e-6, tunable=True),
        ), family="variational", objective="quadratic_regularized_least_squares",
        regularizers=("tikhonov",), sparse_view=True, limited_angle=True,
        failure_modes=("normal_equation_breakdown", "over_smoothing"),
        convergence_criteria=("normalized_regularized_normal_residual", "data_residual", "relative_iterate_change"),
    ),
    "tv_fista": _algorithm(
        "tv_fista", "TV-FISTA", (2,), ("parallel_2d",), "TV-regularized FISTA.",
        _common_parameters(50) + (
            _p("reg_strength", minimum=0.0, default=1e-3, tunable=True, scale="log"),
            _p("step_size", default=None, nullable=True, minimum=0.0, inclusive_minimum=False, tunable=True, theoretical_constraint="0 < step_size <= 1 / ||A||^2"),
            _p("tolerance", minimum=0.0, default=1e-5, tunable=True),
            _p("power_iterations", "int", default=12, minimum=1),
            _p("tv_mode", "str", default="isotropic", choices=("isotropic", "anisotropic")),
            _p("tv_num_iterations", "int", default=50, minimum=1),
            _p("tv_tolerance", minimum=0.0, default=1e-5),
        ), family="variational", objective="tv_regularized_least_squares",
        regularizers=("tv",), sparse_view=True, limited_angle=True,
        requires_proximal=True,
        failure_modes=("step_size_instability", "proximal_budget", "over_smoothing"),
        convergence_criteria=("composite_objective", "prox_gradient_mapping", "relative_iterate_change"),
    ),
    "fdk": _algorithm(
        "fdk", "FDK", (3,), ("cone_3d",), "Cone-beam Feldkamp-Davis-Kress reconstruction.",
        (
            _p("filter_type", "str", default="ram-lak"),
            _p("short_scan", "bool", default=False),
            _p("voxel_supersampling", "int", default=1, minimum=1),
        ), backend="ASTRA CUDA", family="analytical_3d", objective="fdk_filter_backprojection",
        observation_domains=("line_integral", "log_projection"), noise_models=("none",),
        cost="one_backend_reconstruction", direct=True,
        failure_modes=("cuda_unavailable", "cone_geometry_mismatch", "short_scan_artifact"),
        convergence_criteria=("non_iterative_completion",),
    ),
}


def validate_registry() -> tuple[str, ...]:
    """Validate the registry shape and metadata without constructing solvers."""

    registry_ids = tuple(SOLVER_SPECS)
    if registry_ids != CANONICAL_ALGORITHM_IDS:
        missing = sorted(set(CANONICAL_ALGORITHM_IDS) - set(registry_ids))
        extra = sorted(set(registry_ids) - set(CANONICAL_ALGORITHM_IDS))
        raise ValueError(
            "ordinary CT registry must contain exactly the canonical IDs "
            f"(missing={missing}, extra={extra}, order={registry_ids!r})"
        )
    if set(ALGORITHM_ALIASES).intersection(CANONICAL_ALGORITHM_IDS):
        raise ValueError("an algorithm ID cannot also be an alias")
    for alias, target in ALGORITHM_ALIASES.items():
        if not alias or alias == target or target not in CANONICAL_ALGORITHM_IDS:
            raise ValueError(f"invalid algorithm alias {alias!r} -> {target!r}")
    for name in CANONICAL_ALGORITHM_IDS:
        spec = SOLVER_SPECS[name]
        if spec.name != name or spec.aliases != tuple(
            alias for alias, target in ALGORITHM_ALIASES.items() if target == name
        ):
            raise ValueError(f"registry identity/alias mismatch for {name!r}")
        if len(set(spec.parameter_names)) != len(spec.parameter_names):
            raise ValueError(f"duplicate parameter name in {name!r}")
        if spec.parameter_names != tuple(parameter.name for parameter in spec.parameters):
            raise ValueError(f"parameter metadata mismatch for {name!r}")
        if set(spec.parameter_applicability) != set(PARAMETER_APPLICABILITY):
            raise ValueError(f"parameter applicability metadata mismatch for {name!r}")
        if any(
            status not in {"applicable", "alias", "policy_controlled", "not_applicable"}
            for status in spec.parameter_applicability.values()
        ):
            raise ValueError(f"unknown parameter applicability status for {name!r}")
        if any(
            spec.parameter_applicability[category] != "not_applicable"
            for category in NON_APPLICABLE_PARAMETER_CATEGORIES
        ):
            raise ValueError(f"non-applicable parameter category was enabled for {name!r}")
        if spec.regularizer_pairing is not None:
            if spec.regularizer_pairing not in spec.regularizers:
                raise ValueError(f"fixed regularizer pairing is not registered for {name!r}")
            if len(spec.regularizers) != 1 or spec.regularizer_pairing_policy != "fixed":
                raise ValueError(f"fixed regularizer pairing must be singular for {name!r}")
        expected_pairing = {"tikhonov": "tikhonov", "tv_fista": "tv"}.get(name)
        if expected_pairing is not None and (
            spec.regularizer_pairing != expected_pairing
            or spec.regularizer_pairing_policy != "fixed"
        ):
            raise ValueError(f"{name!r} must preserve its fixed regularizer pairing")
        if not spec.required_metadata:
            raise ValueError(f"required metadata is missing for {name!r}")
        unknown_codes = set(spec.compatibility_reason_codes) - set(COMPATIBILITY_REASON_CODES)
        if unknown_codes:
            raise ValueError(f"unknown compatibility reason codes for {name!r}: {sorted(unknown_codes)}")
        unknown_parameter_codes = {
            str(descriptor["reason_code"])
            for descriptor in PARAMETER_APPLICABILITY.values()
            if descriptor.get("reason_code")
        } - set(COMPATIBILITY_REASON_CODES)
        if unknown_parameter_codes:
            raise ValueError(
                "unknown parameter applicability reason codes: "
                f"{sorted(unknown_parameter_codes)}"
            )
        # ``to_dict`` performs the recursive finite/structured check.  The
        # explicit json dump also guards against a future registry field that
        # is not accepted by the recursive converter.
        json.dumps(spec.to_dict(), sort_keys=True, allow_nan=False)
    json.dumps(PARAMETER_APPLICABILITY, sort_keys=True, allow_nan=False, default=list)
    return registry_ids


def registry_contract() -> dict[str, Any]:
    """Return the versioned ordinary-CT inventory/metadata contract."""

    validate_registry()
    records = [SOLVER_SPECS[name].to_dict() for name in CANONICAL_ALGORITHM_IDS]
    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "canonical_algorithm_ids": list(CANONICAL_ALGORITHM_IDS),
        "aliases": dict(ALGORITHM_ALIASES),
        "regularizers": regularizer_records(),
        "parameter_applicability": _jsonable(PARAMETER_APPLICABILITY),
        "non_applicable_parameter_categories": list(NON_APPLICABLE_PARAMETER_CATEGORIES),
        "compatibility_reason_codes": _jsonable(COMPATIBILITY_REASON_CODES),
        "required_algorithm_fields": [
            "name", "display_name", "dimensions", "geometry_types", "parameter_names",
            "family", "objective", "observation_domains", "noise_models", "parameters",
            "canonical_id", "aliases", "fixed_regularizer", "regularizer_pairing",
            "regularizer_pairing_policy", "observation_models", "parameter_applicability",
            "compatibility_reason_codes", "required_metadata", "requirements",
        ],
        "algorithms": records,
    }


def registry_digest() -> str:
    """Return a reproducible digest of the serialized registry contract."""

    payload = json.dumps(registry_contract(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def solver_records() -> list[dict[str, Any]]:
    """Return deterministic, JSON-compatible records for all CT solvers."""

    validate_registry()
    return [SOLVER_SPECS[name].to_dict() for name in CANONICAL_ALGORITHM_IDS]


def regularizer_records() -> list[dict[str, Any]]:
    return [spec.to_dict() for spec in REGULARIZER_SPECS.values()]


def _compatibility_issue(
    code: str,
    message: str,
    *,
    remediation: str = "",
    evidence_source: str = "registry",
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    definition = COMPATIBILITY_REASON_CODES.get(code, {})
    return {
        "code": code,
        "severity": str(definition.get("severity", "error")),
        "message": message,
        "evidence_source": evidence_source,
        "remediation": remediation,
        "details": _jsonable(dict(details or {})),
    }


def compatibility_diagnostics(
    algorithm: str,
    *,
    geometry_type: str | None = None,
    dimension: int | None = None,
    observation_domain: str | None = None,
    regularizer: str | None = None,
    observation_model: str | None = None,
    observation_min: float | None = None,
    observation_finite: bool | None = None,
) -> list[dict[str, Any]]:
    """Return structured compatibility errors without constructing a solver."""

    if algorithm not in SOLVER_SPECS:
        return [_compatibility_issue(
            "unknown_algorithm",
            f"unknown solver: {algorithm!r}",
            remediation="Choose one of the canonical ordinary-CT algorithm IDs.",
        )]
    spec = SOLVER_SPECS[algorithm]
    issues: list[dict[str, Any]] = []
    if dimension is not None:
        try:
            dimension_value = int(dimension)
        except (TypeError, ValueError):
            dimension_value = None
        if dimension_value not in spec.dimensions:
            issues.append(_compatibility_issue(
                "dimension_unsupported",
                f"solver {algorithm!r} supports dimensions {spec.dimensions}, got {dimension}",
                details={"supported": spec.dimensions, "received": dimension},
            ))
    if geometry_type is not None and geometry_type not in spec.geometry_types:
        issues.append(_compatibility_issue(
            "geometry_unsupported",
            f"solver {algorithm!r} does not support geometry {geometry_type!r}",
            details={"supported": spec.geometry_types, "received": geometry_type},
        ))
    if observation_domain is not None and observation_domain not in spec.observation_domains:
        issues.append(_compatibility_issue(
            "observation_domain_unsupported",
            f"solver {algorithm!r} does not support observation domain {observation_domain!r}; "
            f"supported domains are {spec.observation_domains}",
            details={"supported": spec.observation_domains, "received": observation_domain},
        ))
    if observation_model is not None and observation_model not in spec.observation_models:
        issues.append(_compatibility_issue(
            "observation_model_unsupported",
            f"solver {algorithm!r} does not support observation model {observation_model!r}; "
            f"supported models are {spec.observation_models}",
            details={"supported": spec.observation_models, "received": observation_model},
        ))
    if regularizer is not None and regularizer not in spec.regularizers:
        code = "fixed_regularizer_pairing" if spec.regularizer_pairing is not None else "regularizer_not_registered"
        issues.append(_compatibility_issue(
            code,
            f"solver {algorithm!r} is not registered with regularizer {regularizer!r}",
            remediation=(
                f"Use the fixed {spec.regularizer_pairing!r} pairing."
                if spec.regularizer_pairing is not None else "Omit the unsupported regularizer."
            ),
            details={"registered": spec.regularizers, "received": regularizer},
        ))
    for rule in spec.compatibility_rules:
        if rule.matches(
            solver=algorithm,
            regularizer=regularizer,
            geometry_type=geometry_type,
            observation_domain=observation_domain,
            observation_model=observation_model,
        ) and not rule.allowed:
            issues.append(_compatibility_issue(
                rule.reason_code,
                rule.message or f"incompatible solver combination for {algorithm!r}",
                remediation=rule.remediation,
                evidence_source=rule.evidence_source,
            ))
    if spec.supports_nonnegative_observation and observation_domain in {"log_projection", "line_integral"}:
        issues.append(_compatibility_issue(
            "emission_observation_model_required",
            f"{algorithm} requires nonnegative count/intensity observations; "
            f"{observation_domain} is a transmission/log projection domain",
            remediation="Use a count/intensity emission case or a transmission likelihood solver.",
            details={"received_domain": observation_domain},
        ))
    if spec.supports_nonnegative_observation and observation_min is not None:
        try:
            negative = float(observation_min) < 0.0
        except (TypeError, ValueError):
            negative = False
        if negative:
            issues.append(_compatibility_issue(
                "observation_not_nonnegative",
                f"{algorithm} requires nonnegative observations; minimum is {observation_min}",
                details={"observation_min": observation_min},
            ))
    if observation_finite is False:
        issues.append(_compatibility_issue(
            "observation_not_finite",
            f"solver {algorithm!r} cannot run with non-finite observations",
        ))
    if algorithm in {"mlem", "osem"} and (
        observation_domain is not None or observation_model is not None
    ) and (
        observation_domain not in {"nonnegative_counts", "intensity"}
        or observation_model not in EMISSION_OBSERVATION_MODELS
    ):
        issues.append(_compatibility_issue(
            "emission_observation_model_required",
            f"{algorithm} requires an explicit Poisson emission/count observation model; "
            f"got domain={observation_domain!r}, model={observation_model!r}",
            remediation="Use nonnegative count/intensity data and one of the registered emission observation models.",
        ))
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for issue in issues:
        identity = (str(issue["code"]), str(issue["message"]))
        if identity not in seen:
            unique.append(issue)
            seen.add(identity)
    return unique


def validate_compatibility(
    algorithm: str,
    *,
    geometry_type: str | None = None,
    dimension: int | None = None,
    observation_domain: str | None = None,
    regularizer: str | None = None,
    observation_model: str | None = None,
    observation_min: float | None = None,
    observation_finite: bool | None = None,
) -> list[str]:
    """Return compatibility error messages (legacy API) in stable order."""

    return [
        str(issue["message"])
        for issue in compatibility_diagnostics(
            algorithm,
            geometry_type=geometry_type,
            dimension=dimension,
            observation_domain=observation_domain,
            regularizer=regularizer,
            observation_model=observation_model,
            observation_min=observation_min,
            observation_finite=observation_finite,
        )
    ]


validate_compatibility_details = compatibility_diagnostics


def validate_parameter_values(
    algorithm: str,
    parameters: Mapping[str, Any] | None = None,
    *,
    views: int | None = None,
    geometry_type: str | None = None,
    dimension: int | None = None,
    observation_domain: str | None = None,
    observation_model: str | None = None,
    observation_min: float | None = None,
    observation_finite: bool | None = None,
    estimated_lipschitz: float | None = None,
    parameter_estimates: Mapping[str, Any] | None = None,
    parameter_sources: Mapping[str, str] | None = None,
) -> ParameterValidationResult:
    """Normalize and validate parameters using registry metadata.

    ``estimated_lipschitz`` is the squared operator norm ``||A||^2``.  The
    runtime supplies it after building a public-data operator; callers doing
    request-only validation can omit it and receive a warning for step-size
    stability rather than a fabricated estimate.
    """

    if algorithm not in SOLVER_SPECS:
        return ParameterValidationResult(
            algorithm,
            {},
            (f"unknown solver: {algorithm!r}",),
            reason_codes=("unknown_algorithm",),
        )
    spec = SOLVER_SPECS[algorithm]
    incoming = dict(parameters or {})
    allowed = set(spec.parameter_names)
    errors: list[str] = []
    reason_codes: list[str] = []

    def add_error(message: str, code: str = "parameter_invalid") -> None:
        errors.append(message)
        reason_codes.append(code)

    for name in sorted(set(incoming) - allowed):
        add_error(f"unknown parameter {name!r} for {algorithm}", "parameter_unknown")
    normalized: dict[str, Any] = {}
    warnings: list[str] = []
    estimates: dict[str, Any] = dict(parameter_estimates or {})
    sources: dict[str, str] = {str(key): str(value) for key, value in (parameter_sources or {}).items()}

    for parameter in spec.parameters:
        if parameter.name in incoming:
            value = incoming[parameter.name]
        elif parameter.default is not None or parameter.nullable:
            value = parameter.default
        elif parameter.required:
            add_error(f"parameter {parameter.name} is required")
            continue
        else:
            continue
        try:
            normalized[parameter.name] = parameter.normalize(value)
            sources.setdefault(parameter.name, "user" if parameter.name in incoming else "registry_default")
        except ValueError as error:
            add_error(str(error))

    minimum = normalized.get("min_value")
    maximum = normalized.get("max_value")
    if minimum is not None and maximum is not None and minimum > maximum:
        add_error("min_value must be less than or equal to max_value", "parameter_constraint_violation")

    if views is not None:
        views = int(views)
        if views <= 0:
            add_error("number of views must be positive", "parameter_constraint_violation")
        for name in ("block_size",):
            value = normalized.get(name)
            if value is not None and views > 0 and value > views:
                add_error(
                    f"{name} must be no greater than the number of views ({views})",
                    "parameter_constraint_violation",
                )
        subset_count = normalized.get("subset_count")
        if subset_count is not None and views > 0 and subset_count > views:
            add_error(
                f"subset_count must be between 1 and the number of views ({views})",
                "parameter_constraint_violation",
            )

    compatibility = compatibility_diagnostics(
        algorithm,
        geometry_type=geometry_type,
        dimension=dimension,
        observation_domain=observation_domain,
        observation_model=observation_model,
        observation_min=observation_min,
        observation_finite=observation_finite,
    )
    for issue in compatibility:
        add_error(str(issue["message"]), str(issue["code"]))

    if algorithm in {"landweber", "tv_fista"}:
        step = normalized.get("step_size")
        if estimated_lipschitz is None:
            if step is None:
                warnings.append(
                    f"{algorithm} step_size will be derived after operator construction; "
                    "request-only validation cannot prove the spectral bound"
                )
            else:
                warnings.append(
                    f"{algorithm} step_size stability requires ||A||^2; "
                    "no operator estimate was supplied"
                )
        else:
            lipschitz = float(estimated_lipschitz)
            if lipschitz <= 0:
                add_error("estimated ||A||^2 must be positive", "parameter_constraint_violation")
            elif step is None:
                # Use the conservative policy shared with the Agent-side
                # selector: alpha=0.9/L for Landweber (well inside the
                # theoretical 0<alpha<2/L interval), and 0.99/L for FISTA.
                normalized["step_size"] = (
                    0.9 / lipschitz if algorithm == "landweber" else 0.99 / lipschitz
                )
                sources["step_size"] = "power_iteration"
                estimates.update({"operator_norm_squared": lipschitz, "step_size_source": "power_iteration"})
            else:
                upper = (2.0 / lipschitz) if algorithm == "landweber" else (1.0 / lipschitz)
                if algorithm == "landweber" and not (0.0 < step < upper):
                    add_error(
                        f"step_size must satisfy 0 < step_size < 2 / ||A||^2 ({upper:.6g})",
                        "parameter_constraint_violation",
                    )
                if algorithm == "tv_fista" and not (0.0 < step <= upper):
                    add_error(
                        f"step_size must satisfy 0 < step_size <= 1 / ||A||^2 ({upper:.6g})",
                        "parameter_constraint_violation",
                    )
                estimates.update({"operator_norm_squared": lipschitz, "step_size_source": "user"})

    if algorithm in {"sart", "os_sart"}:
        relaxation = normalized.get("relaxation")
        if relaxation is not None and not (0.0 < relaxation <= 1.0):
            add_error("relaxation must satisfy 0 < relaxation <= 1", "parameter_constraint_violation")
        if algorithm == "os_sart" and normalized.get("subset_count") is None and normalized.get("block_size") is None:
            warnings.append("os_sart will use a backend-derived block_size because neither subset_count nor block_size was supplied")

    if algorithm == "tikhonov" and normalized.get("reg_strength") == 0.0:
        warnings.append("reg_strength=0 leaves only the positive-semidefinite normal equation; CG may break down")

    return ParameterValidationResult(
        algorithm=algorithm,
        parameters=normalized,
        errors=tuple(dict.fromkeys(errors)),
        warnings=tuple(dict.fromkeys(warnings)),
        estimates=estimates,
        sources=sources,
        reason_codes=tuple(dict.fromkeys(reason_codes)),
    )


# Fail closed if a registry author introduces a duplicate, non-canonical, or
# non-serializable ordinary-CT entry.  This is intentionally limited to
# metadata and does not construct or execute any numerical solver.
validate_registry()

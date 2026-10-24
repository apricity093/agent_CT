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
from math import ceil, isfinite
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
    "fista",
    "admm",
    "pdhg",
    "constrained_pdhg",
    "asd_pocs_tv",
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
    "transmission_poisson", "transmission_poisson_gaussian", "gaussian",
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
    "dimension_invalid": {
        "severity": "error",
        "description": "A supplied reconstruction dimension must be a strictly positive integer.",
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
    "parameter_type_invalid": {
        "severity": "error",
        "description": "A parameter has a type incompatible with its registry declaration.",
    },
    "parameter_nonfinite": {
        "severity": "error",
        "description": "A numeric parameter or observed bound is not finite.",
    },
    "parameter_out_of_range": {
        "severity": "error",
        "description": "A parameter is outside its declared hard range.",
    },
    "parameter_choice_invalid": {
        "severity": "error",
        "description": "A categorical parameter is not one of its registered choices.",
    },
    "views_invalid": {
        "severity": "error",
        "description": "The number of projection views must be a positive integer.",
    },
    "block_size_exceeds_views": {
        "severity": "error",
        "description": "A row-action block cannot contain more views than the case provides.",
    },
    "subset_count_exceeds_views": {
        "severity": "error",
        "description": "The requested subset count cannot exceed the number of views.",
    },
    "subset_partition_mismatch": {
        "severity": "error",
        "description": "block_size and subset_count describe conflicting view partitions.",
    },
    "spectral_estimate_invalid": {
        "severity": "error",
        "description": "The supplied squared operator-norm estimate is not finite and positive.",
    },
    "spectral_estimate_required": {
        "severity": "error",
        "description": "A theorem-constrained step size requires a usable operator-norm estimate.",
    },
    "spectral_step_invalid": {
        "severity": "error",
        "description": "The step size violates the solver's theorem-based spectral bound.",
    },
    "backend_device_mismatch": {
        "severity": "error",
        "description": "The requested device does not satisfy the algorithm backend contract.",
    },
    "geometry_shape_invalid": {
        "severity": "error",
        "description": "The problem geometry shape violates the algorithm contract.",
    },
    "budget_invalid": {
        "severity": "error",
        "description": "A compute budget limit must be a finite nonnegative integer.",
    },
    "iteration_budget_exceeded": {
        "severity": "error",
        "description": "The configured iteration count exceeds the admitted budget.",
    },
    "forward_call_budget_exceeded": {
        "severity": "error",
        "description": "The conservative forward-operator call estimate exceeds the budget.",
    },
    "adjoint_call_budget_exceeded": {
        "severity": "error",
        "description": "The conservative adjoint-operator call estimate exceeds the budget.",
    },
    "memory_budget_exceeded": {
        "severity": "error",
        "description": "The estimated memory requirement exceeds the budget.",
    },
    "parameter_source_unknown": {
        "severity": "warning",
        "description": "A parameter provenance label is outside the registered vocabulary.",
    },
    "spectral_bound_unavailable": {
        "severity": "warning",
        "description": "Request-only validation cannot prove a spectral step-size bound without an operator estimate.",
    },
    "operator_call_budget_unchecked": {
        "severity": "warning",
        "description": "A call budget was supplied without a conservative call estimate.",
    },
}
CT_COMPATIBILITY_REASON_CODES = COMPATIBILITY_REASON_CODES

PARAMETER_SOURCE_VOCABULARY: tuple[str, ...] = (
    "registry_default",
    "repository_config@sha256",
    "problem_metadata_rule",
    "problem_constraint",
    "power_iteration",
    "validation_search",
    "user_override",
    "recovery_attempt",
)

PARAMETER_VALIDATION_REASON_CODES: tuple[str, ...] = (
    "parameter_unknown",
    "parameter_invalid",
    "parameter_constraint_violation",
    "parameter_type_invalid",
    "parameter_nonfinite",
    "parameter_out_of_range",
    "parameter_choice_invalid",
    "dimension_invalid",
    "views_invalid",
    "block_size_exceeds_views",
    "subset_count_exceeds_views",
    "subset_partition_mismatch",
    "spectral_estimate_invalid",
    "spectral_estimate_required",
    "spectral_step_invalid",
    "backend_device_mismatch",
    "geometry_shape_invalid",
    "budget_invalid",
    "iteration_budget_exceeded",
    "forward_call_budget_exceeded",
    "adjoint_call_budget_exceeded",
    "memory_budget_exceeded",
    "parameter_source_unknown",
    "spectral_bound_unavailable",
    "operator_call_budget_unchecked",
)

_ITERATIVE_ALGORITHM_IDS: tuple[str, ...] = tuple(
    name for name in CANONICAL_ALGORITHM_IDS if name not in {"fbp", "fdk"}
)

# These are metadata categories, not additional runtime parameters.  The
# explicit ``not_applicable`` entries prevent an Agent from inventing ADMM,
# primal/dual, or free-momentum values for the current twelve algorithms.
PARAMETER_APPLICABILITY: dict[str, dict[str, Any]] = {
    "regularization_strength": {
        "status": "applicable",
        "parameter_names": ("reg_strength", "strength"),
        "algorithms": ("tikhonov", "tv_fista", "fista"),
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
        "algorithms": ("landweber", "tv_fista", "fista"),
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
        "status": "applicable", "parameter_names": ("rho",), "algorithms": ("admm",),
        "description": "ADMM rho; validated by solver.",
    },
    "primal_step": {
        "status": "applicable", "parameter_names": ("tau",), "algorithms": ("pdhg",),
        "description": "PDHG tau; validated by solver.",
    },
    "dual_step": {
        "status": "applicable", "parameter_names": ("sigma",), "algorithms": ("pdhg",),
        "description": "PDHG sigma; validated by solver.",
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
        "internal_sequence_algorithms": ("tv_fista", "fista"),
        "reason_code": "free_momentum_not_applicable",
        "description": "TV-FISTA uses its deterministic Nesterov sequence; free momentum is not tunable.",
    },
    "free_momentum": {
        "status": "not_applicable",
        "parameter_names": (),
        "algorithms": (),
        "internal_sequence_algorithms": ("tv_fista", "fista"),
        "reason_code": "free_momentum_not_applicable",
        "description": "The current registry has no free momentum parameter.",
    },
}
NON_APPLICABLE_PARAMETER_CATEGORIES: tuple[str, ...] = (
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
    validation_only_search_space: tuple[Any, ...] = ()

    @property
    def value_type(self) -> str:
        """Alias useful to callers that avoid the built-in name ``type``."""

        return self.type

    def normalize(self, value: Any) -> Any:
        if value is None:
            if self.nullable:
                return None
            if not self.required:
                raise ValueError(f"parameter {self.name} cannot be null")
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
        record["validation_only_search_space"] = list(self.validation_only_search_space)
        record["provenance_vocabulary"] = list(PARAMETER_SOURCE_VOCABULARY)
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
    full_proximal: str = "unavailable"
    supported_solvers: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    shape_requirements: tuple[str, ...] = ()
    transform: str = "identity"
    coefficient_proximal: str = "unavailable"

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
        record["parameter_validation"] = _jsonable(
            dict(self.metadata.get("parameter_validation", {}) or {})
        )
        record["parameter_aliases"] = _jsonable(
            dict(self.metadata.get("parameter_aliases", {}) or {})
        )
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
    warning_codes: tuple[str, ...] = ()

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
            "warning_codes": list(self.warning_codes),
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
        has_proximal=True,
        has_gradient=True,
        convex=True,
        differentiable=True,
        parameter_names=("reg_strength",),
        parameter_sensitivity="low_to_medium",
        description="Half squared norm; full proximal only for identity L.",
        full_proximal="identity_L_only_closed_form", supported_solvers=("tikhonov", "fista", "admm", "pdhg"),
        constraints=("fista: optional box via closed form", "admm/pdhg: unconstrained or nonnegative"), coefficient_proximal="quadratic_closed_form",
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
        description="Convex 2D TV; finite FGP approximate proximal, no gap certificate.",
        full_proximal="finite_FGP_no_gap_certificate", supported_solvers=("tv_fista", "fista", "admm", "pdhg", "constrained_pdhg", "asd_pocs_tv"),
        constraints=("fista: unconstrained", "tv_fista: legacy post-prox box projection", "admm/pdhg: unconstrained or nonnegative"), transform="D", coefficient_proximal="isotropic_vector_shrink_or_anisotropic_soft_threshold",
    ),
    "h1": RegularizerSpec("h1", "Gradient quadratic H1", "quadratic_gradient",
        False, True, True, True, ("strength",), description="0.5||Dx||^2; unit-grid forward differences, zero terminal entries.",
        supported_solvers=("tikhonov", "admm", "pdhg"), constraints=("unconstrained", "admm/pdhg: nonnegative"), transform="D", coefficient_proximal="quadratic_closed_form"),
    "huber": RegularizerSpec("huber", "Componentwise gradient Huber", "gradient_huber",
        False, True, True, True, ("strength", "delta"), description="Componentwise h_delta(Dx); delta in unit-grid difference units; no full proximal.",
        supported_solvers=("admm", "pdhg"), constraints=("unconstrained", "nonnegative"), transform="D", coefficient_proximal="componentwise_huber_closed_form"),
    "haar_l1": RegularizerSpec("haar_l1", "Haar detail L1", "wavelet_l1",
        True, False, True, False, ("strength", "level"),
        description="Periodic even-origin orthonormal Haar; LL unpenalized, unit detail weights.",
        full_proximal="orthogonal_analysis_soft_threshold_synthesis", supported_solvers=("fista", "admm", "pdhg"),
        constraints=("fista: unconstrained", "admm/pdhg: unconstrained or nonnegative"), transform="full_orthogonal_Haar", coefficient_proximal="weighted_soft_threshold_LL_zero", shape_requirements=("level positive integer", "H,W divisible by 2**level")),
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
    pairing_policy = "fixed" if regularizer_pairing is not None else "selectable" if regularizers else "none"
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
    for code in PARAMETER_VALIDATION_REASON_CODES:
        if code not in reason_codes:
            reason_codes.append(code)
    fidelity_requirements={}
    if name in {'tikhonov','fista','admm','pdhg'}:
        fidelity_requirements={
            'l2':{'required_fields':('y_observed',),'approximation':False},
            'gaussian_nll':{'required_fields':('y_observed','sigma'),'approximation':False},
            'wls':{'required_fields':('y_observed','weights'),'approximation':'weight_source_dependent'}}
        if name=='pdhg':fidelity_requirements['transmission_poisson_nll']={
            'required_fields':('counts','I0'),'noise_models':('poisson',),'background':0,
            'constraints':('nonnegative',),'assumptions':('finite_minimizer_exists',),
            'failure_modes':('zero_counts_noncoercive_prior_may_have_no_finite_minimizer',)}
    applicability = (
        dict(parameter_applicability)
        if parameter_applicability is not None
        else _parameter_applicability(name, parameter_names, direct=direct)
    )
    contract_metadata = {
        "fidelity_requirements":fidelity_requirements,
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
            if name in {"tv_fista", "fista"} else "not_applicable"
        ),
        "parameter_validation": {
            "source_vocabulary": PARAMETER_SOURCE_VOCABULARY,
            "finite_numeric_values_required": True,
            "cross_parameter_constraints": (
                "min_value <= max_value",
                "block_size <= number_of_views",
                "subset_count <= number_of_views",
            ),
            "spectral_estimator": (
                "deterministic_power_iteration_on_A_adjoint_A",
                "operator_calls_are_counted_separately_from_solver_calls",
            ),
            "iteration_budget_is_not_convergence": True,
        },
        "parameter_aliases": (
            {"learning_rate": "step_size"}
            if name == "landweber" else {"reg_strength": "strength"} if name in {"fista", "admm", "pdhg"} else {"strength": "reg_strength"} if name == "tikhonov" else {}
        ),
        "theoretical_constraints": {
            parameter.name: parameter.theoretical_constraint
            for parameter in parameters
            if parameter.theoretical_constraint
        },
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



def _constrained_spec(name):
    common = (
        _p('epsilon', required=True, minimum=0., units='line_integral_l2'),
        _p('epsilon_source', 'str', required=True),
        _p('num_iterations', 'int', default=200, minimum=1),
        _p('tolerance', default=1e-5, minimum=0., inclusive_minimum=False),
        _p('feasibility_tolerance', default=1e-5, minimum=0., inclusive_minimum=False),
        _p('tv_mode', 'str', default='isotropic', choices=('isotropic','anisotropic')),
        _p('min_value', default=0., choices=(0.,)),
        _p('max_value', default=None, nullable=True),
    )
    if name == 'asd_pocs_tv':
        common += (
            _p('relaxation', default=1., minimum=0., inclusive_minimum=False, maximum=1.),
            _p('relaxation_decay', default=.99, minimum=0., maximum=1., inclusive_minimum=False, inclusive_maximum=False),
            _p('tv_step', default=.2, minimum=0., inclusive_minimum=False),
            _p('tv_step_decay', default=.95, minimum=0., maximum=1., inclusive_minimum=False, inclusive_maximum=False),
            _p('tv_updates', 'int', default=20, minimum=1),
            _p('update_ratio', default=.95, minimum=0., inclusive_minimum=False),
            _p('block_size', 'int', default=1, minimum=1),
        )
    return _algorithm(name, name, (2,), ('parallel_2d',),
        'Nonnegative unwhitened residual-ball constrained TV.', common,
        family='constrained_variational', objective='residual_ball_constrained_tv',
        regularizers=('tv',), regularizer_pairing='tv', observation_domains=('line_integral',),
        noise_models=('none','gaussian_relative','poisson_log','gaussian','poisson','poisson_gaussian'),
        sparse_view=True, limited_angle=True, differentiable=False, requires_proximal=True,
        convergence_criteria=('feasible_and_primal_dual_fixed_point' if name=='constrained_pdhg' else 'feasible_iterate_stagnation_stalled',),
        failure_modes=('iteration_limit','operator_budget','numerical_failure','constraint_violation'),
        metadata={'constraints':('nonnegative',), 'epsilon_source':'required_explicit_nonempty',
                  'residual':'unwhitened_masked_l2', 'tv_grid':'unit_forward_neumann',
                  'batch_channels':(1,1), 'automatic_epsilon':False,
                  'variant_id':'residual_ball_pdhg_v1' if name=='constrained_pdhg' else 'sart2d_feasibility_backtracking_v1',
                  'global_optimality_claim': name=='constrained_pdhg',
                  'step_policy':'0.99/sqrt(norm_A_bound_squared+8)' if name=='constrained_pdhg' else 'normalized_SART_TV_backtracking'})

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
        _COMMON + (_p(
            "step_size", default=None, nullable=True, minimum=0.0,
            inclusive_minimum=False, tunable=True,
            theoretical_constraint="0 < alpha < 2 / ||A||^2",
            theoretical_derivation="For quadratic least squares, Landweber is stable when 0 < alpha < 2/L with L=||A||^2.",
            default_condition="derived from deterministic power iteration when omitted",
            units="inverse_operator_norm_squared",
            cost_effect="counted gradient, trial forwards and returned-point mapping",
            sensitivity="high",
        ),),
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
            _p("seed", "int", default=None, nullable=True, minimum=0),
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
            _p("seed", "int", default=None, nullable=True, minimum=0),
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
            _p("seed", "int", default=None, nullable=True, minimum=0),
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
            _p("data_fidelity", "str", default="l2", choices=("l2", "gaussian_nll", "wls")),
            _p("weight_source", "str", default=None, nullable=True, choices=("known_variance", "provided", "post_log_poisson", "post_log_poisson_gaussian")),
            _p("reg_strength", minimum=0.0, default=1e-2, tunable=True, scale="log"),
            _p("tolerance", minimum=0.0, default=1e-6, tunable=True),
            _p("regularizer", "str", default="tikhonov", choices=("tikhonov", "h1")),
            _p("strength", default=None, nullable=True, minimum=0.0),
        ), noise_models=("none","gaussian_relative","poisson_log","gaussian","poisson","poisson_gaussian"), family="variational", objective="quadratic_regularized_least_squares",
        regularizers=("tikhonov", "h1"), sparse_view=True, limited_angle=True,
        failure_modes=("normal_equation_breakdown", "over_smoothing"),
        convergence_criteria=("normalized_regularized_normal_residual", "data_residual", "relative_iterate_change"),
        metadata={"data_fidelity":("l2","gaussian_nll","wls"),"fidelity_constraints":{"gaussian_nll":("unconstrained",),"wls":("unconstrained",)}},
    ),
    "tv_fista": _algorithm(
        "tv_fista", "TV-FISTA", (2,), ("parallel_2d",), "TV-regularized FISTA.",
        _common_parameters(50) + (
            _p("reg_strength", minimum=0.0, default=1e-3, tunable=True, scale="log"),
            _p(
                "step_size", default=None, nullable=True, minimum=0.0,
                inclusive_minimum=False, tunable=True,
                theoretical_constraint="Trial step checked by smooth quadratic majorization with halving backtracking",
                theoretical_derivation="Finite power iteration estimates L; backtracking checks the actual smooth majorizer.",
                default_condition="0.99 / finite power estimate when omitted; checked by backtracking",
                units="inverse_operator_norm_squared",
                cost_effect="counted gradient, trial forwards and returned-point mapping",
                sensitivity="high",
            ),
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
    "fista": _algorithm(
        "fista", "FISTA", (2,), ("parallel_2d",), "Composite L2 FISTA with checked smooth majorization.",
        _common_parameters(50) + (
            _p("data_fidelity", "str", default="l2", choices=("l2", "gaussian_nll", "wls", "transmission_poisson_nll")),
            _p("weight_source", "str", default=None, nullable=True, choices=("known_variance", "provided", "post_log_poisson", "post_log_poisson_gaussian")),
            _p("nll_tolerance", default=None, nullable=True, minimum=0., inclusive_minimum=False),
            _p("nll_max_iterations", "int", default=200, minimum=1),
            _p("regularizer", "str", default="tikhonov", choices=("tikhonov", "tv", "haar_l1")),
            _p("strength", default=1e-2, minimum=0.0, tunable=True),
            _p("reg_strength", default=None, nullable=True, minimum=0.0),
            _p("step_size", default=None, nullable=True, minimum=0.0, inclusive_minimum=False,
               theoretical_constraint="Initial trial step; L2 quadratic majorizer checked with halving backtracking"),
            _p("tolerance", default=1e-5, minimum=0.0),
            _p("power_iterations", "int", default=12, minimum=1),
            _p("level", "int", default=2, minimum=1),
            _p("tv_mode", "str", default="isotropic", choices=("isotropic", "anisotropic")),
            _p("tv_num_iterations", "int", default=50, minimum=1),
            _p("tv_tolerance", default=1e-5, minimum=0.0),
        ), noise_models=("none","gaussian_relative","poisson_log","gaussian","poisson","poisson_gaussian"), family="variational", objective="composite_regularized_least_squares",
        regularizers=("tikhonov", "tv", "haar_l1"), requires_proximal=True,
        sparse_view=True, limited_angle=True,
        convergence_criteria=("prox_gradient_mapping", "relative_composite_objective_change"),
        failure_modes=("proximal_inexactness", "backtracking_limit", "over_smoothing"),
        metadata={"data_fidelity": ("l2","gaussian_nll","wls"), "norm_estimate_is_upper_bound": False,
                  "backtracking": "checked_smooth_quadratic_majorizer",
                  "proximal_accuracy": {"tv": "finite_FGP_no_gap_certificate", "tikhonov": "closed_form", "haar_l1": "closed_form"}}),
    "admm": _algorithm("admm", "ADMM", (2,), ("parallel_2d",), "L2 composite ADMM.",
        _common_parameters(200) + (
            _p("data_fidelity", "str", default="l2", choices=("l2", "gaussian_nll", "wls", "transmission_poisson_nll")),
            _p("weight_source", "str", default=None, nullable=True, choices=("known_variance", "provided", "post_log_poisson", "post_log_poisson_gaussian")),
            _p("nll_tolerance", default=None, nullable=True, minimum=0., inclusive_minimum=False),
            _p("nll_max_iterations", "int", default=200, minimum=1),
            _p("regularizer", "str", default="tikhonov", choices=("tikhonov", "tv", "h1", "huber", "haar_l1")),
            _p("strength", default=0.01, minimum=0.),
            _p("reg_strength", default=None, nullable=True, minimum=0.),
            _p("tolerance", default=1e-5, minimum=0.),
            _p("level", "int", default=2, minimum=1),
            _p("delta", default=0.01, minimum=0., inclusive_minimum=False),
            _p("tv_mode", "str", default="isotropic", choices=("isotropic", "anisotropic")),
            _p("rho", default=1., minimum=0., inclusive_minimum=False),
            _p("cg_max_iterations", "int", default=100, minimum=1),
            _p("cg_tolerance", default=1e-6, minimum=0., inclusive_minimum=False),
        ), noise_models=("none","gaussian_relative","poisson_log","gaussian","poisson","poisson_gaussian"), family="variational", objective="composite_regularized_least_squares",
        regularizers=("tikhonov", "tv", "h1", "huber", "haar_l1"), sparse_view=True, limited_angle=True,
        convergence_criteria=("split_residual",), failure_modes=("inner_solve_limit", "numerical_failure", "iteration_limit"),
        metadata={"data_fidelity": ("l2","gaussian_nll","wls"), "constraints":("unconstrained","nonnegative"), "theta":1}),
    "pdhg": _algorithm("pdhg", "PDHG", (2,), ("parallel_2d",), "L2 composite PDHG.",
        _common_parameters(200) + (
            _p("data_fidelity", "str", default="l2", choices=("l2", "gaussian_nll", "wls", "transmission_poisson_nll")),
            _p("weight_source", "str", default=None, nullable=True, choices=("known_variance", "provided", "post_log_poisson", "post_log_poisson_gaussian")),
            _p("nll_tolerance", default=None, nullable=True, minimum=0., inclusive_minimum=False),
            _p("nll_max_iterations", "int", default=200, minimum=1),
            _p("regularizer", "str", default="tikhonov", choices=("tikhonov", "tv", "h1", "huber", "haar_l1")),
            _p("strength", default=0.01, minimum=0.),
            _p("reg_strength", default=None, nullable=True, minimum=0.),
            _p("tolerance", default=1e-5, minimum=0.),
            _p("level", "int", default=2, minimum=1),
            _p("delta", default=0.01, minimum=0., inclusive_minimum=False),
            _p("tv_mode", "str", default="isotropic", choices=("isotropic", "anisotropic")),
            _p("tau", default=None, nullable=True, minimum=0., inclusive_minimum=False),
            _p("sigma", default=None, nullable=True, minimum=0., inclusive_minimum=False),
        ), noise_models=("none","gaussian_relative","poisson_log","gaussian","poisson","poisson_gaussian"), family="variational", objective="composite_regularized_least_squares",
        regularizers=("tikhonov", "tv", "h1", "huber", "haar_l1"), sparse_view=True, limited_angle=True,
        convergence_criteria=("split_residual",), failure_modes=("inner_solve_limit", "numerical_failure", "iteration_limit"),
        metadata={"data_fidelity_constraints":{"transmission_poisson_nll":("nonnegative",)}, "data_fidelity": ("l2","gaussian_nll","wls","transmission_poisson_nll"), "constraints":("unconstrained","nonnegative"), "theta":1}),
    "constrained_pdhg": _constrained_spec("constrained_pdhg"),
    "asd_pocs_tv": _constrained_spec("asd_pocs_tv"),
    "fdk": _algorithm(
        "fdk", "FDK", (3,), ("cone_3d",), "Cone-beam Feldkamp-Davis-Kress reconstruction.",
        (
            _p("filter_type", "str", default="ram-lak", choices=("ram-lak", "ramp", "ramlak")),
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
        for parameter in spec.parameters:
            if parameter.type not in {"int", "float", "bool", "str"}:
                raise ValueError(f"unsupported parameter type for {name!r}.{parameter.name}")
            if parameter.minimum is not None and parameter.maximum is not None:
                if parameter.minimum > parameter.maximum:
                    raise ValueError(f"invalid parameter range for {name!r}.{parameter.name}")
            if parameter.default is not None:
                try:
                    parameter.normalize(parameter.default)
                except ValueError as error:
                    raise ValueError(
                        f"invalid default for {name!r}.{parameter.name}: {error}"
                    ) from error
            if parameter.default is None and parameter.required and not parameter.nullable:
                # Required values are intentionally allowed to be supplied by
                # the problem/runtime; this branch only documents that the
                # registry does not silently invent a default for them.
                continue
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
        expected_pairing = {"tv_fista": "tv"}.get(name)
        if expected_pairing is not None and (
            spec.regularizer_pairing != expected_pairing
            or spec.regularizer_pairing_policy != "fixed"
        ):
            raise ValueError(f"{name!r} must preserve its fixed regularizer pairing")
        if name in {"tikhonov", "fista"} and (spec.regularizer_pairing_policy != "selectable" or not set(spec.regularizers) <= set(REGULARIZER_SPECS)):
            raise ValueError(f"invalid selectable regularizers for {name}")
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
    json.dumps(list(PARAMETER_SOURCE_VOCABULARY), sort_keys=True, allow_nan=False)
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
        "parameter_source_vocabulary": list(PARAMETER_SOURCE_VOCABULARY),
        "parameter_validation_reason_codes": list(PARAMETER_VALIDATION_REASON_CODES),
        "compatibility_reason_codes": _jsonable(COMPATIBILITY_REASON_CODES),
        "required_algorithm_fields": [
            "name", "display_name", "dimensions", "geometry_types", "parameter_names",
            "family", "objective", "observation_domains", "noise_models", "parameters",
            "canonical_id", "aliases", "fixed_regularizer", "regularizer_pairing",
            "regularizer_pairing_policy", "observation_models", "parameter_applicability",
            "compatibility_reason_codes", "required_metadata", "requirements",
            "parameter_specs", "parameter_validation",
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
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            issues.append(_compatibility_issue(
                "dimension_invalid",
                f"dimension must be a strictly positive integer, got {dimension!r}",
                details={"received": dimension},
            ))
            dimension_value = None
        else:
            dimension_value = int(dimension)
        if dimension_value is not None and dimension_value not in spec.dimensions:
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
    image_shape: Any | None = None,
    domain_shape: Any | None = None,
    device: str | None = None,
    backend_capabilities: Mapping[str, Any] | Any | None = None,
    max_iterations: int | None = None,
    max_forward_calls: int | None = None,
    max_adjoint_calls: int | None = None,
    expected_operator_calls: Mapping[str, Any] | None = None,
    max_memory_mb: int | None = None,
    estimated_memory_mb: float | None = None,
    budget: Mapping[str, Any] | None = None,
) -> ParameterValidationResult:
    """Normalize and validate parameters using registry metadata.

    ``estimated_lipschitz`` is the squared operator norm ``||A||^2``.  The
    runtime supplies it after building a public-data operator; callers doing
    request-only validation can omit it and receive a warning for step-size
    stability rather than a fabricated estimate.  ``expected_operator_calls``
    and ``estimated_memory_mb`` are conservative admission estimates supplied
    by a runtime/selector; this function never silently invents a cost model.
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
    warnings: list[str] = []
    warning_codes: list[str] = []

    def add_reason(code: str, *, warning: bool = False) -> None:
        target = warning_codes if warning else reason_codes
        if code not in target:
            target.append(code)

    def add_error(message: str, code: str = "parameter_invalid") -> None:
        errors.append(message)
        add_reason(code)

    def add_warning(message: str, code: str = "parameter_invalid") -> None:
        warnings.append(message)
        add_reason(code, warning=True)

    # ``learning_rate`` is a declared semantic alias, not a second free
    # parameter.  Normalize it before unknown-parameter handling so the
    # solver and provenance records always expose the canonical name.
    canonical_incoming = dict(incoming)
    alias_sources = dict(parameter_sources or {})
    if "learning_rate" in canonical_incoming and algorithm == "landweber":
        if "step_size" in canonical_incoming:
            add_error(
                "learning_rate and step_size cannot both be supplied",
                "parameter_constraint_violation",
            )
        else:
            canonical_incoming["step_size"] = canonical_incoming.pop("learning_rate")
            if "learning_rate" in alias_sources and "step_size" not in alias_sources:
                alias_sources["step_size"] = alias_sources["learning_rate"]

    if algorithm in {"fista", "tikhonov", "admm", "pdhg"}:
        if incoming.get("strength") is not None and incoming.get("reg_strength") is not None and incoming["strength"] != incoming["reg_strength"]:
            add_error("strength and reg_strength conflict", "parameter_constraint_violation")
        if algorithm in {"fista", "admm", "pdhg"} and incoming.get("reg_strength") is not None:
            canonical_incoming["strength"] = incoming["reg_strength"]
            alias_sources["strength"] = alias_sources.get("reg_strength", "user_override")
            canonical_incoming.pop("reg_strength", None)
        if algorithm == "tikhonov" and incoming.get("strength") is not None:
            canonical_incoming["reg_strength"] = incoming["strength"]
            alias_sources["reg_strength"] = alias_sources.get("strength", "user_override")
            canonical_incoming.pop("strength", None)

    non_applicable_names: dict[str, str] = {}
    for category, descriptor in PARAMETER_APPLICABILITY.items():
        status = str(descriptor.get("status", ""))
        if status in {"not_applicable", "policy_controlled"} or algorithm not in descriptor.get("algorithms", ()):
            reason = str(descriptor.get("reason_code", "parameter_not_applicable"))
            for name in descriptor.get("parameter_names", ()):
                non_applicable_names[str(name)] = reason
            # The categories themselves are also useful diagnostics when an
            # Agent sends metadata vocabulary as if it were a solver field.
            non_applicable_names.setdefault(category, reason)

    for name in sorted(set(canonical_incoming) - allowed):
        code = non_applicable_names.get(name, "parameter_unknown")
        if code == "parameter_not_applicable":
            message = f"parameter {name!r} is not applicable to {algorithm}"
        elif code == "free_momentum_not_applicable":
            message = f"parameter {name!r} is not a free momentum parameter for {algorithm}"
        else:
            message = f"unknown parameter {name!r} for {algorithm}"
        add_error(message, code)

    normalized: dict[str, Any] = {}
    estimates: dict[str, Any] = dict(parameter_estimates or {})
    sources: dict[str, str] = {}

    def source_for(name: str, supplied: bool) -> str:
        value = alias_sources.get(name)
        source = str(value) if value is not None else (
            "user_override" if supplied else "registry_default"
        )
        if source not in PARAMETER_SOURCE_VOCABULARY:
            add_warning(
                f"parameter {name} uses unregistered provenance source {source!r}",
                "parameter_source_unknown",
            )
        return source

    def parameter_error_code(error: ValueError) -> str:
        message = str(error).lower()
        if "finite" in message:
            return "parameter_nonfinite"
        if "one of" in message:
            return "parameter_choice_invalid"
        if "null" in message or "required" in message:
            return "parameter_type_invalid"
        if any(token in message for token in ("integer", "numeric", "bool", "string")):
            return "parameter_type_invalid"
        return "parameter_out_of_range"

    for parameter in spec.parameters:
        supplied = parameter.name in canonical_incoming
        if supplied:
            value = canonical_incoming[parameter.name]
        elif parameter.default is not None or parameter.nullable:
            value = parameter.default
        elif parameter.required:
            add_error(f"parameter {parameter.name} is required", "parameter_invalid")
            continue
        else:
            continue
        try:
            normalized[parameter.name] = parameter.normalize(value)
            sources[parameter.name] = source_for(parameter.name, supplied)
        except ValueError as error:
            add_error(str(error), parameter_error_code(error))

    minimum = normalized.get("min_value")
    maximum = normalized.get("max_value")
    if minimum is not None and maximum is not None and minimum > maximum:
        add_error("min_value must be less than or equal to max_value", "parameter_constraint_violation")

    view_count: int | None = None
    if views is not None:
        if isinstance(views, bool) or not isinstance(views, int):
            add_error("number of views must be a positive integer", "views_invalid")
        else:
            view_count = int(views)
            if view_count <= 0:
                add_error("number of views must be positive", "views_invalid")
        for name in ("block_size",):
            value = normalized.get(name)
            if value is not None and view_count is not None and view_count > 0 and value > view_count:
                add_error(
                    f"{name} must be no greater than the number of views ({view_count})",
                    "block_size_exceeds_views",
                )
        subset_count = normalized.get("subset_count")
        if subset_count is not None and view_count is not None and view_count > 0 and subset_count > view_count:
            add_error(
                f"subset_count must be between 1 and the number of views ({view_count})",
                "subset_count_exceeds_views",
            )
        block_size = normalized.get("block_size")
        if (
            algorithm in {"os_sart", "osem"}
            and view_count is not None
            and view_count > 0
            and block_size is not None
            and subset_count is not None
            and ceil(view_count / block_size) != subset_count
        ):
            add_error(
                "block_size and subset_count must describe the same balanced view partition",
                "subset_partition_mismatch",
            )

    if algorithm in {'constrained_pdhg', 'asd_pocs_tv'}:
        if not str(normalized.get('epsilon_source', '')).strip():
            add_error('epsilon_source must be explicit', 'parameter_constraint_violation')
        if normalized.get('min_value') != 0 or normalized.get('max_value') is not None:
            add_error('constrained TV requires x >= 0 without upper bound', 'parameter_constraint_violation')
        for shape in (image_shape, domain_shape):
            if shape is not None and (len(shape) != 3 or shape[0] != 1):
                add_error('constrained TV requires one 2D channel', 'geometry_shape_invalid')

    if algorithm in {"admm", "pdhg"}:
        if normalized.get("min_value") not in (None, 0) or normalized.get("max_value") is not None:
            add_error("only unconstrained and nonnegative constraints supported", "parameter_constraint_violation")
    if algorithm=='tikhonov' and normalized.get('data_fidelity','l2')!='l2' and (normalized.get('min_value') is not None or normalized.get('max_value') is not None):
        add_error('weighted Tikhonov CG supports unconstrained reconstruction only','parameter_constraint_violation')
    if algorithm=='pdhg' and normalized.get('data_fidelity')=='transmission_poisson_nll' and normalized.get('min_value')!=0:
        add_error('P3 transmission_poisson_nll requires min_value=0', 'parameter_constraint_violation')
    if algorithm in {"fista", "admm"} and normalized.get("data_fidelity") == "transmission_poisson_nll":
        add_error("transmission_poisson_nll requires PDHG", "parameter_constraint_violation")
    if algorithm in {"fista", "admm", "pdhg"} and normalized.get("data_fidelity") == "wls" and normalized.get("weight_source") is None:
        add_error("wls requires explicit weight_source", "parameter_constraint_violation")
    if algorithm in {"fista", "tikhonov"}:
        prior = normalized.get("regularizer", "tikhonov")
        bounded = normalized.get("min_value") is not None or normalized.get("max_value") is not None
        if bounded and ((algorithm == "fista" and prior in {"tv", "haar_l1"}) or prior == "h1"):
            add_error(f"{algorithm}+{prior} requires unconstrained reconstruction", "parameter_constraint_violation")
        if prior == "haar_l1" and "level" in normalized:
            for shape in (image_shape, domain_shape):
                if shape is not None:
                    from ..operators.wavelet import validate_haar
                    try:
                        validate_haar(shape, normalized["level"])
                    except ValueError as error:
                        add_error(str(error), "geometry_shape_invalid")

    # Validate observed bounds before passing them to the compatibility gate;
    # NaN must never be treated as a harmless lower bound.
    normalized_observation_min: float | None = None
    if observation_min is not None:
        try:
            if isinstance(observation_min, bool):
                raise ValueError
            normalized_observation_min = float(observation_min)
        except (TypeError, ValueError):
            add_error("observation_min must be finite numeric data", "parameter_nonfinite")
        else:
            if not isfinite(normalized_observation_min):
                add_error("observation_min must be finite", "parameter_nonfinite")
                observation_finite = False

    compatibility = compatibility_diagnostics(
        algorithm,
        geometry_type=geometry_type,
        dimension=dimension,
        observation_domain=observation_domain,
        observation_model=observation_model,
        observation_min=normalized_observation_min,
        observation_finite=observation_finite,
    )
    for issue in compatibility:
        add_error(str(issue["message"]), str(issue["code"]))

    # Geometry and backend checks are conditional on context.  Request-only
    # config validation stays usable without a case, while a runtime that has
    # supplied the facts gets a hard preflight rejection.
    if algorithm == "fdk":
        if device is not None and not str(device).lower().startswith("cuda"):
            add_error("fdk requires a CUDA device", "backend_device_mismatch")
        for shape_name, shape in (
            ("domain_shape", domain_shape),
            ("image_shape", image_shape),
        ):
            if shape is None:
                continue
            try:
                shape_tuple = tuple(shape)
            except (TypeError, ValueError):
                shape_tuple = ()
            shape_is_positive_integer = (
                len(shape_tuple) == 3
                and all(
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and value > 0
                    for value in shape_tuple
                )
            )
            if not shape_is_positive_integer:
                add_error(
                    f"fdk {shape_name} must contain exactly three strictly positive integers",
                    "geometry_shape_invalid",
                )
            elif len(set(shape_tuple)) != 1:
                add_error(f"fdk {shape_name} must be cubic", "cubic_volume_required")
        if backend_capabilities:
            capability_names: set[str] = set()
            missing_cuda = False
            missing_astra = False
            if isinstance(backend_capabilities, Mapping):
                for key, value in backend_capabilities.items():
                    normalized_key = str(key).lower().replace("-", "_").replace(" ", "_")
                    if not bool(value):
                        if normalized_key in {"cuda", "cuda_device", "cuda_available"}:
                            missing_cuda = True
                        if normalized_key in {"astra", "astra_cuda", "astra_cuda_available"}:
                            missing_astra = True
                    elif bool(value):
                        capability_names.add(normalized_key)
            elif isinstance(backend_capabilities, str):
                capability_names = {
                    backend_capabilities.lower().replace("-", "_").replace(" ", "_")
                }
            else:
                capability_names = {
                    str(value).lower().replace("-", "_").replace(" ", "_")
                    for value in backend_capabilities
                }
            has_cuda = any(name in capability_names for name in {"cuda", "cuda_device", "cuda_available", "astra_cuda"})
            has_astra = any(name in capability_names for name in {"astra", "astra_cuda", "astra_cuda_available"})
            if missing_cuda or not has_cuda:
                add_error("fdk requires CUDA backend capability", "cuda_device_required")
            if missing_astra or not has_astra:
                add_error("fdk requires ASTRA CUDA backend capability", "astra_cuda_required")

    # A caller may pass either explicit limits or a ComputeBudget-like
    # mapping.  Keep the scalar API convenient for the runtime while making
    # all budget type/range failures deterministic and pre-loop.
    budget_values = dict(budget or {})

    def budget_limit(explicit: Any, key: str) -> Any:
        return explicit if explicit is not None else budget_values.get(key)

    validated_limits: dict[str, int | None] = {}
    for key, explicit in (
        ("max_iterations", max_iterations),
        ("max_forward_calls", max_forward_calls),
        ("max_adjoint_calls", max_adjoint_calls),
        ("max_memory_mb", max_memory_mb),
    ):
        value = budget_limit(explicit, key)
        if value is None:
            validated_limits[key] = None
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            add_error(f"{key} must be a finite nonnegative integer", "budget_invalid")
            validated_limits[key] = None
        else:
            validated_limits[key] = int(value)

    requested_iterations = normalized.get("num_iterations")
    iteration_limit = validated_limits["max_iterations"]
    if (
        iteration_limit is not None
        and algorithm not in {"fbp", "fdk"}
        and isinstance(requested_iterations, int)
        and requested_iterations > iteration_limit
    ):
        add_error(
            f"config requests {requested_iterations} iterations; budget allows {iteration_limit}",
            "iteration_budget_exceeded",
        )

    expected = dict(expected_operator_calls or {})
    for key, reason in (
        ("forward", "forward_call_budget_exceeded"),
        ("adjoint", "adjoint_call_budget_exceeded"),
    ):
        limit = validated_limits[f"max_{key}_calls"]
        if limit is None:
            continue
        value = expected.get(key)
        if value is None:
            if limit == 0 and algorithm != "fdk":
                add_error(
                    f"{algorithm} requires at least one {key} operator call",
                    reason,
                )
            else:
                add_warning(
                    f"{key} call budget cannot be checked without a conservative estimate",
                    "operator_call_budget_unchecked",
                )
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            add_error(f"expected {key} operator calls must be a nonnegative integer", "budget_invalid")
        elif value > limit:
            add_error(
                f"expected {key} operator calls ({value}) exceed budget ({limit})",
                reason,
            )

    memory_limit = validated_limits["max_memory_mb"]
    if memory_limit is not None and estimated_memory_mb is not None:
        try:
            memory_estimate = float(estimated_memory_mb)
        except (TypeError, ValueError):
            memory_estimate = float("nan")
        if not isfinite(memory_estimate) or memory_estimate < 0.0:
            add_error("estimated memory must be finite and nonnegative", "budget_invalid")
        elif memory_estimate > memory_limit:
            add_error(
                f"estimated memory ({memory_estimate:.6g} MB) exceeds budget ({memory_limit} MB)",
                "memory_budget_exceeded",
            )

    if algorithm == "landweber":
        step = normalized.get("step_size")
        if estimated_lipschitz is None and estimates.get("operator_norm_squared") is not None:
            estimated_lipschitz = estimates.get("operator_norm_squared")
        if estimated_lipschitz is None:
            if estimates.get("operator_norm_error"):
                add_error(
                    "operator-norm estimation failed; theorem-constrained step_size cannot be validated",
                    "spectral_estimate_required",
                )
            elif step is None:
                add_warning(
                    f"{algorithm} step_size will be derived after operator construction; "
                    "request-only validation cannot prove the spectral bound",
                    "spectral_bound_unavailable",
                )
            else:
                add_warning(
                    f"{algorithm} step_size stability requires ||A||^2; "
                    "no operator estimate was supplied",
                    "spectral_bound_unavailable",
                )
        else:
            try:
                if isinstance(estimated_lipschitz, bool):
                    raise ValueError
                lipschitz = float(estimated_lipschitz)
            except (TypeError, ValueError):
                lipschitz = float("nan")
            if not isfinite(lipschitz) or lipschitz <= 0:
                add_error("estimated ||A||^2 must be finite and positive", "spectral_estimate_invalid")
            elif step is None:
                # Use the conservative policy shared with the Agent-side
                # selector: alpha=0.9/L for Landweber (well inside the
                # theoretical 0<alpha<2/L interval), and 0.99/L for FISTA.
                safety_factor = 0.9 if algorithm == "landweber" else 0.99
                normalized["step_size"] = safety_factor / lipschitz
                sources["step_size"] = "power_iteration"
                estimates.update({
                    "operator_norm_squared": lipschitz,
                    "step_size_source": "power_iteration",
                    "step_size_safety_factor": safety_factor,
                })
            else:
                upper = (2.0 / lipschitz) if algorithm == "landweber" else (1.0 / lipschitz)
                if algorithm == "landweber" and not (0.0 < step < upper):
                    add_error(
                        f"step_size must satisfy 0 < step_size < 2 / ||A||^2 ({upper:.6g})",
                        "spectral_step_invalid",
                    )
                if algorithm == "tv_fista" and not (0.0 < step <= upper):
                    add_error(
                        f"step_size must satisfy 0 < step_size <= 1 / ||A||^2 ({upper:.6g})",
                        "spectral_step_invalid",
                    )
                estimates.update({
                    "operator_norm_squared": lipschitz,
                    "step_size_source": sources.get("step_size", "user_override"),
                })
            estimator_iterations = estimates.get("operator_norm_estimator_iterations")
            if estimator_iterations is not None:
                if (
                    isinstance(estimator_iterations, bool)
                    or not isinstance(estimator_iterations, int)
                    or estimator_iterations < 1
                ):
                    add_error(
                        "operator-norm estimator iterations must be a positive integer",
                        "parameter_invalid",
                    )
                    estimator_iterations = None
            if estimator_iterations is not None:
                estimates["spectral_estimator_contract"] = {
                    "method": "deterministic_power_iteration",
                    "iterations": int(estimator_iterations),
                    "initial_vector": "linspace(0.5, 1.5, prod(domain_shape))",
                    "operator": "A_adjoint_A",
                    "forward_calls": int(estimator_iterations) + 1,
                    "adjoint_calls": int(estimator_iterations) + 1,
                    "counted_separately": True,
                }

    if algorithm in {"sart", "os_sart"}:
        relaxation = normalized.get("relaxation")
        if relaxation is not None and not (0.0 < relaxation <= 1.0):
            add_error("relaxation must satisfy 0 < relaxation <= 1", "parameter_constraint_violation")
        if algorithm == "os_sart" and normalized.get("subset_count") is None and normalized.get("block_size") is None:
            add_warning(
                "os_sart will use a backend-derived block_size because neither subset_count nor block_size was supplied",
                "parameter_constraint_violation",
            )

    if algorithm == "osem" and normalized.get("subset_count") is None and normalized.get("block_size") is None:
        add_warning(
            "osem will use a backend-derived block_size because neither subset_count nor block_size was supplied",
            "parameter_constraint_violation",
        )

    if algorithm == "tikhonov" and normalized.get("reg_strength") == 0.0:
        add_warning(
            "reg_strength=0 leaves only the positive-semidefinite normal equation; CG may break down",
            "parameter_constraint_violation",
        )

    return ParameterValidationResult(
        algorithm=algorithm,
        parameters=normalized,
        errors=tuple(dict.fromkeys(errors)),
        warnings=tuple(dict.fromkeys(warnings)),
        estimates=estimates,
        sources=sources,
        reason_codes=tuple(dict.fromkeys(reason_codes)),
        warning_codes=tuple(dict.fromkeys(warning_codes)),
    )


# Fail closed if a registry author introduces a duplicate, non-canonical, or
# non-serializable ordinary-CT entry.  This is intentionally limited to
# metadata and does not construct or execute any numerical solver.
validate_registry()

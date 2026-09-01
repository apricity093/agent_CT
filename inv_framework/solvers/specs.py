"""Structured CT solver and regularizer metadata.

The registry in this module is the single source of truth for capabilities
that have to be visible to an orchestrator.  Numerical implementations keep
their small, stable ``solve(measurement, operator)`` interface; this module
describes the assumptions around that interface.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any, Mapping


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


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
        return _jsonable(asdict(self))


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

    def matches(
        self,
        *,
        solver: str,
        regularizer: str | None = None,
        geometry_type: str | None = None,
        observation_domain: str | None = None,
    ) -> bool:
        if solver != self.solver:
            return False
        if self.regularizer is not None and regularizer != self.regularizer:
            return False
        if self.geometry_types and geometry_type not in self.geometry_types:
            return False
        if self.observation_domains and observation_domain not in self.observation_domains:
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

    def parameter(self, name: str) -> ParameterSpec:
        for parameter in self.parameters:
            if parameter.name == name:
                return parameter
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        record = _jsonable(asdict(self))
        record["parameter_specs"] = [parameter.to_dict() for parameter in self.parameters]
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
) -> CTAlgorithmSpec:
    return CTAlgorithmSpec(
        name=name,
        display_name=display_name,
        dimensions=dimensions,
        geometry_types=geometry_types,
        parameter_names=tuple(parameter.name for parameter in parameters),
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
        convergence_criteria=("normal_residual", "data_residual", "relative_iterate_change"),
    ),
    "lsqr": _algorithm(
        "lsqr", "LSQR", (2,), ("parallel_2d",), "Golub-Kahan LSQR.",
        _common_parameters(10) + (
            _p("damping", minimum=0.0, default=0.0, tunable=True),
            _p("atol", minimum=0.0, default=1e-6, tunable=True),
            _p("btol", minimum=0.0, default=1e-6, tunable=True),
        ), sparse_view=True, limited_angle=True,
        failure_modes=("ill_conditioning", "premature_tolerance"),
        convergence_criteria=("data_residual", "relative_iterate_change"),
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
        compatibility_rules=(CompatibilityRule("mlem", observation_domains=("nonnegative_counts", "intensity"), requires_nonnegative_observation=True),),
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
        compatibility_rules=(CompatibilityRule("osem", observation_domains=("nonnegative_counts", "intensity"), requires_nonnegative_observation=True),),
    ),
    "tikhonov": _algorithm(
        "tikhonov", "Tikhonov", (2,), ("parallel_2d",), "Quadratic Tikhonov reconstruction.",
        _COMMON + (
            _p("reg_strength", minimum=0.0, default=1e-2, tunable=True, scale="log"),
            _p("tolerance", minimum=0.0, default=1e-6, tunable=True),
        ), family="variational", objective="quadratic_regularized_least_squares",
        regularizers=("tikhonov",), sparse_view=True, limited_angle=True,
        failure_modes=("normal_equation_breakdown", "over_smoothing"),
        convergence_criteria=("normal_residual", "data_residual", "relative_iterate_change"),
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
        convergence_criteria=("objective", "prox_mapping", "relative_iterate_change"),
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


def solver_records() -> list[dict[str, Any]]:
    """Return deterministic, JSON-compatible records for all CT solvers."""

    return [SOLVER_SPECS[name].to_dict() for name in SOLVER_SPECS]


def regularizer_records() -> list[dict[str, Any]]:
    return [spec.to_dict() for spec in REGULARIZER_SPECS.values()]


def validate_compatibility(
    algorithm: str,
    *,
    geometry_type: str | None = None,
    dimension: int | None = None,
    observation_domain: str | None = None,
    regularizer: str | None = None,
) -> list[str]:
    """Return compatibility errors without constructing a numerical solver."""

    if algorithm not in SOLVER_SPECS:
        return [f"unknown solver: {algorithm!r}"]
    spec = SOLVER_SPECS[algorithm]
    errors: list[str] = []
    if dimension is not None and int(dimension) not in spec.dimensions:
        errors.append(f"solver {algorithm!r} supports dimensions {spec.dimensions}, got {dimension}")
    if geometry_type is not None and geometry_type not in spec.geometry_types:
        errors.append(f"solver {algorithm!r} does not support geometry {geometry_type!r}")
    if observation_domain is not None and observation_domain not in spec.observation_domains:
        errors.append(
            f"solver {algorithm!r} does not support observation domain {observation_domain!r}; "
            f"supported domains are {spec.observation_domains}"
        )
    if regularizer is not None and regularizer not in spec.regularizers:
        errors.append(f"solver {algorithm!r} is not registered with regularizer {regularizer!r}")
    for rule in spec.compatibility_rules:
        if rule.matches(
            solver=algorithm,
            regularizer=regularizer,
            geometry_type=geometry_type,
            observation_domain=observation_domain,
        ) and not rule.allowed:
            errors.append(rule.message or f"incompatible solver combination for {algorithm!r}")
    if spec.supports_nonnegative_observation and observation_domain in {"log_projection", "line_integral"}:
        errors.append(
            f"{algorithm} requires nonnegative count/intensity observations; "
            f"{observation_domain} is a transmission/log projection domain"
        )
    return errors


def validate_parameter_values(
    algorithm: str,
    parameters: Mapping[str, Any] | None = None,
    *,
    views: int | None = None,
    geometry_type: str | None = None,
    dimension: int | None = None,
    observation_domain: str | None = None,
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
        return ParameterValidationResult(algorithm, {}, (f"unknown solver: {algorithm!r}",))
    spec = SOLVER_SPECS[algorithm]
    incoming = dict(parameters or {})
    allowed = set(spec.parameter_names)
    errors = [f"unknown parameter {name!r} for {algorithm}" for name in sorted(set(incoming) - allowed)]
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
            errors.append(f"parameter {parameter.name} is required")
            continue
        else:
            continue
        try:
            normalized[parameter.name] = parameter.normalize(value)
            sources.setdefault(parameter.name, "user" if parameter.name in incoming else "registry_default")
        except ValueError as error:
            errors.append(str(error))

    minimum = normalized.get("min_value")
    maximum = normalized.get("max_value")
    if minimum is not None and maximum is not None and minimum > maximum:
        errors.append("min_value must be less than or equal to max_value")

    if views is not None:
        views = int(views)
        if views <= 0:
            errors.append("number of views must be positive")
        for name in ("block_size",):
            value = normalized.get(name)
            if value is not None and views > 0 and value > views:
                errors.append(f"{name} must be no greater than the number of views ({views})")
        subset_count = normalized.get("subset_count")
        if subset_count is not None and views > 0 and subset_count > views:
            errors.append(f"subset_count must be between 1 and the number of views ({views})")

    errors.extend(
        validate_compatibility(
            algorithm,
            geometry_type=geometry_type,
            dimension=dimension,
            observation_domain=observation_domain,
        )
    )

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
                errors.append("estimated ||A||^2 must be positive")
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
                    errors.append(f"step_size must satisfy 0 < step_size < 2 / ||A||^2 ({upper:.6g})")
                if algorithm == "tv_fista" and not (0.0 < step <= upper):
                    errors.append(f"step_size must satisfy 0 < step_size <= 1 / ||A||^2 ({upper:.6g})")
                estimates.update({"operator_norm_squared": lipschitz, "step_size_source": "user"})

    if algorithm in {"sart", "os_sart"}:
        relaxation = normalized.get("relaxation")
        if relaxation is not None and not (0.0 < relaxation <= 1.0):
            errors.append("relaxation must satisfy 0 < relaxation <= 1")
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
    )

"""Public CT request and result contracts."""
from dataclasses import dataclass, field
from typing import Any, Literal, Callable

from inv_framework.api.execution import ConfigError, load_algorithm_config
from inv_framework.ct.registry import ALGORITHM_ALIASES, SOLVER_SPECS, validate_parameter_values
from pathlib import Path


@dataclass
class CTTaskRequest:
    case_id: str
    algorithm: str
    backend: Literal["torch", "astra", "leap"] = "torch"
    device: str | None = None
    data_root: str | None = None
    fit_view_indices: list[int] | None = None
    heldout_view_indices: list[int] | None = None
    stopping_policy: dict[str, Any] | None = None
    problem_constraints: dict[str, Any] | None = None
    callback: Callable[[Any], bool | None] | None = None
    fidelity: str | None = None
    regularizer: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class CTTaskResult:
    status: Literal["success", "failed"]
    case_id: str
    algorithm_summary: str
    metrics: dict[str, float]
    runtime_sec: float
    iterations: int
    converged: bool
    reconstruction: Any = None
    error_message: str | None = None
    convergence_status: str = "invalid_parameters"
    stopping_reason: str = "parameter_validation_failed"
    reason_code: str = "parameter_validation_failed"
    history: list[dict[str, Any]] = field(default_factory=list)


def normalize_request(request):
    if isinstance(request, dict):
        try:
            request = CTTaskRequest(**request)
        except TypeError as error:
            raise ConfigError(str(error)) from error
    if not isinstance(request, CTTaskRequest):
        raise ConfigError("request must be a CTTaskRequest or dict")
    for key in ("case_id", "algorithm", "backend"):
        if not isinstance(getattr(request, key), str) or not getattr(request, key):
            raise ConfigError(f"{key} must be a nonempty string")
    name = ALGORITHM_ALIASES.get(request.algorithm, request.algorithm)
    if name not in SOLVER_SPECS:
        raise ConfigError(f"unknown algorithm: {name}")
    if request.backend not in {"torch", "astra", "leap"}:
        raise ConfigError(f"unknown backend: {request.backend}")
    if not isinstance(request.params, dict):
        raise ConfigError("params must be a flat dict")
    if not all(isinstance(key, str) for key in request.params):
        raise ConfigError("params keys must be strings")
    if request.device is not None:
        import torch
        try:
            device = torch.device(request.device)
        except (TypeError, RuntimeError) as error:
            raise ConfigError(str(error)) from error
        if device.type not in {"cpu", "cuda"}:
            raise ConfigError("device must be cpu or cuda[:index]")
    if request.callback is not None and not callable(request.callback):
        raise ConfigError("callback must be callable")
    for key in ("stopping_policy", "problem_constraints"):
        if getattr(request, key) is not None and not isinstance(getattr(request, key), dict):
            raise ConfigError(f"{key} must be a dict")
    if request.stopping_policy is not None:
        policy = request.stopping_policy
        allowed = {"schema_version", "mode", "min_iterations", "check_every", "patience",
                   "stop_on_convergence", "relative_iterate_tolerance",
                   "normalized_normal_residual_tolerance", "relative_objective_tolerance",
                   "prox_gradient_mapping_tolerance", "discrepancy", "stalled", "divergence"}
        if set(policy) - allowed:
            raise ConfigError(f"unknown stopping fields: {sorted(set(policy)-allowed)}")
        for key in ("min_iterations", "check_every", "patience"):
            if key in policy and (type(policy[key]) is not int or policy[key] <= 0):
                raise ConfigError(f"{key} must be a positive integer")
        for key in ("discrepancy", "stalled", "divergence"):
            if key in policy and not isinstance(policy[key], dict):
                raise ConfigError(f"{key} must be a dict")
    if request.problem_constraints is not None:
        import math
        constraints = request.problem_constraints
        if set(constraints) - {"min_value", "max_value", "value_range", "data_range"}:
            raise ConfigError("unsupported physical constraint")
        if "value_range" in constraints:
            bounds = constraints["value_range"]
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2 or any(
                type(v) not in (int, float) or not math.isfinite(v) for v in bounds):
                raise ConfigError("value_range must contain two finite numbers")
        for key in ("min_value", "max_value", "data_range"):
            value = constraints.get(key)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                raise ConfigError(f"{key} must be finite numeric")
        if "data_range" in constraints and constraints["data_range"] <= 0:
            raise ConfigError("data_range must be positive")
    if (request.fit_view_indices is None) != (request.heldout_view_indices is None):
        raise ConfigError("fit and heldout indices must be supplied together")
    for indices in (request.fit_view_indices, request.heldout_view_indices):
        if indices is not None and (not indices or any(type(i) is not int for i in indices)):
            raise ConfigError("view indices must be nonempty integer lists")
    explicit = dict(request.params)
    # Public spellings map onto the existing registry's constructor parameters.
    for alias, canonical in {"max_iter": "num_iterations", "num_subsets": "subset_count", "lam": "strength"}.items():
        if alias in explicit:
            value = explicit.pop(alias)
            if canonical in explicit and explicit[canonical] != value:
                raise ConfigError(f"{alias} and {canonical} conflict")
            explicit[canonical] = value
    for key, value in (("data_fidelity", request.fidelity), ("regularizer", request.regularizer)):
        if value is not None:
            if not isinstance(value, str):
                raise ConfigError(f"{key} must be a string")
            if key in explicit and explicit[key] != value:
                raise ConfigError(f"top-level {key} conflicts with params")
            explicit[key] = value
    unknown = set(explicit) - set(SOLVER_SPECS[name].parameter_names)
    if name == "landweber":
        unknown.discard("learning_rate")
    if unknown:
        raise ConfigError(f"unsupported parameters for {name}: {sorted(unknown)}")
    root = Path(__file__).resolve().parents[2]
    _, defaults, _ = load_algorithm_config(root / "configs/algorithms" / f"{name}.yaml", name)
    parameters = {**defaults, **explicit}
    if request.problem_constraints:
        from .execution import _explicit_problem_bounds
        lower, upper = _explicit_problem_bounds(request.problem_constraints)
        if lower is not None and upper is not None and lower > upper:
            raise ConfigError("physical lower bound exceeds upper bound")
        for key, bound, combine in (("min_value", lower, max), ("max_value", upper, min)):
            if bound is not None and key in SOLVER_SPECS[name].parameter_names:
                current = parameters.get(key)
                parameters[key] = bound if current is None else combine(current, bound)
    validation = validate_parameter_values(name, parameters)
    if validation.errors:
        raise ConfigError("; ".join(validation.errors))
    return request, name, validation.parameters


def validate_ct_request(request):
    """Validate public parameters, geometry and measurement compatibility without solving.

    Return resolved parameters or raise ValueError with the rejection reason.
    """
    from . import execution
    request, name, parameters = normalize_request(request)
    device = request.device or ("cuda" if name == "fdk" else "cpu")
    case = execution.load_ct_case(request.case_id, data_root=request.data_root, device=device)
    if request.fit_view_indices is not None:
        if set(request.fit_view_indices) & set(request.heldout_view_indices):
            raise ConfigError("fit and heldout views must be disjoint")
        execution.restrict_ct_case(case, request.heldout_view_indices)
        case = execution.restrict_ct_case(case, request.fit_view_indices)
    execution.validate_solver_case(name, case)
    validation = execution.validate_parameters(name, parameters, case=case,
                                               problem_constraints=request.problem_constraints)
    if validation.errors:
        raise ConfigError("; ".join(validation.errors))
    return validation.parameters

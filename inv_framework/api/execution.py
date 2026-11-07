"""Shared in-memory CT execution, assembly and endpoint evaluation."""

from __future__ import annotations

import csv
import json
import math
import shutil
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from inv_framework.ct.cases import list_ct_cases, load_ct_case, restrict_ct_case
from inv_framework.solvers.convergence import CORE_CT_ALGORITHMS, ConvergenceReport, ConvergenceStatus, confirm_endpoint, normalize_status, post_run_validation, status_reason_class
from inv_framework.ct.backends import ParallelBeamRadon2D
from inv_framework.core.regularizer import TVRegularizer, TikhonovRegularizer, HaarL1Regularizer, ZeroRegularizer, GradientHuberRegularizer
from inv_framework.core.operator import GradientOperator
from inv_framework.solvers import CGLSSolver, FDKSolver, FBPSolver, LSQRSolver, LandweberSolver, MLEMSolver, OSEMSolver, OSSARTSolver, SARTSolver, SIRTSolver, TVFISTASolver, FISTASolver, TikhonovSolver, SolveControl
from inv_framework.ct.registry import ALGORITHM_ALIASES, CANONICAL_ALGORITHM_IDS, CTAlgorithmSpec, EMISSION_OBSERVATION_MODELS, REGISTRY_SCHEMA_VERSION, SOLVER_SPECS, ParameterValidationResult, regularizer_records, solver_records as _solver_records, validate_registry, validate_compatibility, validate_parameter_values
from inv_framework.utils.metrics import ssim


SCHEMA_VERSION = 1
RESULT_FILENAME = "reconstruction.pt"


class ConfigError(ValueError):
    """Raised for invalid user-authored CLI configuration."""


class InvalidParametersError(ConfigError):
    """Structured preflight rejection that must not enter a solver loop."""


class BackendUnavailable(RuntimeError):
    """Raised when an optional numerical backend cannot run a requested job."""


class NumericalFailure(RuntimeError):
    """Raised when a solver returns an invalid numerical result."""


def solver_records() -> list[dict[str, Any]]:
    """Return the canonical registry, annotated with runtime reachability."""

    validate_registry()
    records = {record["name"]: record for record in _solver_records()}
    return [
        {
            **records[name],
            "canonical_id": name,
            "build_reachable": name in SOLVER_BUILDERS,
            "config_filename": f"{name}.yaml",
            "stopping_policy_versions": ["1.0"],
            "endpoint_confirmation": True,
        }
        for name in CANONICAL_ALGORITHM_IDS
    ]


def algorithm_config_ids(config_root: str | Path | None = None) -> tuple[str, ...]:
    """Return algorithm IDs represented by YAML files in a config directory."""

    root = (
        Path(config_root).expanduser().resolve()
        if config_root is not None
        else Path(__file__).resolve().parents[2] / "configs" / "algorithms"
    )
    if not root.is_dir():
        return ()
    file_ids = {path.stem for path in root.glob("*.yaml") if path.is_file()}
    return tuple(
        name for name in CANONICAL_ALGORITHM_IDS if name in file_ids
    ) + tuple(sorted(file_ids - set(CANONICAL_ALGORITHM_IDS)))


def runtime_registry_contract(config_root: str | Path | None = None) -> dict[str, Any]:
    """Describe registry/build/config identity without executing a solver."""

    registry_ids = tuple(SOLVER_SPECS)
    build_ids = tuple(SOLVER_BUILDERS)
    config_ids = algorithm_config_ids(config_root)
    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "canonical_algorithm_ids": list(CANONICAL_ALGORITHM_IDS),
        "registry_ids": list(registry_ids),
        "build_ids": list(build_ids),
        "config_ids": list(config_ids),
        "aliases": dict(ALGORITHM_ALIASES),
        "missing_from_registry": sorted(
            set(CANONICAL_ALGORITHM_IDS) - set(registry_ids)
        ),
        "extra_in_registry": sorted(set(registry_ids) - set(CANONICAL_ALGORITHM_IDS)),
        "missing_from_build": sorted(set(CANONICAL_ALGORITHM_IDS) - set(build_ids)),
        "extra_in_build": sorted(set(build_ids) - set(CANONICAL_ALGORITHM_IDS)),
        "missing_configs": sorted(set(CANONICAL_ALGORITHM_IDS) - set(config_ids)),
        "extra_configs": sorted(set(config_ids) - set(CANONICAL_ALGORITHM_IDS)),
        "all_build_reachable": set(registry_ids)
        == set(build_ids)
        == set(CANONICAL_ALGORITHM_IDS),
        "all_configured": set(config_ids) == set(CANONICAL_ALGORITHM_IDS),
    }


def validate_runtime_registry(config_root: str | Path | None = None) -> dict[str, Any]:
    """Fail closed when registry, build map, or YAML inventory drifts."""

    validate_registry()
    report = runtime_registry_contract(config_root)
    if not report["all_build_reachable"] or not report["all_configured"]:
        raise ValueError(f"ordinary CT registry/build/config contract mismatch: {report}")
    return report


def regularizer_records_public() -> list[dict[str, Any]]:
    """Return registry regularizers without making the solver module public API mandatory."""

    return regularizer_records()


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{label} must be a mapping.")
    return dict(value)


def _check_keys(payload: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ConfigError(f"{label} contains unknown field(s): {', '.join(unknown)}")


def load_yaml(path: str | Path) -> tuple[dict[str, Any], Path]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ConfigError(f"YAML file does not exist: {source}")
    try:
        import yaml
    except ImportError as error:
        raise ConfigError("YAML support requires PyYAML.") from error
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except Exception as error:
        raise ConfigError(f"Cannot parse YAML {source}: {error}") from error
    return _require_mapping(payload, str(source)), source


def load_algorithm_config(
    path: str | Path, expected_solver: str | None = None
) -> tuple[str, dict[str, Any], Path]:
    payload, source = load_yaml(path)
    try:
        _check_keys(
            payload, {"schema_version", "name", "parameters"}, "algorithm config"
        )
        if payload.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
            raise ConfigError(
                f"algorithm config schema_version must be {SCHEMA_VERSION}."
            )
        name = payload.get("name")
        if not isinstance(name, str) or name not in SOLVER_SPECS:
            raise ConfigError(f"unknown solver in algorithm config: {name!r}")
        if expected_solver is not None and name != expected_solver:
            raise ConfigError(
                f"algorithm config names {name!r}, but command requested {expected_solver!r}."
            )
        parameters = _require_mapping(
            payload.get("parameters", {}), "algorithm parameters"
        )
    except ConfigError as error:
        raise InvalidParametersError(str(error)) from error
    accepted_names = set(SOLVER_SPECS[name].parameter_names)
    if name == "landweber":
        accepted_names.add("learning_rate")
    unknown = sorted(set(parameters) - accepted_names)
    if unknown:
        raise InvalidParametersError(
            f"unsupported parameter(s) for {name}: {', '.join(unknown)}"
        )
    try:
        _validate_parameter_types(parameters)
    except ConfigError as error:
        raise InvalidParametersError(str(error)) from error
    validation = validate_parameter_values(
        name,
        parameters,
        parameter_sources={
            str(parameter_name): "repository_config" for parameter_name in parameters
        },
    )
    if validation.errors:
        error = InvalidParametersError(f"invalid parameters for {name}: {'; '.join(validation.errors)}")
        error.validation = validation
        raise error
    return name, validation.parameters, source


def _validate_parameter_types(parameters: Mapping[str, Any]) -> None:
    integer_names = {
        "num_iterations",
        "block_size",
        "subset_count",
        "seed",
        "power_iterations",
        "tv_num_iterations",
        "voxel_supersampling",
    }
    numeric_names = {
        "scale",
        "min_value",
        "max_value",
        "step_size",
        "learning_rate",
        "tol",
        "damping",
        "atol",
        "btol",
        "relaxation",
        "eps",
        "initial_value",
        "reg_strength",
        "tolerance",
        "tv_tolerance",
    }
    string_names = {"order_strategy", "tv_mode", "filter_type"}
    nullable = {"scale", "min_value", "max_value", "step_size", "block_size", "seed"}
    for name, value in parameters.items():
        if value is None and name in nullable:
            continue
        if name in integer_names and (isinstance(value, bool) or not isinstance(value, int)):
            raise ConfigError(f"parameter {name} must be an integer.")
        if name in numeric_names and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ConfigError(f"parameter {name} must be numeric.")
        if name in string_names and not isinstance(value, str):
            raise ConfigError(f"parameter {name} must be a string.")
        if name == "short_scan" and not isinstance(value, bool):
            raise ConfigError("parameter short_scan must be a bool.")


def _case_observation_domain(case: Any) -> str:
    """Infer the public observation domain when a caller did not provide one."""

    metadata = getattr(case, "metadata", {}) or {}
    explicit = (metadata.get("measurement", {}) or {}).get("observation_domain", metadata.get("observation_domain"))
    if explicit:
        return str(explicit)
    measurement = metadata.get("measurement", {}) or {}
    noise_model = str(measurement.get("noise_model", ""))
    kind = str(measurement.get("kind", ""))
    if noise_model in {"poisson", "poisson_counts", "counts", "poisson_count"} or kind in {"count", "counts", "intensity"}:
        return "nonnegative_counts"
    if "log" in noise_model:
        return "log_projection"
    return "line_integral"


def _case_observation_model(case: Any) -> str | None:
    """Read the public likelihood/observation model, if one is declared."""

    metadata = getattr(case, "metadata", {}) or {}
    measurement = metadata.get("measurement", {}) or {}
    value = measurement.get("observation_model", metadata.get("observation_model"))
    return str(value) if value is not None and str(value) else None


def _explicit_problem_bounds(problem_constraints: Mapping[str, Any] | None) -> tuple[float | None, float | None]:
    """Read caller-provided value bounds without inspecting ground truth."""

    if not problem_constraints:
        return None, None
    lower = problem_constraints.get("min_value")
    upper = problem_constraints.get("max_value")
    value_range = problem_constraints.get("value_range")
    if isinstance(value_range, (list, tuple)) and len(value_range) == 2:
        lower = value_range[0] if lower is None else lower
        upper = value_range[1] if upper is None else upper

    def numeric(value: Any) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    return numeric(lower), numeric(upper)


def estimate_lipschitz_squared(operator: Any, reference: torch.Tensor, num_iterations: int = 12) -> float:
    """Estimate ``||A||^2`` with a deterministic power iteration."""

    if isinstance(num_iterations, bool) or not isinstance(num_iterations, int) or num_iterations < 1:
        raise ValueError("power-iteration count must be a positive integer")
    domain_shape = tuple(int(value) for value in operator.domain_shape)
    if not domain_shape or any(value <= 0 for value in domain_shape):
        raise ValueError("operator domain_shape must contain positive dimensions")
    sample_size = math.prod(domain_shape)
    dtype = reference.dtype if reference.is_floating_point() else torch.float32
    seed = torch.linspace(
        0.5,
        1.5,
        steps=sample_size,
        dtype=dtype,
        device=reference.device,
    ).reshape((1, *domain_shape))
    denominator = seed.reshape(seed.shape[0], -1).norm(dim=1)
    if not bool(torch.isfinite(denominator).all().item()) or bool(torch.any(denominator <= 0).item()):
        raise ValueError("deterministic power-iteration seed is invalid")
    vector = seed / denominator.reshape((1,) + (1,) * len(domain_shape)).clamp_min(1e-12)
    for _ in range(num_iterations):
        next_vector = operator.adjoint(operator.forward(vector))
        if not bool(torch.isfinite(next_vector).all().item()):
            raise FloatingPointError("power iteration produced a non-finite vector")
        norm = next_vector.reshape(next_vector.shape[0], -1).norm(dim=1)
        if bool(torch.all(norm <= 1e-12)):
            return 0.0
        vector = next_vector / norm.reshape((next_vector.shape[0],) + (1,) * len(domain_shape)).clamp_min(1e-12)
    image = operator.adjoint(operator.forward(vector))
    if not bool(torch.isfinite(image).all().item()):
        raise FloatingPointError("power iteration produced a non-finite estimate")
    value = (vector * image).reshape(vector.shape[0], -1).sum(dim=1).clamp_min(0.0).max()
    result = float(value.item())
    if not math.isfinite(result):
        raise FloatingPointError("power iteration returned a non-finite estimate")
    return result


def validate_parameters(
    name: str,
    parameters: Mapping[str, Any] | None = None,
    *,
    case: Any | None = None,
    operator: Any | None = None,
    observation_domain: str | None = None,
    problem_constraints: Mapping[str, Any] | None = None,
    parameter_sources: Mapping[str, str] | None = None,
    max_iterations: int | None = None,
    max_memory_mb: int | None = None,
    estimated_memory_mb: float | None = None,
    backend_capabilities: Mapping[str, Any] | Sequence[str] | None = None,
) -> ParameterValidationResult:
    """Validate one solver against public case metadata and operator bounds.

    Validation is deliberately staged.  Scalar, cross-parameter, geometry,
    observation and budget checks run before any power iteration.  Only a
    parameterization that passes that stage may request the deterministic
    operator-norm estimate used by Landweber/TV-FISTA.
    """

    geometry: Mapping[str, Any] = getattr(case, "geometry", {}) or {}
    geometry_type = str(geometry.get("type")) if geometry.get("type") else None
    dimension = None
    if case is not None:
        dimension = (getattr(case, "metadata", {}) or {}).get("dimension")
    views = None
    if case is not None:
        measurement = getattr(case, "measurement", None)
        if isinstance(measurement, torch.Tensor) and measurement.ndim >= 3:
            views = int(measurement.shape[-2])
        elif geometry.get("range_shape") and len(geometry["range_shape"]) >= 2:
            views = int(geometry["range_shape"][-2])
    domain = observation_domain or (_case_observation_domain(case) if case is not None else None)
    observation_model = _case_observation_model(case) if case is not None else None
    supplied = dict(parameters or {})
    sources: dict[str, str] = {str(key): str(value) for key, value in (parameter_sources or {}).items()}
    lower, upper = _explicit_problem_bounds(problem_constraints)
    if lower is not None and "min_value" not in supplied:
        supplied["min_value"] = lower
        sources["min_value"] = "problem_constraint"
    if upper is not None and "max_value" not in supplied:
        supplied["max_value"] = upper
        sources["max_value"] = "problem_constraint"

    measurement = getattr(case, "measurement", None) if case is not None else None
    if isinstance(measurement, torch.Tensor):
        try:
            observation_min = float(measurement.min().item()) if measurement.numel() else None
            observation_finite = bool(torch.isfinite(measurement).all().item())
        except (AttributeError, TypeError, ValueError, RuntimeError):
            observation_min = None
            observation_finite = None
    else:
        observation_min = None
        observation_finite = None

    case_metadata = (getattr(case, "metadata", {}) or {}) if case is not None else {}
    explicit_backend_capabilities = (
        backend_capabilities
        if backend_capabilities is not None
        else case_metadata.get("backend_capabilities")
    )
    runtime_device = str(getattr(measurement, "device", "")) or None
    domain_shape = geometry.get("domain_shape")
    image_shape = geometry.get("image_shape")
    # Preserve the raw context value.  The registry validator deliberately
    # rejects floats, bools, strings, zero and negative dimensions instead of
    # accepting an accidental ``int(...)`` truncation.
    dimension_value = dimension

    common_kwargs = {
        "views": views,
        "geometry_type": geometry_type,
        "dimension": dimension_value,
        "observation_domain": domain,
        "observation_model": observation_model,
        "observation_min": observation_min,
        "observation_finite": observation_finite,
        "parameter_sources": sources,
        "image_shape": image_shape,
        "domain_shape": domain_shape,
        "device": runtime_device,
        "backend_capabilities": explicit_backend_capabilities,
        "max_iterations": max_iterations,
        "max_memory_mb": max_memory_mb,
        "estimated_memory_mb": estimated_memory_mb,
    }

    # This first pass is intentionally operator-free.  It prevents malformed
    # power_iterations and other invalid inputs from reaching ``int(...)`` or
    # the solver/operator loop.
    preflight = validate_parameter_values(name, supplied, **common_kwargs)
    if preflight.errors or operator is None or name != "landweber":
        return preflight

    iterations = preflight.parameters.get("power_iterations", 12)
    reference = measurement
    if not isinstance(reference, torch.Tensor):
        reference = torch.ones((1, *tuple(operator.range_shape)), dtype=torch.float32)
    estimates: dict[str, Any] = {
        "operator_norm_estimator": "deterministic_power_iteration",
        "operator_norm_estimator_iterations": iterations,
    }
    try:
        estimated = estimate_lipschitz_squared(operator, reference, iterations)
    except Exception as error:
        estimates["operator_norm_error"] = str(error)
        return validate_parameter_values(
            name,
            supplied,
            parameter_estimates=estimates,
            **common_kwargs,
        )

    estimates["operator_norm_squared"] = estimated
    return validate_parameter_values(
        name,
        supplied,
        parameter_estimates=estimates,
        estimated_lipschitz=estimated,
        **common_kwargs,
    )


def validate_solver_case(name: str, case: Any, observation_domain: str | None = None) -> CTAlgorithmSpec:
    if name not in SOLVER_SPECS:
        raise ConfigError(f"unknown solver: {name!r}")
    spec = SOLVER_SPECS[name]
    raw_dimension = case.metadata.get("dimension", len(case.geometry.get("domain_shape", ())))
    if isinstance(raw_dimension, bool) or not isinstance(raw_dimension, int) or raw_dimension <= 0:
        raise ConfigError(
            f"case {case.case_id!r} has invalid dimension={raw_dimension!r}; "
            "dimension must be a strictly positive integer"
        )
    dimension = int(raw_dimension)
    geometry_type = str(case.geometry.get("type", ""))
    if dimension not in spec.dimensions or geometry_type not in spec.geometry_types:
        raise ConfigError(
            f"solver {name!r} supports dimension={spec.dimensions} and geometry="
            f"{spec.geometry_types}, but case {case.case_id!r} has dimension={dimension} "
            f"and geometry={geometry_type!r}."
        )
    domain = observation_domain or _case_observation_domain(case)
    observation_model = _case_observation_model(case)
    if name in {"mlem", "osem"}:
        measurement_meta = dict(case.metadata.get("measurement", {}) or {})
        observation_model = str(
            measurement_meta.get("observation_model", case.metadata.get("observation_model", ""))
        ).lower()
        # Nonnegative values alone do not establish the Poisson emission
        # model.  X-ray line integrals, including positive ones, are not
        # interchangeable with counts/intensities for EM updates.
        measurement_kind = str(measurement_meta.get("kind", "")).lower()
        if (
            measurement_kind in {"line_integral", "log_projection", "transmission"}
            or observation_model not in EMISSION_OBSERVATION_MODELS
        ):
            raise ConfigError(
                f"solver {name!r} requires an explicit Poisson emission/count "
                "observation_model; X-ray line-integral data are incompatible"
            )
    incompatibilities = validate_compatibility(
        name,
        geometry_type=geometry_type,
        dimension=dimension,
        observation_domain=domain,
        observation_model=observation_model,
    )
    if incompatibilities:
        raise ConfigError("; ".join(incompatibilities))
    if name == "fdk":
        shape_validation = validate_parameter_values(
            name,
            {},
            geometry_type=geometry_type,
            dimension=dimension,
            observation_domain=domain,
            domain_shape=case.geometry.get("domain_shape"),
            image_shape=case.geometry.get("image_shape"),
        )
        if shape_validation.errors:
            raise ConfigError("; ".join(shape_validation.errors))
    return spec


def require_astra_cuda(device):
    """Check FDK dependencies before moving case arrays to the GPU."""
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        raise BackendUnavailable("FDK requires --device cuda, PyTorch CUDA, and ASTRA CUDA.")
    try:
        from inv_framework.ct.backends import astra_adapter as astra_backend
    except ImportError as error:
        raise BackendUnavailable("astra-toolbox is not installed.") from error
    if not astra_backend._HAS_ASTRA:
        raise BackendUnavailable("astra-toolbox is not installed.")
    astra = astra_backend.astra
    if not astra.use_cuda():
        raise BackendUnavailable("FDK requires --device cuda, PyTorch CUDA, and ASTRA CUDA.")
    return astra


def build_operator(case: Any, device: str | torch.device, backend="torch"):
    geometry_type = case.geometry.get("type")
    resolved = torch.device(device)
    if geometry_type == "parallel_2d":
        domain = tuple(int(v) for v in case.geometry["domain_shape"])
        if len(domain) != 3 or domain[-1] != domain[-2]:
            raise ConfigError("ParallelBeamRadon2D requires square (C,H,W) domain_shape.")
        angles = torch.as_tensor(case.geometry["angles_rad"], dtype=torch.float32, device=resolved)
        if backend == "leap":
            from inv_framework.ct.backends.leap_adapter import LEAPOperator2D
            return LEAPOperator2D(domain[-1], angles, device=str(resolved), in_channels=domain[0])
        if backend == "astra":
            from inv_framework.ct.backends.astra_adapter import ASTRAOperator2D
            return ASTRAOperator2D(domain[-1], angles, device=str(resolved), in_channels=domain[0])
        return ParallelBeamRadon2D(domain[-1], angles=angles, device=str(resolved), in_channels=domain[0])
    if geometry_type == "cone_3d":
        from inv_framework.ct.backends import ASTRAFDKOperator3D
        astra = require_astra_cuda(resolved)
        domain = tuple(int(v) for v in case.geometry["domain_shape"])
        if len(domain) != 3 or len(set(domain)) != 1:
            raise ConfigError("ASTRA FDK requires a cubic 3D domain_shape.")
        angles = np.asarray(case.geometry["angles_rad"], dtype=np.float32)
        volume_geometry = astra.create_vol_geom(*domain)
        projection_geometry = astra.create_proj_geom(
            "cone",
            float(case.geometry["detector_spacing_row"]),
            float(case.geometry["detector_spacing_col"]),
            int(case.geometry["detector_rows"]),
            int(case.geometry["detector_cols"]),
            angles,
            float(case.geometry["source_origin_distance"]),
            float(case.geometry["origin_detector_distance"]),
        )
        return ASTRAFDKOperator3D(volume_geometry, projection_geometry)
    raise ConfigError(f"unsupported CT geometry type: {geometry_type!r}")


def _build_tv_fista(**parameters: Any):
    """Construct the fixed TV-FISTA/prior pairing from normalized parameters."""

    params = dict(parameters)
    tv = TVRegularizer(
        mode=params.pop("tv_mode", "isotropic"),
        num_iterations=int(params.pop("tv_num_iterations", 50)),
        tolerance=float(params.pop("tv_tolerance", 1e-5)),
    )
    return TVFISTASolver(regularizer=tv, **params)


def _build_fista(**parameters):
    params = dict(parameters)
    prior = params.pop("regularizer", "tikhonov")
    level = params.pop("level", 2)
    tv_mode = params.pop("tv_mode", "isotropic")
    tv_iterations = params.pop("tv_num_iterations", 50)
    tv_tolerance = params.pop("tv_tolerance", 1e-5)
    delta = params.pop('delta', .01)
    if prior in (None, 'none'):
        reg = ZeroRegularizer()
        params['strength'] = 0.
        params.pop('reg_strength', None)
    elif prior == 'h1':
        # The concrete image shape is supplied by build_solver below.
        reg = params.pop('smooth_regularizer')
    elif prior == 'huber':
        reg = GradientHuberRegularizer(delta)
    elif prior == "tikhonov":
        reg = TikhonovRegularizer()
    elif prior == "haar_l1":
        reg = HaarL1Regularizer(level)
        reg.num_iterations, reg.tolerance = tv_iterations, tv_tolerance
    elif prior == "tv":
        reg = TVRegularizer(mode=tv_mode, num_iterations=tv_iterations, tolerance=tv_tolerance)
    else:
        raise ValueError(f"unsupported FISTA regularizer: {prior}")
    return FISTASolver(regularizer=reg, **params)


CONSTRAINED_TV_IDS = {"constrained_pdhg", "asd_pocs_tv"}


def _build_constrained(name, **parameters):
    from inv_framework.solvers import ConstrainedTVSolver
    params = dict(parameters)
    params.pop('min_value', None)
    params.pop('max_value', None)
    return ConstrainedTVSolver(algorithm='pdhg' if name=='constrained_pdhg' else name, **params)


def _confirm_constrained_endpoint(solver, result, reconstruction, prediction, measurement, mask):
    from inv_framework.core.fidelity import ResidualBall
    cfg = solver.parameters
    raw_residual = float(((prediction-measurement)*(mask.to(prediction) if mask is not None else 1.)).norm())
    if solver.whitening is not None:
        prediction, measurement = prediction*solver.whitening, measurement*solver.whitening
    constraint = ResidualBall(measurement, cfg['epsilon'], source=cfg['epsilon_source'], mask=mask).diagnostics(prediction)
    tv = float(solver.regularizer.value(reconstruction))
    feasible = constraint['constraint_violation'] <= cfg['feasibility_tolerance']
    complete = result.metadata['endpoint_diagnostics_complete']
    consistent = complete and abs(tv-result.final_objective) <= 1e-5*max(1., abs(tv)) and abs(constraint['raw_residual_norm']-result.final_residual) <= 1e-5*max(1.,result.final_residual)
    passed = bool(consistent and torch.isfinite(reconstruction).all() and (reconstruction>=0).all())
    if result.status == 'converged':
        passed = passed and feasible and result.metadata['optimization_residual'] <= result.metadata['parameters']['tolerance']
    return {'schema_version':'ct.endpoint_confirmation.v1', 'status':'confirmed' if passed else 'unconfirmed',
            'passed':passed, 'finite':bool(torch.isfinite(reconstruction).all() and math.isfinite(tv) and math.isfinite(constraint['raw_residual_norm'])), 'endpoint_diagnostics_complete':complete, 'tv':tv,
            'constraint_diagnostics':{**constraint, 'residual_metric':cfg['residual_metric'], 'unwhitened_residual_norm':raw_residual}, 'feasibility_status':'feasible' if feasible else 'violated',
            'solver_status':result.status, 'optimization_residual':result.metadata['optimization_residual'],
            'regularizer_work':{'endpoint_tv_value_calls':1, 'transform_forward_calls':1},
            'reasons':[] if passed else ['constrained_endpoint_inconsistent']}


def _build_split(name, **parameters):
    from inv_framework.solvers.splitting import ADMMSolver, PDHGSolver
    from inv_framework.core.regularizer import GradientHuberRegularizer
    params = dict(parameters)
    prior = params.pop("regularizer", "tikhonov")
    level = params.pop("level", 2)
    delta = params.pop("delta", .01)
    mode = params.pop("tv_mode", "isotropic")
    reg = {"none": lambda: ZeroRegularizer(), "tikhonov": lambda: TikhonovRegularizer(), "h1": lambda: "h1",
           "tv": lambda: TVRegularizer(mode=mode), "huber": lambda: GradientHuberRegularizer(delta),
           "haar_l1": lambda: HaarL1Regularizer(level)}[prior]()
    return (ADMMSolver if name == "admm" else PDHGSolver)(regularizer=reg, **params)


# Keep this map at module scope so reachability is inspectable and testable.
# The values are constructors, not additional algorithm registrations.
SOLVER_BUILDERS: dict[str, Callable[..., Any]] = {
    "fbp": FBPSolver,
    "sirt": SIRTSolver,
    "landweber": LandweberSolver,
    "cgls": CGLSSolver,
    "lsqr": LSQRSolver,
    "sart": SARTSolver,
    "os_sart": OSSARTSolver,
    "mlem": MLEMSolver,
    "osem": OSEMSolver,
    "tikhonov": TikhonovSolver,
    "tv_fista": _build_tv_fista,
    "fista": _build_fista,
    "admm": lambda **p: _build_split("admm", **p),
    "pdhg": lambda **p: _build_split("pdhg", **p),
    "constrained_pdhg": lambda **p: _build_constrained("constrained_pdhg", **p),
    "asd_pocs_tv": lambda **p: _build_constrained("asd_pocs_tv", **p),
    "fdk": FDKSolver,
}
BUILD_SOLVER_MAP = SOLVER_BUILDERS
BUILD_ALGORITHM_IDS = tuple(SOLVER_BUILDERS)


def build_solver(name: str, parameters: Mapping[str, Any], case: Any):
    params = dict(parameters)
    fidelity_options = {key: params.pop(key, None) for key in
        ('data_fidelity','weight_source','nll_tolerance','nll_max_iterations')}
    if name not in SOLVER_SPECS:
        raise ConfigError(f"unknown solver: {name!r}")
    if name not in SOLVER_BUILDERS:
        raise ConfigError(f"solver {name!r} has no registered build path")
    num_views = int(case.measurement.shape[-2])
    if name in {"os_sart", "osem"} and params.get("subset_count") is not None:
        count = int(params.pop("subset_count"))
        if count <= 0 or count > num_views:
            raise ConfigError(f"subset_count must be between 1 and the number of views ({num_views}).")
        # A floor-based block size can silently create a different number of
        # subsets when the view count is not divisible by the requested
        # count.  Explicit balanced partitions make the protocol exact.
        base, remainder = divmod(num_views, count)
        subsets = []
        start = 0
        for index in range(count):
            size = base + (1 if index < remainder else 0)
            subsets.append(tuple(range(start, start + size)))
            start += size
        params.pop("block_size", None)
        params["subset_indices"] = subsets
    elif name in {"os_sart", "osem"}:
        # ``None`` is a valid registry default for the optional subset count,
        # but it is not a constructor keyword.  Leave backend-derived subset
        # partitioning to the solver when neither subset_count nor block_size
        # was supplied.
        params.pop("subset_count", None)
    if name == "tikhonov":
        prior = params.pop("regularizer", "tikhonov")
        strength = params.pop("strength", None)
        if strength is not None:
            if "reg_strength" in params and params["reg_strength"] != strength:
                raise ConfigError("strength and reg_strength conflict")
            params["reg_strength"] = strength
        if prior == "h1":
            params["regularization_operator"] = GradientOperator(case.geometry["domain_shape"])
    if name == 'fista' and params.get('regularizer') == 'h1':
        params['smooth_regularizer'] = TikhonovRegularizer(GradientOperator(case.geometry['domain_shape']))
    builder = SOLVER_BUILDERS[name]
    try:
        solver = builder(**params)
        if name in CONSTRAINED_TV_IDS and params.get('residual_metric') == 'gaussian_whitened':
            from inv_framework.ct.measurement import _parameter
            public=getattr(case, 'public_measurement', {}) or {}
            if 'sigma' not in public: raise ValueError('gaussian whitening requires public sigma')
            solver.whitening=_parameter(public['sigma'],case.measurement,'sigma',True).reciprocal()
        if name in {'fista','admm','pdhg','tikhonov','cgls','lsqr'}:
            from inv_framework.core.fidelity import build_fidelity
            model = fidelity_options['data_fidelity'] or 'l2'
            if model=='transmission_poisson_nll' and params.get('min_value')!=0:
                raise ValueError('P3 transmission_poisson_nll requires min_value=0')
            solver.fidelity = build_fidelity(model,case.measurement,getattr(case,'public_measurement',None),
                (getattr(case,'metadata',{}) or {}).get('measurement',{}), mask=getattr(case,'valid_measurement_mask',None),
                weight_source=fidelity_options['weight_source'], tolerance=fidelity_options['nll_tolerance'],
                max_iterations=fidelity_options['nll_max_iterations'] or 200) if model != 'l2' or (getattr(case,'valid_measurement_mask',None) is not None and not bool(getattr(case,'valid_measurement_mask',None).all())) else None
            if name=='tikhonov' and solver.fidelity is not None and (params.get('min_value') is not None or params.get('max_value') is not None):
                raise ValueError('weighted Tikhonov CG supports unconstrained reconstruction only')
            if model == 'transmission_poisson_nll':
                solver.fidelity.validate_minimizer(params.get('regularizer'), params.get('strength',0), params.get('max_value'))
        return solver
    except (TypeError, ValueError) as error:
        raise ConfigError(f"invalid parameters for {name}: {error}") from error


def _relative_data_residual(
    measurement: torch.Tensor,
    predicted: torch.Tensor,
    valid_measurement_mask: torch.Tensor | None = None,
) -> tuple[float, float, float | None]:
    difference = predicted - measurement
    valid_fraction = None
    if valid_measurement_mask is not None:
        mask = valid_measurement_mask.to(device=measurement.device, dtype=torch.bool)
        if mask.ndim == measurement.ndim - 1:
            mask = mask.unsqueeze(0)
        if tuple(mask.shape) == tuple(measurement.shape):
            difference = difference.masked_fill(~mask, 0.0)
            measurement = measurement.masked_fill(~mask, 0.0)
            valid_fraction = float(mask.float().mean().item())
    numerator = difference.reshape(difference.shape[0], -1).norm(dim=1)
    denominator = measurement.reshape(measurement.shape[0], -1).norm(dim=1).clamp_min(1e-12)
    return float((numerator / denominator).mean().item()), float(measurement.norm().item()), valid_fraction


def _metrics(bundle: Mapping[str, Any], runtime_seconds: float | None = None) -> dict[str, Any]:
    reconstruction = bundle["reconstruction"].float()
    measurement = bundle["measurement"].float()
    predicted = bundle["predicted_measurement"].float()
    data_residual, measurement_norm, valid_fraction = _relative_data_residual(
        measurement,
        predicted,
        bundle.get("valid_measurement_mask"),
    )
    truth_available = bool(bundle.get("ground_truth_available", True))
    result = {
        "schema_version": SCHEMA_VERSION,
        "solver": str(bundle["solver"]),
        "case_id": str(bundle["case_id"]),
        # ``status`` remains the backwards-compatible artifact-exists status;
        # solver termination is reported separately and must not be mistaken
        # for convergence (in particular, max_iterations is preserved).
        "status": "success",
        "execution_status": str(bundle.get("execution_status", "success")),
        "convergence_status": str(bundle.get("convergence_status", bundle.get("execution_status", "success"))),
        "stopping_reason": bundle.get("stopping_reason"),
        "dimension": int(bundle["dimension"]),
        "data_residual": data_residual,
        "measurement_norm": measurement_norm,
        "ground_truth_available": truth_available,
        "quality_metrics_valid": truth_available,
    }
    # A staged Agent case contains a zero tensor only to satisfy the legacy
    # loader shape contract.  Never compute image metrics from that tensor.
    if truth_available:
        truth = bundle["truth"].float()
        data_range = float(bundle["data_range"])
        difference = reconstruction - truth
        rmse = difference.square().mean().sqrt()
        relative_error = difference.norm() / truth.norm().clamp_min(1e-12)
        psnr_value = 20.0 * torch.log10(torch.as_tensor(data_range) / rmse.clamp_min(1e-12))
        if reconstruction.ndim == 4 and int(bundle["dimension"]) == 3:
            batch, depth, height, width = reconstruction.shape
            ssim_value = ssim(
                reconstruction.reshape(batch * depth, 1, height, width),
                truth.reshape(batch * depth, 1, height, width),
                data_range=data_range,
            ).mean()
            ssim_definition = "axial_mean_ssim"
        else:
            ssim_value = ssim(reconstruction, truth, data_range=data_range).mean()
            ssim_definition = "2d_ssim"
        result.update({
            "relative_error": float(relative_error.item()),
            "rmse": float(rmse.item()),
            "psnr": float(psnr_value.item()),
            "ssim": float(ssim_value.item()),
            "ssim_definition": ssim_definition,
        })
    if runtime_seconds is not None:
        result["runtime_seconds"] = float(runtime_seconds)
    if valid_fraction is not None:
        result["valid_measurement_fraction"] = valid_fraction
    heldout_measurement = bundle.get("heldout_measurement")
    heldout_predicted = bundle.get("heldout_predicted_measurement")
    if isinstance(heldout_measurement, torch.Tensor) and isinstance(heldout_predicted, torch.Tensor):
        heldout_residual, heldout_norm, heldout_fraction = _relative_data_residual(
            heldout_measurement.float(),
            heldout_predicted.float(),
            bundle.get("heldout_valid_measurement_mask"),
        )
        result.update({
            "held_out_projection_residual": heldout_residual,
            "held_out_measurement_norm": heldout_norm,
        })
        if heldout_fraction is not None:
            result["held_out_valid_measurement_fraction"] = heldout_fraction
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _discrepancy_target(case: Any, policy: Mapping[str, Any]) -> float | None:
    discrepancy = dict(policy.get("discrepancy", {}) or {})
    case_metadata = getattr(case, "metadata", {}) or {}
    measurement_metadata = dict(case_metadata.get("measurement", {}) or {})
    measurement_kind = str(
        measurement_metadata.get("kind", case_metadata.get("measurement_kind", ""))
    ).lower()
    observation_model = str(
        measurement_metadata.get(
            "observation_model", case_metadata.get("observation_model", "")
        )
    ).lower()
    if (
        observation_model in EMISSION_OBSERVATION_MODELS
        or measurement_kind in {"count", "counts", "intensity", "emission"}
    ):
        # The log-projection uncertainty estimate below is not meaningful for
        # emission data.  Use an explicit normalized Poisson-deviance target
        # when one is supplied; otherwise native EM evidence remains the
        # applicable stopping criterion.
        explicit_target = discrepancy.get(
            "normalized_poisson_deviance_target",
            discrepancy.get("poisson_deviance_target", discrepancy.get("target")),
        )
        if explicit_target is None:
            return None
        try:
            target = float(explicit_target)
        except (TypeError, ValueError, OverflowError) as error:
            raise ConfigError(
                "count-domain discrepancy target must be finite and nonnegative"
            ) from error
        if not math.isfinite(target) or target < 0.0:
            raise ConfigError(
                "count-domain discrepancy target must be finite and nonnegative"
            )
        return target
    parameters = dict((case_metadata.get("measurement") or {}).get("parameters", {}) or {})
    if not parameters:
        parameters = dict(case_metadata.get("measurement_parameters", {}) or {})
    epsilon = float(discrepancy.get("epsilon", 1e-12))
    floor = float(discrepancy.get("count_floor", 1.0))
    values = case.measurement
    counts = case.public_measurement.get("counts")
    if counts is None:
        incident = parameters.get("incident_photon_count")
        if not isinstance(incident, (int, float)) or float(incident) <= 0.0:
            raise ConfigError("discrepancy stopping requires counts or incident_photon_count")
        counts = float(incident) * torch.exp(-values)
    counts = counts.to(values).clamp_min(floor)
    mask = case.valid_measurement_mask if discrepancy.get("use_valid_measurement_mask", True) else None
    if mask is not None:
        mask = mask.to(device=values.device, dtype=torch.bool)
        if mask.ndim == values.ndim - 1:
            mask = mask.unsqueeze(0)
        values, counts = values[mask], counts[mask]
    else:
        values, counts = values.reshape(-1), counts.reshape(-1)
    sigma_sq_sum = counts.reciprocal().sum()
    delta = float(torch.sqrt(sigma_sq_sum).item()) / max(float(values.norm().item()), epsilon)
    return float(discrepancy.get("tau", 1.05)) * delta


def _failure(
    destination: Path,
    status: str,
    error: BaseException,
    *,
    algorithm: str | None = None,
    reason_code: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        canonical = normalize_status(
            status,
            algorithm=algorithm,
            direct=algorithm in {"fbp", "fdk"},
        ).value
        reason_class = status_reason_class(canonical)
    except ValueError:
        # ``failed`` is a process-level legacy artifact status, not a
        # convergence status.  Keep it serializable for the generic CLI
        # exception path without pretending it is numerical convergence.
        canonical = str(status)
        reason_class = "execution"
    reason = str(
        reason_code
        or {
            "invalid_parameters": "parameter_validation_failed",
            "unavailable": "backend_unavailable",
            "resource_exhausted": "resource_budget_exhausted",
            "numerical_error": "numerical_error",
            "cancelled": "cancelled_by_callback",
            "diverged": "persistent_worsening",
        }.get(canonical, canonical)
    )
    record = {
        "schema_version": SCHEMA_VERSION,
        "status": canonical,
        "convergence_status": canonical if reason_class != "execution" else None,
        "reason_code": reason,
        "reason_class": reason_class,
        "error_type": type(error).__name__,
        "message": str(error),
        "details": dict(details or {}),
    }
    if destination is not None:
        _write_json(destination / "failure.json", record)
        (destination / "failure_report.md").write_text(
            f"# 运行失败\n\n- 状态：`{status}`\n- 类型：`{type(error).__name__}`\n- 原因：{error}\n",
            encoding="utf-8",
        )
        (destination / "traceback.log").write_text(
            "".join(traceback.format_exception(error)), encoding="utf-8"
        )
    return record


def _structured_failure(
    destination: Path,
    *,
    status: str,
    error: BaseException,
    algorithm: str | None = None,
    reason_code: str | None = None,
    parameters: Mapping[str, Any] | None = None,
    resources: Mapping[str, Any] | None = None,
    validation: ParameterValidationResult | None = None,
    solve_result=None,
) -> dict[str, Any]:
    """Write a portable no-loop/error result and keep status layers aligned."""

    canonical = normalize_status(
        status,
        algorithm=algorithm,
        direct=algorithm in {"fbp", "fdk"},
    ).value
    reason = str(
        reason_code
        or {
            "invalid_parameters": "parameter_validation_failed",
            "unavailable": "backend_unavailable",
            "resource_exhausted": "resource_budget_exhausted",
            "numerical_error": "numerical_error",
        }.get(canonical, canonical)
    )
    validation_record = validation.to_dict() if validation is not None else None
    report = ConvergenceReport(
        status=canonical,
        stopping_reason=reason,
        algorithm=algorithm,
        failure_reason=str(error),
        terminal_evidence=(
            {
                "validation_errors": (
                    list(validation.errors) if validation is not None else [str(error)]
                ),
                "validation_reason_codes": (
                    list(validation.reason_codes) if validation is not None else []
                ),
                "parameters": dict(parameters or {}),
            }
            if canonical == "invalid_parameters"
            else {}
        ),
    )
    record = _failure(
        destination,
        canonical,
        error,
        algorithm=algorithm,
        reason_code=reason,
        details=(
            {
                "parameters": dict(parameters or {}),
                "parameter_validation": validation_record,
            }
            if parameters or validation_record is not None
            else None
        ),
    )
    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "status": canonical,
        "execution_status": canonical,
        "convergence_status": canonical,
        "stopping_reason": reason,
        "reason_code": report.reason_code,
        "reason_class": report.reason_class,
        "iterations_requested": 0,
        "iterations_completed": 0,
        "convergence": report.to_dict(),
        "resources": dict(resources or {}),
        "parameters": dict(parameters or {}),
        "parameter_validation": validation_record,
        "parameter_sources": dict(validation.sources) if validation is not None else {},
        "estimates": dict(validation.estimates) if validation is not None else {},
    }
    if solve_result is not None:
        diagnostics["iterations_completed"] = solve_result.actual_iterations
        diagnostics["solver_metadata"] = dict(solve_result.metadata)
        diagnostics["convergence"]["iterations"] = solve_result.actual_iterations
        diagnostics["convergence"]["trajectory"] = [
            row.to_dict() for row in solve_result.trajectory
        ]
        diagnostics["resources"] = {
            **dict(solve_result.resources),
            **dict(resources or {}),
        }
        diagnostics["partial_reconstruction"] = "partial_reconstruction.pt"
        if destination is not None:
            torch.save(solve_result.reconstruction.detach().cpu(), destination / "partial_reconstruction.pt")
    if destination is not None:
        _write_json(destination / "diagnostics.json", diagnostics)
    return {
        "status": canonical,
        "execution_status": canonical,
        "convergence_status": canonical,
        "stopping_reason": reason,
        "output_dir": destination,
        "diagnostics": diagnostics,
        "partial_reconstruction": solve_result.reconstruction.detach().cpu() if solve_result is not None else None,
        **record,
    }


def execute_case(
    solver_name: str,
    case_id: str,
    parameters: Mapping[str, Any],
    *,
    device: str = "cpu",
    data_root: str | Path | None = None,
    observation_domain: str | None = None,
    problem_constraints: Mapping[str, Any] | None = None,
    max_iterations: int | None = None,
    stopping_policy: Mapping[str, Any] | None = None,
    parameter_overrides: Mapping[str, Any] | None = None,
    parameter_sources: Mapping[str, str] | None = None,
    fit_view_indices: Sequence[int] | None = None,
    heldout_view_indices: Sequence[int] | None = None,
    split_metadata: Mapping[str, Any] | None = None,
    fixed_compute: bool = False,
    callback=None,
    backend: str = "torch",
    input_case=None,
) -> dict[str, Any]:
    destination = None
    operator = heldout_operator = None
    solve_result = None
    run_started = time.perf_counter()
    diagnostics: dict[str, Any] = {}
    validation: ParameterValidationResult | None = None
    try:
        name = solver_name
        parameters = dict(parameters)
        if parameter_overrides:
            accepted_names = set(SOLVER_SPECS[name].parameter_names)
            if name == "landweber":
                accepted_names.add("learning_rate")
            unknown = sorted(set(parameter_overrides) - accepted_names)
            if unknown:
                raise InvalidParametersError(
                    f"unsupported parameter override(s) for {name}: {', '.join(unknown)}"
                )
            parameters = {**parameters, **dict(parameter_overrides)}
            try:
                _validate_parameter_types(parameter_overrides)
            except ConfigError as error:
                raise InvalidParametersError(str(error)) from error
            override_validation = validate_parameter_values(name, parameters)
            validation = override_validation
            if override_validation.errors:
                raise InvalidParametersError(
                    f"invalid parameter overrides for {name}: {'; '.join(override_validation.errors)}"
                )
        parameter_source_map = dict(parameter_sources or {})
        requested_stopping_policy = ({
            "min_iterations": 1, "check_every": 1, "patience": 5,
            "stop_on_convergence": True, "relative_iterate_tolerance": 1e-5,
            "normalized_normal_residual_tolerance": 1e-5,
            "relative_objective_tolerance": 1e-5,
            "prox_gradient_mapping_tolerance": 1e-5,
            **stopping_policy,
        } if stopping_policy is not None else None)
        for parameter_name in parameters:
            parameter_source_map.setdefault(parameter_name, "repository_config")
        full_case = input_case if input_case is not None else load_ct_case(case_id, data_root=data_root, device=device)
        if (fit_view_indices is None) != (heldout_view_indices is None):
            raise ConfigError(
                "fit_view_indices and heldout_view_indices must be supplied together"
            )
        case = full_case
        heldout_case = None
        if fit_view_indices is not None and heldout_view_indices is not None:
            fit = tuple((int(value) for value in fit_view_indices))
            heldout = tuple((int(value) for value in heldout_view_indices))
            if set(fit).intersection(heldout):
                raise ConfigError("fit and held-out view indices must be disjoint")
            case = restrict_ct_case(full_case, fit, partition="fit")
            heldout_case = restrict_ct_case(full_case, heldout, partition="heldout")
        domain = observation_domain or _case_observation_domain(case)
        try:
            spec = validate_solver_case(name, case, domain)
        except ConfigError as error:
            raise InvalidParametersError(str(error)) from error
        if (
            name in CONSTRAINED_TV_IDS
            and requested_stopping_policy is not None
            and (requested_stopping_policy.get("discrepancy", {}) or {}).get(
                "enabled", True
            )
        ):
            raise InvalidParametersError(
                "constrained TV uses epsilon; set discrepancy.enabled=false"
            )
        operator = build_operator(case, device, backend)
        validation = validate_parameters(
            name,
            parameters,
            case=case,
            operator=operator,
            observation_domain=domain,
            problem_constraints=problem_constraints
            or case.metadata.get("problem_constraints"),
            parameter_sources=parameter_source_map,
            max_iterations=max_iterations,
        )
        if validation.errors:
            raise InvalidParametersError(
                f"invalid parameters for {name}: {'; '.join(validation.errors)}"
            )
        requested_iterations = int(validation.parameters.get("num_iterations", 0) or 0)
        try:
            solver = build_solver(name, validation.parameters, case)
        except ConfigError as error:
            raise InvalidParametersError(str(error)) from error
        operator_norm_squared = validation.estimates.get("operator_norm_squared")
        if (
            requested_stopping_policy is not None
            and name not in {"fbp", "fdk"}
            and (operator_norm_squared is None)
        ):
            operator_norm_squared = estimate_lipschitz_squared(
                operator, case.measurement, num_iterations=8
            )
            validation.estimates["operator_norm_squared"] = operator_norm_squared
            validation.estimates["operator_norm_estimator"] = (
                "power_iteration_8_for_stopping_policy"
            )
        detail_tolerance = None
        if fixed_compute and name not in {"fbp", "fdk"}:
            detail_tolerance = 0.0
        elif name not in {"fbp", "fdk", "cgls", "lsqr"}:
            detail_tolerance = float(validation.parameters.get("tolerance", 1e-05))
        if (
            validation.parameters.get("data_fidelity", "l2") != "l2"
            and requested_stopping_policy is not None
            and (requested_stopping_policy.get("discrepancy", {}) or {}).get(
                "enabled", True
            )
        ):
            raise InvalidParametersError(
                "non-L2 fidelity requires discrepancy.enabled=false"
            )
        effective_stopping_policy = (
            dict(requested_stopping_policy)
            if requested_stopping_policy is not None
            else None
        )
        control_kwargs: dict[str, Any] = {}
        if effective_stopping_policy is not None and name not in {"fbp", "fdk"}:
            stalled_policy = dict(effective_stopping_policy.get("stalled", {}) or {})
            divergence_policy = dict(
                effective_stopping_policy.get("divergence", {}) or {}
            )
            discrepancy_cfg = dict(
                effective_stopping_policy.get("discrepancy", {}) or {}
            )
            try:
                target = (
                    _discrepancy_target(case, effective_stopping_policy)
                    if discrepancy_cfg.get("enabled", True)
                    else None
                )
            except ConfigError as error:
                raise InvalidParametersError(str(error)) from error
            effective_stopping_policy["effective"] = {"discrepancy_target": target}
            control_kwargs = {
                "min_iterations": int(effective_stopping_policy["min_iterations"]),
                "check_every": int(effective_stopping_policy["check_every"]),
                "patience": int(effective_stopping_policy["patience"]),
                "stop_on_convergence": bool(
                    effective_stopping_policy["stop_on_convergence"]
                ),
                "relative_iterate_tolerance": float(
                    effective_stopping_policy["relative_iterate_tolerance"]
                ),
                "normalized_normal_residual_tolerance": float(
                    effective_stopping_policy["normalized_normal_residual_tolerance"]
                ),
                "relative_objective_tolerance": float(
                    effective_stopping_policy["relative_objective_tolerance"]
                ),
                "prox_gradient_mapping_tolerance": float(
                    effective_stopping_policy["prox_gradient_mapping_tolerance"]
                ),
                "discrepancy_target": target,
                "stall_enabled": bool(stalled_policy.get("enabled", True)),
                "stall_relative_iterate_tolerance": float(
                    stalled_policy.get("relative_iterate_tolerance", 1e-08)
                ),
                "stall_patience": int(stalled_policy.get("patience", 5)),
                "divergence_enabled": bool(divergence_policy.get("enabled", True)),
                "divergence_relative_increase_tolerance": float(
                    divergence_policy.get("relative_increase_tolerance", 0.0001)
                ),
                "divergence_patience": int(divergence_policy.get("patience", 5)),
            }
        operator_norm_for_diagnostics = (
            1.0
            if operator_norm_squared is None
            else math.sqrt(float(operator_norm_squared))
        )
        control = SolveControl(
            callback=callback,
            max_iterations=max(1, requested_iterations or 1),
            tolerance=detail_tolerance,
            metadata={
                "source": "ct.api",
                "solver": name,
                "effective_stopping_policy": effective_stopping_policy,
                "operator_norm_estimate": operator_norm_for_diagnostics,
            },
            max_trajectory_points=(
                max(200, requested_iterations)
                if requested_stopping_policy is not None
                else 200
            ),
            **(
                {**control_kwargs, "stop_on_convergence": False}
                if fixed_compute
                else control_kwargs
            ),
        )
        solve_result = solver.solve_detailed(
            case.measurement,
            operator,
            control=control,
            **(
                {"mask": case.valid_measurement_mask}
                if name in CONSTRAINED_TV_IDS
                else {}
            ),
            **(
                {"fidelity": solver.fidelity}
                if getattr(solver, "fidelity", None) is not None
                else {}
            ),
        )
        if solve_result.status in {"resource_exhausted", "cancelled"}:
            return _structured_failure(
                destination,
                status=solve_result.status,
                error=RuntimeError(solve_result.stopping_reason),
                algorithm=name,
                reason_code=solve_result.stopping_reason,
                solve_result=solve_result,
                resources={},
            )
        if solve_result.status == "invalid_parameters":
            raise InvalidParametersError(solve_result.stopping_reason)
        if solve_result.status == "unavailable":
            raise BackendUnavailable(solve_result.stopping_reason)
        if solve_result.status == "numerical_error":
            raise NumericalFailure(solve_result.stopping_reason)
        reconstruction = solve_result.reconstruction
        if not isinstance(reconstruction, torch.Tensor):
            raise NumericalFailure(
                f"solver returned {type(reconstruction).__name__}, expected a tensor"
            )
        if tuple(reconstruction.shape) != tuple(case.truth.shape):
            raise NumericalFailure(
                f"solver returned shape {tuple(reconstruction.shape)}, expected {tuple(case.truth.shape)}"
            )
        if not torch.isfinite(reconstruction).all():
            raise NumericalFailure("solver returned non-finite values")
        predicted = solve_result.predicted_measurement
        if predicted is None:
            predicted = operator.forward(reconstruction).detach()
        else:
            predicted = predicted.detach()
        if not torch.isfinite(predicted).all():
            raise NumericalFailure("solver returned a non-finite predicted measurement")
        heldout_predicted = None
        if heldout_case is not None:
            heldout_operator = build_operator(heldout_case, device, backend)
            heldout_predicted = heldout_operator.forward(reconstruction).detach()
            if not torch.isfinite(heldout_predicted).all():
                raise NumericalFailure(
                    "held-out evaluation returned a non-finite prediction"
                )
        runtime_seconds = time.perf_counter() - run_started
        dimension = int(case.metadata.get("dimension", 2))
        truth_available = (
            heldout_case is None
            and bool(case.metadata.get("ground_truth"))
            and (not bool(case.metadata.get("inverse_agent_sanitized", False)))
        )
        split_info = dict(split_metadata or {})
        if fit_view_indices is not None and heldout_view_indices is not None:
            split_info.setdefault(
                "fit_view_indices", [int(value) for value in fit_view_indices]
            )
            split_info.setdefault(
                "heldout_view_indices", [int(value) for value in heldout_view_indices]
            )
        effective_constraints = problem_constraints or case.metadata.get(
            "problem_constraints"
        )
        explicit_lower, explicit_upper = _explicit_problem_bounds(effective_constraints)
        if name in CONSTRAINED_TV_IDS:
            explicit_lower = 0.0
        explicit_range = None
        if effective_constraints:
            raw_range = effective_constraints.get("data_range")
            if raw_range is not None:
                try:
                    explicit_range = float(raw_range)
                except (TypeError, ValueError):
                    explicit_range = None
        data_range = explicit_range
        if data_range is None and truth_available:
            data_range = float(
                case.metadata.get("ground_truth", {}).get("data_range", 1.0)
            )
        if data_range is None or not math.isfinite(data_range) or data_range <= 0:
            data_range = 1.0
        bundle = {
            "schema_version": SCHEMA_VERSION,
            "solver": name,
            "case_id": case.case_id,
            "execution_status": solve_result.status,
            "convergence_status": solve_result.status,
            "stopping_reason": solve_result.stopping_reason,
            "dimension": dimension,
            "data_range": data_range,
            "ground_truth_available": truth_available,
            "reconstruction": reconstruction.detach().cpu(),
            "truth": (
                case.truth.detach().cpu()
                if truth_available
                else torch.zeros_like(case.truth).detach().cpu()
            ),
            "measurement": case.measurement.detach().cpu(),
            "predicted_measurement": predicted.detach().cpu(),
            "roi_mask": (
                case.roi_mask.detach().cpu()
                if truth_available and case.roi_mask is not None
                else None
            ),
            "valid_measurement_mask": (
                case.valid_measurement_mask.detach().cpu()
                if case.valid_measurement_mask is not None
                else None
            ),
        }
        if heldout_case is not None and heldout_predicted is not None:
            bundle.update(
                {
                    "heldout_measurement": heldout_case.measurement.detach().cpu(),
                    "heldout_predicted_measurement": heldout_predicted.detach().cpu(),
                    "heldout_valid_measurement_mask": (
                        heldout_case.valid_measurement_mask.detach().cpu()
                        if heldout_case.valid_measurement_mask is not None
                        else None
                    ),
                }
            )
        endpoint_fidelity = getattr(solver, "fidelity", None)
        endpoint_fidelity_before = {
            k: getattr(endpoint_fidelity, k, 0)
            for k in (
                "prox_calls",
                "inner_iterations",
                "inner_failures",
                "prox_seconds",
                "weight_multiplications",
                "mask_calls",
            )
        }
        endpoint_prediction = (
            operator.forward(reconstruction).detach()
            if requested_stopping_policy is not None
            else predicted
        )
        endpoint_report = post_run_validation(
            reconstruction,
            measurement=case.measurement,
            predicted_measurement=endpoint_prediction,
            valid_measurement_mask=case.valid_measurement_mask,
            operator=operator,
            trajectory=[record.to_dict() for record in solve_result.trajectory],
            iterations=solve_result.actual_iterations,
            max_iterations=requested_iterations or None,
            tolerance=float(validation.parameters.get("tolerance", 1e-05) or 1e-05),
            non_iterative=spec.direct,
            algorithm=name,
        )
        try:
            endpoint_confirmation = (
                _confirm_constrained_endpoint(
                    solver,
                    solve_result,
                    reconstruction,
                    endpoint_prediction,
                    case.measurement,
                    case.valid_measurement_mask,
                )
                if name in CONSTRAINED_TV_IDS
                else (
                    confirm_endpoint(
                        algorithm=name,
                        reconstruction=reconstruction,
                        measurement=case.measurement,
                        operator=operator,
                        valid_measurement_mask=case.valid_measurement_mask,
                        policy=effective_stopping_policy,
                        trajectory=[
                            record.to_dict() for record in solve_result.trajectory
                        ],
                        solver_status=solve_result.status,
                        solver_stopping_reason=solve_result.stopping_reason,
                        iterations=solve_result.actual_iterations,
                        max_iterations=requested_iterations or None,
                        parameters=validation.parameters,
                        operator_norm_estimate=operator_norm_for_diagnostics,
                        regularization_operator=getattr(
                            solver, "regularization_operator", None
                        ),
                        split_state=getattr(solver, "endpoint_state", None),
                        fidelity=getattr(solver, "fidelity", None),
                        previous_reconstruction=solve_result.previous_reconstruction,
                    )
                    if requested_stopping_policy is not None
                    or name in CONSTRAINED_TV_IDS
                    else {
                        "schema_version": "ct.endpoint_confirmation.v1",
                        "status": "not_requested",
                        "passed": True,
                        "finite": bool(torch.isfinite(reconstruction).all()),
                        "normalized_data_residual": endpoint_report.final_residual,
                    }
                )
            )
        except RuntimeError as error:
            if str(error) not in {
                "poisson_prox_inner_max_iterations",
                "poisson_prox_returned_precision",
                "joint_prox_inner_max_iterations",
                "joint_prox_returned_precision",
            }:
                raise
            endpoint_confirmation = {
                "schema_version": "ct.endpoint_confirmation.v1",
                "passed": False,
                "status": "inner_unconverged",
                "native_criterion_value": None,
                "reasons": [str(error)],
                "endpoint_diagnostics_complete": False,
            }
        if name in CONSTRAINED_TV_IDS and (not endpoint_confirmation["passed"]):
            raise NumericalFailure("constrained_endpoint_inconsistent")
        bundle["line_integral_prediction"] = endpoint_prediction.detach().cpu()
        if endpoint_fidelity is not None:
            bundle["fidelity_outputs"] = {
                k: v.detach().cpu()
                for k, v in endpoint_fidelity.measurement_diagnostics(
                    endpoint_prediction
                ).items()
            }
            endpoint_confirmation["fidelity_work"] = {
                k: getattr(endpoint_fidelity, k, 0) - v
                for k, v in endpoint_fidelity_before.items()
            }
            if callable(getattr(endpoint_fidelity, "precision_report", None)):
                endpoint_confirmation["fidelity_precision"] = (
                    endpoint_fidelity.precision_report()
                )
        if (
            endpoint_report.stopping_reason == "trajectory_ended_before_convergence"
            and solve_result.status in {"converged", "stalled"}
            and (requested_stopping_policy is not None or name in CONSTRAINED_TV_IDS)
            and endpoint_confirmation.get("passed", False)
        ):
            endpoint_report = replace(
                endpoint_report,
                status=ConvergenceStatus(solve_result.status),
                stopping_reason="native_endpoint_confirmed",
                failure_reason=None,
                reason_code="native_endpoint_confirmed",
                reason_class=status_reason_class(solve_result.status),
            )
        if solve_result.stopping_reason in {
            "joint_prox_inner_max_iterations", "joint_prox_returned_precision"
        }:
            # A finite returned iterate with an explicit proximal stopping
            # reason is not an unexplained truncated trajectory or convergence.
            endpoint_report = replace(
                endpoint_report,
                status=ConvergenceStatus(solve_result.status),
                stopping_reason=solve_result.stopping_reason,
                failure_reason=None,
                reason_code=solve_result.stopping_reason,
                reason_class=status_reason_class(solve_result.status),
            )
        if endpoint_report.status == ConvergenceStatus.NUMERICAL_ERROR:
            raise NumericalFailure(
                endpoint_report.failure_reason or endpoint_report.stopping_reason
            )
        runtime_seconds = time.perf_counter() - run_started
        convergence = solve_result.to_dict()
        convergence["trajectory"] = [
            record.to_dict() for record in solve_result.trajectory
        ]
        convergence["endpoint_validation"] = endpoint_report.to_dict()
        convergence["endpoint_confirmation"] = endpoint_confirmation
        convergence["requested_policy"] = requested_stopping_policy
        convergence["effective_policy"] = effective_stopping_policy
        if solve_result.status == "converged" and (
            not endpoint_confirmation.get("passed", False)
        ):
            convergence["status"] = ConvergenceStatus.NUMERICAL_ERROR.value
            convergence["stopping_reason"] = "endpoint_confirmation_failed"
            convergence["reason_code"] = "endpoint_confirmation_failed"
            convergence["reason_class"] = status_reason_class(
                ConvergenceStatus.NUMERICAL_ERROR
            )
            convergence["failure_reason"] = (
                "; ".join(endpoint_confirmation.get("reasons", ()))
                or "endpoint confirmation failed"
            )
        if (
            name in {"fbp", "fdk"}
            and convergence.get("status") == ConvergenceStatus.COMPLETED_VALID.value
        ):
            convergence["status"] = ConvergenceStatus.COMPLETED_VALID.value
            convergence["stopping_reason"] = convergence.get(
                "stopping_reason", "direct_reconstruction_valid"
            )
            convergence["reason_code"] = convergence["stopping_reason"]
            convergence["reason_class"] = status_reason_class(
                ConvergenceStatus.COMPLETED_VALID
            )
        convergence["converged_at_budget_boundary"] = bool(
            endpoint_confirmation.get("converged_at_budget_boundary", False)
        )
        convergence_status = normalize_status(
            convergence["status"],
            algorithm=name,
            iterations=solve_result.actual_iterations,
            max_iterations=requested_iterations or None,
            direct=name in {"fbp", "fdk"},
        ).value
        convergence["status"] = convergence_status
        convergence["convergence_status"] = convergence_status
        convergence.setdefault("reason_code", convergence.get("stopping_reason"))
        convergence["reason_class"] = status_reason_class(convergence_status)
        convergence_reason = str(convergence["stopping_reason"])
        execution_status = (
            "completed"
            if requested_stopping_policy is not None
            and convergence_status
            not in {
                "diverged",
                "numerical_error",
                "cancelled",
                "resource_exhausted",
                "invalid_parameters",
            }
            else convergence_status
        )
        bundle["execution_status"] = execution_status
        bundle["convergence_status"] = convergence_status
        bundle["stopping_reason"] = convergence_reason
        objective = solve_result.final_objective
        if (
            objective is None
            and getattr(solver, "fidelity", None) is None
            and (name not in {"fista", "admm", "pdhg"} | CONSTRAINED_TV_IDS)
        ):
            objective = float(
                0.5 * (predicted - case.measurement).square().sum().item()
            )
        diagnostics = {
            "schema_version": SCHEMA_VERSION,
            "solver": name,
            "algorithm": name,
            "status": convergence_status,
            "convergence_status": convergence_status,
            "regularizer": validation.parameters.get(
                "regularizer", spec.regularizers[0] if spec.regularizers else None
            ),
            "initialization": spec.initialization,
            "dtype": str(case.measurement.dtype),
            "device": str(case.measurement.device),
            "reconstruction_device": str(solve_result.reconstruction.device),
            "convergence_criteria": list(spec.convergence_criteria),
            "trajectory_available": solve_result.trajectory_available,
            "parameters": validation.parameters,
            "parameter_sources": validation.sources,
            "parameter_overrides": dict(parameter_overrides or {}),
            "parameter_validation": validation.to_dict(),
            "estimates": validation.estimates,
            "observation_domain": domain,
            "geometry_type": str(case.geometry.get("type", "")),
            "ground_truth_available": truth_available,
            "execution_status": execution_status,
            "reason_code": convergence.get("reason_code", convergence_reason),
            "reason_class": status_reason_class(convergence_status),
            "stopping_reason": convergence_reason,
            "iterations_requested": requested_iterations,
            "iterations_completed": solve_result.actual_iterations,
            "objective": objective,
            "final_residual": (
                solve_result.final_residual
                if solve_result.final_residual is not None
                else endpoint_report.final_residual
            ),
            "final_objective": objective,
            "convergence": convergence,
            "endpoint_confirmation": endpoint_confirmation,
            "resources": {"runtime_seconds": runtime_seconds},
            "projection_split": split_info or None,
            "fixed_compute": bool(fixed_compute),
            "requested_stopping_policy": requested_stopping_policy,
            "effective_stopping_policy": effective_stopping_policy,
            "legacy_tolerance_status": (
                "superseded_by_stopping_policy"
                if requested_stopping_policy is not None
                else "active"
            ),
            "constraints": {"min_value": explicit_lower, "max_value": explicit_upper},
            "masks": {
                "roi_available": case.roi_mask is not None,
                "valid_measurement_available": case.valid_measurement_mask is not None,
            },
        }
        metrics = _metrics(bundle, runtime_seconds)
        if convergence_status in {"diverged", "numerical_error"}:
            record = _failure(
                destination,
                convergence_status,
                NumericalFailure(convergence_reason),
                algorithm=name,
                reason_code=convergence.get("reason_code", convergence_reason),
            )
            return {
                "status": convergence_status,
                "execution_status": execution_status,
                "convergence_status": convergence_status,
                "stopping_reason": convergence_reason,
                "output_dir": destination,
                "metrics": metrics,
                "bundle": bundle,
                "diagnostics": diagnostics,
                **record,
            }
        return {
            "status": "success",
            "execution_status": execution_status,
            "convergence_status": convergence_status,
            "stopping_reason": convergence_reason,
            "reason_code": convergence.get("reason_code", convergence_reason),
            "reason_class": status_reason_class(convergence_status),
            "output_dir": destination,
            "metrics": metrics,
            "diagnostics": diagnostics,
            "bundle": bundle,
        }
    except InvalidParametersError as error:
        invalid_resources = {}
        return _structured_failure(
            destination,
            status="invalid_parameters",
            error=error,
            algorithm=locals().get("name", solver_name),
            reason_code=(
                validation.reason_codes[0]
                if validation is not None and validation.reason_codes
                else None
            ),
            parameters=locals().get("parameters")
            or (validation.parameters if validation is not None else None),
            validation=locals().get("validation"),
            resources=invalid_resources,
        )
    except BackendUnavailable as error:
        return _structured_failure(
            destination,
            status="unavailable",
            error=error,
            algorithm=locals().get("name", solver_name),
            resources={},
        )
    except NumericalFailure as error:
        return _structured_failure(
            destination,
            status="numerical_error",
            error=error,
            algorithm=locals().get("name", solver_name),
            reason_code="invalid_solver_output",
            solve_result=solve_result,
            resources={},
        )

    finally:
        for owned_operator in (heldout_operator, operator):
            close = getattr(owned_operator, "close", None)
            if close is not None:
                close()

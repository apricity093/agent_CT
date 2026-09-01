"""Production runtime for the invct command-line interface."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from inv_framework.benchmarks import (
    BENCHMARK_PROTOCOLS,
    BenchmarkBudget,
    BenchmarkResult,
    check_fairness,
    EQUAL_OPERATOR_CALLS,
    ORACLE_CALIBRATION,
    enforce_budget,
    list_ct_cases,
    load_ct_case,
    make_heldout_projection_split,
    restrict_ct_case,
    pareto_front,
)
from inv_framework.convergence import ConvergenceStatus, post_run_validation
from inv_framework.instrumentation import (
    CountingLinearOperator,
    OperatorBudgetExceeded,
    start_memory_tracing,
)
from inv_framework.operators.ct import ParallelBeamRadon2D
from inv_framework.regularizers import TVRegularizer
from inv_framework.solvers import (
    CGLSSolver,
    FDKSolver,
    FBPSolver,
    LSQRSolver,
    LandweberSolver,
    MLEMSolver,
    OSEMSolver,
    OSSARTSolver,
    SARTSolver,
    SIRTSolver,
    TVFISTASolver,
    TikhonovSolver,
    SolveControl,
)
from inv_framework.solvers.specs import (
    CTAlgorithmSpec,
    SOLVER_SPECS,
    ParameterValidationResult,
    regularizer_records,
    solver_records as _solver_records,
    validate_compatibility,
    validate_parameter_values,
)
from inv_framework.utils.metrics import ssim


SCHEMA_VERSION = 1
RESULT_FILENAME = "reconstruction.pt"


# Kept as a compatibility alias for callers that imported the old runtime
# type.  The actual registry lives in ``solvers.specs``.
SolverSpec = CTAlgorithmSpec


class ConfigError(ValueError):
    """Raised for invalid user-authored CLI configuration."""


class BackendUnavailable(RuntimeError):
    """Raised when an optional numerical backend cannot run a requested job."""


class NumericalFailure(RuntimeError):
    """Raised when a solver returns an invalid numerical result."""


def solver_records() -> list[dict[str, Any]]:
    return _solver_records()


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


def load_algorithm_config(path: str | Path, expected_solver: str | None = None) -> tuple[str, dict[str, Any], Path]:
    payload, source = load_yaml(path)
    _check_keys(payload, {"schema_version", "name", "parameters"}, "algorithm config")
    if payload.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ConfigError(f"algorithm config schema_version must be {SCHEMA_VERSION}.")
    name = payload.get("name")
    if not isinstance(name, str) or name not in SOLVER_SPECS:
        raise ConfigError(f"unknown solver in algorithm config: {name!r}")
    if expected_solver is not None and name != expected_solver:
        raise ConfigError(f"algorithm config names {name!r}, but command requested {expected_solver!r}.")
    parameters = _require_mapping(payload.get("parameters", {}), "algorithm parameters")
    unknown = sorted(set(parameters) - set(SOLVER_SPECS[name].parameter_names))
    if unknown:
        raise ConfigError(f"unsupported parameter(s) for {name}: {', '.join(unknown)}")
    _validate_parameter_types(parameters)
    validation = validate_parameter_values(name, parameters)
    if validation.errors:
        raise ConfigError(f"invalid parameters for {name}: {'; '.join(validation.errors)}")
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
    explicit = metadata.get("observation_domain")
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

    domain_shape = tuple(int(value) for value in operator.domain_shape)
    sample_size = math.prod(domain_shape)
    seed = torch.linspace(
        0.5,
        1.5,
        steps=sample_size,
        dtype=reference.dtype,
        device=reference.device,
    ).reshape((1, *domain_shape))
    denominator = seed.reshape(seed.shape[0], -1).norm(dim=1)
    vector = seed / denominator.reshape((1,) + (1,) * len(domain_shape)).clamp_min(1e-12)
    for _ in range(max(1, int(num_iterations))):
        next_vector = operator.adjoint(operator.forward(vector))
        norm = next_vector.reshape(next_vector.shape[0], -1).norm(dim=1)
        if bool(torch.all(norm <= 1e-12)):
            return 0.0
        vector = next_vector / norm.reshape((next_vector.shape[0],) + (1,) * len(domain_shape)).clamp_min(1e-12)
    image = operator.adjoint(operator.forward(vector))
    value = (vector * image).reshape(vector.shape[0], -1).sum(dim=1).clamp_min(0.0).max()
    return float(value.item())


def validate_parameters(
    name: str,
    parameters: Mapping[str, Any] | None = None,
    *,
    case: Any | None = None,
    operator: Any | None = None,
    observation_domain: str | None = None,
    problem_constraints: Mapping[str, Any] | None = None,
    parameter_sources: Mapping[str, str] | None = None,
) -> ParameterValidationResult:
    """Validate one solver against public case metadata and operator bounds."""

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
    supplied = dict(parameters or {})
    sources: dict[str, str] = {str(key): str(value) for key, value in (parameter_sources or {}).items()}
    lower, upper = _explicit_problem_bounds(problem_constraints)
    if lower is not None and "min_value" not in supplied:
        supplied["min_value"] = lower
        sources["min_value"] = "problem_constraint"
    if upper is not None and "max_value" not in supplied:
        supplied["max_value"] = upper
        sources["max_value"] = "problem_constraint"

    estimated = None
    estimates: dict[str, Any] = {}
    if operator is not None and name in {"landweber", "tv_fista"}:
        iterations = supplied.get("power_iterations", 12)
        reference = getattr(case, "measurement", None)
        if not isinstance(reference, torch.Tensor):
            reference = torch.ones((1, *tuple(operator.range_shape)), dtype=torch.float32)
        try:
            estimated = estimate_lipschitz_squared(operator, reference, int(iterations))
            estimates["operator_norm_estimator"] = "power_iteration"
        except Exception as error:
            estimates["operator_norm_error"] = str(error)

    return validate_parameter_values(
        name,
        supplied,
        views=views,
        geometry_type=geometry_type,
        dimension=int(dimension) if dimension is not None else None,
        observation_domain=domain,
        estimated_lipschitz=estimated,
        parameter_estimates=estimates,
        parameter_sources=sources,
    )


def validate_solver_case(name: str, case: Any, observation_domain: str | None = None) -> SolverSpec:
    if name not in SOLVER_SPECS:
        raise ConfigError(f"unknown solver: {name!r}")
    spec = SOLVER_SPECS[name]
    dimension = int(case.metadata.get("dimension", len(case.geometry.get("domain_shape", ()))))
    geometry_type = str(case.geometry.get("type", ""))
    if dimension not in spec.dimensions or geometry_type not in spec.geometry_types:
        raise ConfigError(
            f"solver {name!r} supports dimension={spec.dimensions} and geometry="
            f"{spec.geometry_types}, but case {case.case_id!r} has dimension={dimension} "
            f"and geometry={geometry_type!r}."
        )
    domain = observation_domain or _case_observation_domain(case)
    if name in {"mlem", "osem"}:
        measurement_meta = dict(case.metadata.get("measurement", {}) or {})
        observation_model = str(
            measurement_meta.get("observation_model", case.metadata.get("observation_model", ""))
        ).lower()
        # Nonnegative values alone do not establish the Poisson emission
        # model.  X-ray line integrals, including positive ones, are not
        # interchangeable with counts/intensities for EM updates.
        if str(measurement_meta.get("kind", "")).lower() == "line_integral" and observation_model not in {
            "emission",
            "emission_counts",
            "poisson_emission",
        }:
            raise ConfigError(
                f"solver {name!r} requires an explicit Poisson emission/count "
                "observation_model; X-ray line-integral data are incompatible"
            )
    incompatibilities = validate_compatibility(
        name,
        geometry_type=geometry_type,
        dimension=dimension,
        observation_domain=domain,
    )
    if incompatibilities:
        raise ConfigError("; ".join(incompatibilities))
    return spec


def build_operator(case: Any, device: str | torch.device):
    geometry_type = case.geometry.get("type")
    resolved = torch.device(device)
    if geometry_type == "parallel_2d":
        domain = tuple(int(v) for v in case.geometry["domain_shape"])
        if len(domain) != 3 or domain[-1] != domain[-2]:
            raise ConfigError("ParallelBeamRadon2D requires square (C,H,W) domain_shape.")
        angles = torch.as_tensor(case.geometry["angles_rad"], dtype=torch.float32, device=resolved)
        return ParallelBeamRadon2D(domain[-1], angles=angles, device=str(resolved), in_channels=domain[0])
    if geometry_type == "cone_3d":
        try:
            from inv_framework.operators.ct import ASTRAFDKOperator3D
            from inv_framework.operators.ct import astra_adapter as astra_backend
        except ImportError as error:
            raise BackendUnavailable("astra-toolbox is not installed.") from error
        if not astra_backend._HAS_ASTRA:
            raise BackendUnavailable("astra-toolbox is not installed.")
        astra = astra_backend.astra
        if not astra.use_cuda() or not torch.cuda.is_available() or resolved.type != "cuda":
            raise BackendUnavailable("FDK requires --device cuda, PyTorch CUDA, and ASTRA CUDA.")
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


def build_solver(name: str, parameters: Mapping[str, Any], case: Any):
    params = dict(parameters)
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
    classes = {
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
        "fdk": FDKSolver,
    }
    if name == "tv_fista":
        tv = TVRegularizer(
            mode=params.pop("tv_mode", "isotropic"),
            num_iterations=int(params.pop("tv_num_iterations", 50)),
            tolerance=float(params.pop("tv_tolerance", 1e-5)),
        )
        return TVFISTASolver(regularizer=tv, **params)
    try:
        return classes[name](**params)
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


def _resource_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    """Return phase-local operator counters without losing aggregate totals."""

    result: dict[str, Any] = {}
    for name in ("forward_calls", "adjoint_calls", "total_operator_calls"):
        if before.get(name) is not None and after.get(name) is not None:
            result[name] = int(after[name]) - int(before[name])
    for name in ("forward_seconds", "adjoint_seconds", "operator_runtime_seconds"):
        if before.get(name) is not None and after.get(name) is not None:
            result[name] = max(0.0, float(after[name]) - float(before[name]))
    return result


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
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _git_provenance(root: str | Path | None = None) -> dict[str, Any]:
    """Return commit plus a digest of tracked and untracked checkout changes.

    ``git rev-parse HEAD`` alone is insufficient for an external checkout
    during development: the CT integration may live in a dirty diff or in
    newly-created files.  This manifest is intentionally content-addressed
    and does not copy the diff into experiment artifacts.
    """

    directory = Path(root or Path.cwd()).expanduser().resolve()
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=directory,
            capture_output=True, text=True, check=False,
        )
        status_result = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=directory,
            capture_output=True, text=True, check=False,
        )
        diff_result = subprocess.run(
            ["git", "diff", "--binary", "HEAD"], cwd=directory,
            capture_output=True, check=False,
        )
    except OSError:
        return {"available": False, "commit": None, "dirty": None, "patch_sha256": None, "changed_files": []}

    status_text = status_result.stdout or ""
    changed: list[dict[str, Any]] = []
    for line in status_text.splitlines():
        if len(line) < 4:
            continue
        state, relative = line[:2], line[3:]
        # Rename records use ``old -> new``; hashing the new file is the
        # useful provenance for the checkout that will actually be imported.
        relative = relative.split(" -> ")[-1]
        path = directory / relative
        digest = None
        if path.is_file():
            try:
                digest = _sha256(path)
            except OSError:
                digest = None
        changed.append({"path": relative, "status": state, "sha256": digest})
    patch_material = (diff_result.stdout or b"") + json.dumps(changed, sort_keys=True).encode("utf-8")
    patch_hash = hashlib.sha256(patch_material).hexdigest()
    return {
        "available": commit_result.returncode == 0,
        "commit": (commit_result.stdout or "").strip() or None,
        "dirty": bool(changed),
        "patch_sha256": patch_hash,
        "changed_files": changed,
    }


def _load_parameter_sources(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ConfigError(f"parameter source file does not exist: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ConfigError(f"cannot parse parameter source file {source}: {error}") from error
    if not isinstance(payload, Mapping):
        raise ConfigError("parameter source file must contain a JSON object")
    return {str(key): str(value) for key, value in payload.items()}


def _prepare_output(path: str | Path, overwrite: bool) -> Path:
    destination = Path(path).expanduser().resolve()
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise ConfigError(f"output path must be a directory and not a symlink: {destination}")
    if destination.exists() and any(destination.iterdir()):
        if not overwrite:
            raise ConfigError(f"output directory is not empty: {destination}; use --overwrite")
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _plot_comparison(
    path: Path,
    truth: torch.Tensor,
    reconstruction: torch.Tensor,
    dimension: int,
    *,
    truth_available: bool = True,
) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    reconstruction_image = reconstruction[0, 0] if dimension == 2 else reconstruction[0, reconstruction.shape[1] // 2]
    if truth_available:
        truth_image = truth[0, 0] if dimension == 2 else truth[0, truth.shape[1] // 2]
        images = (truth_image, reconstruction_image)
        titles = ("Reference", "Reconstruction")
    else:
        images = (reconstruction_image,)
        titles = ("Reconstruction (truth withheld)",)
    vmin = float(torch.stack([image.min() for image in images]).min().item())
    vmax = float(torch.stack([image.max() for image in images]).max().item())
    if vmin == vmax:
        vmax = vmin + 1.0
    figure, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4), squeeze=False, constrained_layout=True)
    for axis, image, title in zip(axes[0], images, titles):
        axis.imshow(image.detach().cpu().numpy(), cmap="gray", vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.axis("off")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _write_checksums(destination: Path) -> None:
    files = sorted(path for path in destination.iterdir() if path.is_file() and path.name != "artifacts.sha256")
    text = "".join(f"{_sha256(path)}  {path.name}\n" for path in files)
    (destination / "artifacts.sha256").write_text(text, encoding="ascii")


def _failure(destination: Path, status: str, error: BaseException) -> dict[str, Any]:
    record = {"schema_version": SCHEMA_VERSION, "status": status, "error_type": type(error).__name__, "message": str(error)}
    _write_json(destination / "failure.json", record)
    (destination / "failure_report.md").write_text(
        f"# 运行失败\n\n- 状态：`{status}`\n- 类型：`{type(error).__name__}`\n- 原因：{error}\n",
        encoding="utf-8",
    )
    (destination / "traceback.log").write_text("".join(traceback.format_exception(error)), encoding="utf-8")
    _write_checksums(destination)
    return record


def run_case(
    solver_name: str,
    case_id: str,
    config_path: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cpu",
    data_root: str | Path | None = None,
    overwrite: bool = False,
    observation_domain: str | None = None,
    problem_constraints: Mapping[str, Any] | None = None,
    max_iterations: int | None = None,
    max_forward_calls: int | None = None,
    max_adjoint_calls: int | None = None,
    parameter_sources_path: str | Path | None = None,
    parameter_overrides: Mapping[str, Any] | None = None,
    parameter_sources: Mapping[str, str] | None = None,
    fit_view_indices: Sequence[int] | None = None,
    heldout_view_indices: Sequence[int] | None = None,
    split_metadata: Mapping[str, Any] | None = None,
    fixed_compute: bool = False,
) -> dict[str, Any]:
    destination = _prepare_output(output_dir, overwrite)
    diagnostics: dict[str, Any] = {}
    counted_operator: CountingLinearOperator | None = None
    try:
        name, parameters, config_source = load_algorithm_config(config_path, solver_name)
        if parameter_overrides:
            unknown = sorted(set(parameter_overrides) - set(SOLVER_SPECS[name].parameter_names))
            if unknown:
                raise ConfigError(
                    f"unsupported parameter override(s) for {name}: {', '.join(unknown)}"
                )
            parameters = {**parameters, **dict(parameter_overrides)}
            _validate_parameter_types(parameters)
            override_validation = validate_parameter_values(name, parameters)
            if override_validation.errors:
                raise ConfigError(
                    f"invalid parameter overrides for {name}: "
                    f"{'; '.join(override_validation.errors)}"
                )
        parameter_source_map = _load_parameter_sources(parameter_sources_path)
        parameter_source_map.update({str(key): str(value) for key, value in (parameter_sources or {}).items()})
        for parameter_name in parameters:
            parameter_source_map.setdefault(parameter_name, "repository_config@sha256")
        full_case = load_ct_case(case_id, data_root=data_root, device=device)
        if (fit_view_indices is None) != (heldout_view_indices is None):
            raise ConfigError("fit_view_indices and heldout_view_indices must be supplied together")
        case = full_case
        heldout_case = None
        if fit_view_indices is not None and heldout_view_indices is not None:
            fit = tuple(int(value) for value in fit_view_indices)
            heldout = tuple(int(value) for value in heldout_view_indices)
            if set(fit).intersection(heldout):
                raise ConfigError("fit and held-out view indices must be disjoint")
            case = restrict_ct_case(full_case, fit, partition="fit")
            heldout_case = restrict_ct_case(full_case, heldout, partition="heldout")
        domain = observation_domain or _case_observation_domain(case)
        spec = validate_solver_case(name, case, domain)
        operator = build_operator(case, device)
        start_memory_tracing()
        counted_operator = CountingLinearOperator(
            operator,
            max_forward_calls=max_forward_calls,
            max_adjoint_calls=max_adjoint_calls,
        )
        run_started = time.perf_counter()
        initial_counters = counted_operator.stats()
        validation = validate_parameters(
            name,
            parameters,
            case=case,
            operator=counted_operator,
            observation_domain=domain,
            problem_constraints=problem_constraints or case.metadata.get("problem_constraints"),
            parameter_sources=parameter_source_map,
        )
        validation_counters = counted_operator.stats()
        if validation.errors:
            raise ConfigError(f"invalid parameters for {name}: {'; '.join(validation.errors)}")
        requested_iterations = int(validation.parameters.get("num_iterations", 0) or 0)
        if max_iterations is not None and requested_iterations > int(max_iterations):
            raise ConfigError(
                f"config requests {requested_iterations} iterations; budget allows {int(max_iterations)}"
            )
        solver = build_solver(name, validation.parameters, case)
        # Detailed solver paths expose the actual loop boundary.  The
        # configured iteration count remains the execution upper bound; it is
        # never converted into a convergence claim.
        detail_tolerance = None
        if fixed_compute and name not in {"fbp", "fdk"}:
            detail_tolerance = 0.0
        elif name not in {"fbp", "fdk", "cgls", "lsqr"}:
            detail_tolerance = float(validation.parameters.get("tolerance", 1e-5))
        control = SolveControl(
            max_iterations=max(1, requested_iterations or 1),
            tolerance=detail_tolerance,
            metadata={"source": "ct_runtime", "solver": name},
        )
        solve_result = solver.solve_detailed(case.measurement, counted_operator, control=control)
        solver_counters = counted_operator.stats()
        reconstruction = solve_result.reconstruction
        if tuple(reconstruction.shape) != tuple(case.truth.shape):
            raise NumericalFailure(f"solver returned shape {tuple(reconstruction.shape)}, expected {tuple(case.truth.shape)}")
        if not torch.isfinite(reconstruction).all():
            raise NumericalFailure("solver returned non-finite values")
        predicted = solve_result.predicted_measurement
        if predicted is None:
            predicted = counted_operator.forward(reconstruction).detach()
        else:
            predicted = predicted.detach()
        if not torch.isfinite(predicted).all():
            raise NumericalFailure("solver returned a non-finite predicted measurement")
        heldout_predicted = None
        if heldout_case is not None:
            heldout_operator = CountingLinearOperator(
                build_operator(heldout_case, device),
                max_forward_calls=max_forward_calls,
                max_adjoint_calls=max_adjoint_calls,
                counters=counted_operator.counters,
            )
            heldout_predicted = heldout_operator.forward(reconstruction).detach()
            if not torch.isfinite(heldout_predicted).all():
                raise NumericalFailure("held-out evaluation returned a non-finite prediction")
        final_evaluation_counters = counted_operator.stats()
        runtime_seconds = time.perf_counter() - run_started
        dimension = int(case.metadata.get("dimension", 2))
        truth_available = (
            heldout_case is None
            and bool(case.metadata.get("ground_truth"))
            and not bool(case.metadata.get("inverse_agent_sanitized", False))
        )
        split_info = dict(split_metadata or {})
        if fit_view_indices is not None and heldout_view_indices is not None:
            split_info.setdefault("fit_view_indices", [int(value) for value in fit_view_indices])
            split_info.setdefault("heldout_view_indices", [int(value) for value in heldout_view_indices])
        effective_constraints = problem_constraints or case.metadata.get("problem_constraints")
        explicit_lower, explicit_upper = _explicit_problem_bounds(effective_constraints)
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
            data_range = float(case.metadata.get("ground_truth", {}).get("data_range", 1.0))
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
            "truth": case.truth.detach().cpu() if truth_available else torch.zeros_like(case.truth).detach().cpu(),
            "measurement": case.measurement.detach().cpu(),
            "predicted_measurement": predicted.detach().cpu(),
            "roi_mask": case.roi_mask.detach().cpu() if truth_available and case.roi_mask is not None else None,
            "valid_measurement_mask": case.valid_measurement_mask.detach().cpu() if case.valid_measurement_mask is not None else None,
        }
        if heldout_case is not None and heldout_predicted is not None:
            bundle.update({
                "heldout_measurement": heldout_case.measurement.detach().cpu(),
                "heldout_predicted_measurement": heldout_predicted.detach().cpu(),
                "heldout_valid_measurement_mask": (
                    heldout_case.valid_measurement_mask.detach().cpu()
                    if heldout_case.valid_measurement_mask is not None else None
                ),
            })
        endpoint_report = post_run_validation(
            reconstruction,
            measurement=case.measurement,
            predicted_measurement=predicted,
            valid_measurement_mask=case.valid_measurement_mask,
            operator=counted_operator,
            trajectory=[record.to_dict() for record in solve_result.trajectory],
            iterations=solve_result.actual_iterations,
            max_iterations=int(max_iterations) if max_iterations is not None else (requested_iterations or None),
            tolerance=float(validation.parameters.get("tolerance", 1e-5) or 1e-5),
            non_iterative=spec.direct,
            algorithm=name,
        )
        if endpoint_report.status == ConvergenceStatus.NUMERICAL_FAILURE:
            raise NumericalFailure(endpoint_report.failure_reason or endpoint_report.stopping_reason)
        counters = counted_operator.stats()
        phase_resources = {
            "parameter_estimation": _resource_delta(initial_counters, validation_counters),
            "solver": _resource_delta(validation_counters, solver_counters),
            "final_evaluation": _resource_delta(solver_counters, final_evaluation_counters),
        }
        convergence = solve_result.to_dict()
        convergence["trajectory"] = [record.to_dict() for record in solve_result.trajectory]
        convergence["endpoint_validation"] = endpoint_report.to_dict()
        objective = solve_result.final_objective
        if objective is None:
            objective = float(0.5 * (predicted - case.measurement).square().sum().item())
        diagnostics = {
            "schema_version": SCHEMA_VERSION,
            "solver": name,
            "algorithm": name,
            "regularizer": spec.regularizers[0] if spec.regularizers else None,
            "initialization": spec.initialization,
            "dtype": str(case.measurement.dtype),
            "device": str(case.measurement.device),
            "convergence_criteria": list(spec.convergence_criteria),
            "trajectory_available": solve_result.trajectory_available,
            "parameters": validation.parameters,
            "parameter_sources": validation.sources,
            "parameter_overrides": dict(parameter_overrides or {}),
            "config_sha256": _sha256(config_source),
            "parameter_validation": validation.to_dict(),
            "estimates": validation.estimates,
            "observation_domain": domain,
            "geometry_type": str(case.geometry.get("type", "")),
            "ground_truth_available": truth_available,
            "execution_status": solve_result.status,
            "stopping_reason": solve_result.stopping_reason,
            "iterations_requested": requested_iterations,
            "iterations_completed": solve_result.actual_iterations,
            "objective": objective,
            "final_residual": solve_result.final_residual if solve_result.final_residual is not None else endpoint_report.final_residual,
            "final_objective": objective,
            "convergence": convergence,
            "resources": {
                **dict(solve_result.resources),
                # The aggregate counter snapshot includes endpoint and
                # held-out evaluation calls; it must override the solver's
                # earlier snapshot for the top-level resource record.
                **counters,
                "runtime_seconds": runtime_seconds,
                "phases": phase_resources,
            },
            "projection_split": split_info or None,
            "fixed_compute": bool(fixed_compute),
            "constraints": {"min_value": explicit_lower, "max_value": explicit_upper},
            "masks": {
                "roi_available": case.roi_mask is not None,
                "valid_measurement_available": case.valid_measurement_mask is not None,
            },
        }
        torch.save(bundle, destination / RESULT_FILENAME)
        metrics = _metrics(bundle, runtime_seconds)
        _write_json(destination / "metrics.json", metrics)
        _write_json(destination / "diagnostics.json", diagnostics)
        _plot_comparison(destination / "comparison.png", case.truth, reconstruction, dimension, truth_available=truth_available)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "success",
            "solver": spec.to_dict(),
            "parameters": validation.parameters,
            "parameter_sources": validation.sources,
            "parameter_overrides": dict(parameter_overrides or {}),
            "parameter_validation": validation.to_dict(),
            "case_id": case.case_id,
            "case_source": str(case.source_path) if case.source_path else None,
            "data_root": str(Path(data_root).resolve()) if data_root else None,
            "config_source": str(config_source),
            "projection_split": split_info or None,
            "observation_domain": domain,
            "ground_truth_available": truth_available,
            "execution_status": solve_result.status,
            "convergence_status": solve_result.status,
            "stopping_reason": solve_result.stopping_reason,
            "diagnostics": "diagnostics.json",
            "device": str(device),
            "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "git_revision": _git_revision()},
            "repository_provenance": _git_provenance(Path(__file__).resolve().parents[1]),
            "artifacts": {RESULT_FILENAME: _sha256(destination / RESULT_FILENAME), "metrics.json": _sha256(destination / "metrics.json"), "diagnostics.json": _sha256(destination / "diagnostics.json"), "comparison.png": _sha256(destination / "comparison.png")},
        }
        _write_json(destination / "manifest.json", manifest)
        _write_checksums(destination)
        execution_status = str(solve_result.status)
        if execution_status in {"diverged", "numerical_failure"}:
            record = _failure(destination, execution_status, NumericalFailure(solve_result.stopping_reason))
            _write_json(destination / "diagnostics.json", diagnostics)
            _write_checksums(destination)
            return {"status": execution_status, "output_dir": destination, "metrics": metrics, "diagnostics": diagnostics, **record}
        return {
            # Keep the old successful-process API stable.  Consumers needing
            # the actual solver termination use execution_status or the
            # nested convergence report; max_iterations is never rewritten
            # as converged.
            "status": "success",
            "execution_status": execution_status,
            "output_dir": destination,
            "metrics": metrics,
            "diagnostics": diagnostics,
        }
    except BackendUnavailable as error:
        return {"status": "unavailable", "output_dir": destination, **_failure(destination, "unavailable", error)}
    except OperatorBudgetExceeded as error:
        record = _failure(destination, "resource_exhausted", error)
        diagnostics = {
            "schema_version": SCHEMA_VERSION,
            "status": "resource_exhausted",
            "convergence": {"status": ConvergenceStatus.NUMERICAL_FAILURE.value, "stopping_reason": "operator_call_budget_exhausted"},
            "resources": counted_operator.stats() if counted_operator is not None else {},
        }
        _write_json(destination / "diagnostics.json", diagnostics)
        _write_checksums(destination)
        return {"status": "resource_exhausted", "output_dir": destination, **record, "diagnostics": diagnostics}
    except NumericalFailure as error:
        record = _failure(destination, "numerical_failure", error)
        diagnostics = {
            "schema_version": SCHEMA_VERSION,
            "status": "numerical_failure",
            "convergence": {
                "status": ConvergenceStatus.NUMERICAL_FAILURE.value,
                "stopping_reason": "invalid_solver_output",
                "failure_reason": str(error),
            },
            "resources": counted_operator.stats() if counted_operator is not None else {},
        }
        _write_json(destination / "diagnostics.json", diagnostics)
        _write_checksums(destination)
        return {"status": "numerical_failure", "output_dir": destination, **record, "diagnostics": diagnostics}
    except Exception as error:
        _failure(destination, "failed", error)
        raise


def _load_result(path: Path) -> dict[str, Any]:
    try:
        bundle = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        bundle = torch.load(path, map_location="cpu")
    required = {"solver", "case_id", "dimension", "data_range", "reconstruction", "truth", "measurement", "predicted_measurement"}
    if not isinstance(bundle, Mapping) or not required.issubset(bundle):
        raise ConfigError(f"invalid reconstruction artifact: {path}")
    return dict(bundle)


def load_protocol(path: str | Path) -> tuple[dict[str, Any], Path]:
    payload, source = load_yaml(path)
    _check_keys(payload, {"schema_version", "name", "expected_statuses", "required_metrics", "thresholds", "min_records"}, "protocol")
    if payload.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ConfigError(f"protocol schema_version must be {SCHEMA_VERSION}.")
    statuses = payload.get("expected_statuses", ["success"])
    required = payload.get("required_metrics", ["relative_error", "rmse", "psnr", "ssim", "data_residual"])
    thresholds = _require_mapping(payload.get("thresholds", {}), "protocol thresholds")
    if not isinstance(statuses, list) or not all(isinstance(v, str) for v in statuses):
        raise ConfigError("protocol expected_statuses must be a list of strings.")
    if not isinstance(required, list) or not all(isinstance(v, str) for v in required):
        raise ConfigError("protocol required_metrics must be a list of strings.")
    normalized: dict[str, dict[str, float]] = {}
    for metric, limits in thresholds.items():
        limits = _require_mapping(limits, f"threshold {metric}")
        _check_keys(limits, {"min", "max"}, f"threshold {metric}")
        if not limits:
            raise ConfigError(f"threshold {metric} must define min or max.")
        normalized[str(metric)] = {key: float(value) for key, value in limits.items()}
    payload["expected_statuses"] = statuses
    payload["required_metrics"] = required
    payload["thresholds"] = normalized
    payload["min_records"] = int(payload.get("min_records", 1))
    return payload, source


def evaluate_metrics(records: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    allowed = set(protocol["expected_statuses"])
    for index, original in enumerate(records):
        record = dict(original)
        failures: list[str] = []
        if record.get("status") not in allowed:
            failures.append(f"status {record.get('status')!r} not in {sorted(allowed)}")
        for metric in protocol["required_metrics"]:
            if record.get(metric) is None:
                failures.append(f"missing metric {metric}")
        for metric, limits in protocol["thresholds"].items():
            value = record.get(metric)
            if value is None:
                failures.append(f"missing threshold metric {metric}")
                continue
            numeric = float(value)
            if "min" in limits and numeric < limits["min"]:
                failures.append(f"{metric}={numeric:.6g} < {limits['min']:.6g}")
            if "max" in limits and numeric > limits["max"]:
                failures.append(f"{metric}={numeric:.6g} > {limits['max']:.6g}")
        checks.append({"index": index, "solver": record.get("solver"), "case_id": record.get("case_id"), "passed": not failures, "failures": failures})
    minimum_met = len(records) >= int(protocol["min_records"])
    return {"schema_version": SCHEMA_VERSION, "passed": bool(records) and minimum_met and all(item["passed"] for item in checks), "record_count": len(records), "minimum_records_met": minimum_met, "checks": checks}


def evaluate_run(run_dir: str | Path, protocol_path: str | Path) -> dict[str, Any]:
    directory = Path(run_dir).expanduser().resolve()
    bundle = _load_result(directory / RESULT_FILENAME)
    runtime = None
    metrics_path = directory / "metrics.json"
    if metrics_path.is_file():
        runtime = json.loads(metrics_path.read_text(encoding="utf-8")).get("runtime_seconds")
    metrics = _metrics(bundle, runtime)
    protocol, source = load_protocol(protocol_path)
    evaluation = evaluate_metrics([metrics], protocol)
    evaluation.update({"protocol": str(source), "metrics": metrics})
    _write_json(directory / "evaluation.json", evaluation)
    status = "通过" if evaluation["passed"] else "未通过"
    lines = ["# CT 重建评估", "", f"- 结果：**{status}**", f"- Solver：`{metrics['solver']}`", f"- Case：`{metrics['case_id']}`", "", "## 指标", ""]
    for name in ("relative_error", "rmse", "psnr", "ssim", "data_residual", "runtime_seconds"):
        if name in metrics:
            lines.append(f"- `{name}`：{metrics[name]:.8g}")
    failures = evaluation["checks"][0]["failures"]
    if failures:
        lines.extend(["", "## 未通过原因", ""] + [f"- {value}" for value in failures])
    (directory / "evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_checksums(directory)
    return evaluation


def _resolve_relative(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_suite(path: str | Path) -> tuple[dict[str, Any], Path]:
    payload, source = load_yaml(path)
    _check_keys(
        payload,
        {
            "schema_version", "name", "data_root", "output_root", "device",
            "overwrite", "continue_on_error", "protocol", "benchmark_protocol",
            "budget", "retry_budget", "groups",
        },
        "benchmark suite",
    )
    if payload.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ConfigError(f"suite schema_version must be {SCHEMA_VERSION}.")
    if not isinstance(payload.get("name"), str) or not payload["name"].strip():
        raise ConfigError("suite name must be a non-empty string.")
    if not isinstance(payload.get("output_root"), str) or not payload["output_root"].strip():
        raise ConfigError("suite output_root must be a non-empty string.")
    for optional_path in ("data_root", "protocol"):
        if optional_path in payload and not isinstance(payload[optional_path], str):
            raise ConfigError(f"suite {optional_path} must be a string path.")
    if "device" in payload and not isinstance(payload["device"], str):
        raise ConfigError("suite device must be a string.")
    for flag in ("overwrite", "continue_on_error"):
        if flag in payload and not isinstance(payload[flag], bool):
            raise ConfigError(f"suite {flag} must be a bool.")
    if "benchmark_protocol" in payload and payload["benchmark_protocol"] not in BENCHMARK_PROTOCOLS:
        raise ConfigError(
            f"suite benchmark_protocol must be one of {list(BENCHMARK_PROTOCOLS)}."
        )
    if "budget" in payload:
        try:
            suite_budget = BenchmarkBudget.from_mapping(payload["budget"])
            suite_budget.validate()
        except (TypeError, ValueError) as error:
            raise ConfigError(f"invalid suite benchmark budget: {error}") from error
    if "retry_budget" in payload and (
        isinstance(payload["retry_budget"], bool)
        or not isinstance(payload["retry_budget"], int)
        or payload["retry_budget"] < 0
    ):
        raise ConfigError("suite retry_budget must be a nonnegative integer")
    groups = payload.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ConfigError("suite groups must be a non-empty list.")
    for index, group_value in enumerate(groups):
        group = _require_mapping(group_value, f"suite group {index}")
        _check_keys(
            group,
            {
                "algorithms", "cases", "benchmark_protocol", "budget",
                "observation_domain", "problem_constraints", "max_iterations",
                "retry_budget", "heldout_split", "calibration_cases", "oracle_metric",
            },
            f"suite group {index}",
        )
        if "benchmark_protocol" in group and group["benchmark_protocol"] not in BENCHMARK_PROTOCOLS:
            raise ConfigError(
                f"suite group {index} benchmark_protocol must be one of {list(BENCHMARK_PROTOCOLS)}."
            )
        if "budget" in group:
            try:
                group_budget = BenchmarkBudget.from_mapping(group["budget"])
                group_budget.validate()
            except (TypeError, ValueError) as error:
                raise ConfigError(f"invalid suite group {index} budget: {error}") from error
        if "retry_budget" in group and (
            isinstance(group["retry_budget"], bool)
            or not isinstance(group["retry_budget"], int)
            or group["retry_budget"] < 0
        ):
            raise ConfigError(f"suite group {index} retry_budget must be a nonnegative integer")
        if "heldout_split" in group:
            split = _require_mapping(group["heldout_split"], f"suite group {index} heldout_split")
            _check_keys(split, {"folds", "protocol_version"}, f"suite group {index} heldout_split")
            folds = split.get("folds", 3)
            if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
                raise ConfigError(f"suite group {index} heldout_split folds must be an integer >= 2")
            if "protocol_version" in split and not isinstance(split["protocol_version"], str):
                raise ConfigError(f"suite group {index} heldout_split protocol_version must be a string")
        if "calibration_cases" in group:
            calibration_cases = group["calibration_cases"]
            if (
                not isinstance(calibration_cases, list)
                or not calibration_cases
                or not all(isinstance(value, str) and value for value in calibration_cases)
            ):
                raise ConfigError(
                    f"suite group {index} calibration_cases must be a non-empty string list"
                )
        if "oracle_metric" in group:
            if group["oracle_metric"] not in {
                "psnr", "ssim", "rmse", "relative_error", "data_residual"
            }:
                raise ConfigError(
                    f"suite group {index} oracle_metric must be one of "
                    "psnr, ssim, rmse, relative_error, data_residual"
                )
        if not isinstance(group.get("cases"), list) or not group["cases"] or not all(isinstance(v, str) for v in group["cases"]):
            raise ConfigError(f"suite group {index} cases must be a non-empty string list.")
        algorithms = group.get("algorithms")
        if not isinstance(algorithms, list) or not algorithms:
            raise ConfigError(f"suite group {index} algorithms must be a non-empty list.")
        for alg_index, algorithm_value in enumerate(algorithms):
            algorithm = _require_mapping(algorithm_value, f"suite group {index} algorithm {alg_index}")
            _check_keys(
                algorithm,
                {"name", "config", "parameter_grid"},
                f"suite group {index} algorithm {alg_index}",
            )
            if algorithm.get("name") not in SOLVER_SPECS or not isinstance(algorithm.get("config"), str):
                raise ConfigError(f"suite group {index} has invalid algorithm entry {algorithm!r}.")
            if "parameter_grid" in algorithm:
                grid = algorithm["parameter_grid"]
                if not isinstance(grid, list) or not grid or not all(isinstance(item, Mapping) for item in grid):
                    raise ConfigError(
                        f"suite group {index} algorithm {alg_index} parameter_grid "
                        "must be a non-empty list of mappings"
                    )
                allowed_parameters = set(SOLVER_SPECS[algorithm["name"]].parameter_names)
                for trial_index, trial in enumerate(grid):
                    unknown = sorted(set(trial) - allowed_parameters)
                    if unknown:
                        raise ConfigError(
                            f"suite group {index} algorithm {alg_index} parameter_grid "
                            f"trial {trial_index} has unsupported parameter(s): {', '.join(unknown)}"
                        )
                    try:
                        _validate_parameter_types(trial)
                    except ConfigError as error:
                        raise ConfigError(
                            f"suite group {index} algorithm {alg_index} parameter_grid "
                            f"trial {trial_index}: {error}"
                        ) from error
        effective_protocol = group.get("benchmark_protocol", payload.get("benchmark_protocol"))
        if effective_protocol is None and isinstance(group.get("budget"), Mapping):
            effective_protocol = group["budget"].get("protocol")
        if effective_protocol is None and isinstance(payload.get("budget"), Mapping):
            effective_protocol = payload["budget"].get("protocol")
        if effective_protocol == ORACLE_CALIBRATION:
            calibration_cases = group.get("calibration_cases")
            if not calibration_cases:
                raise ConfigError(
                    f"suite group {index} oracle_calibration requires explicit calibration_cases"
                )
            if set(calibration_cases).intersection(group["cases"]):
                raise ConfigError(
                    f"suite group {index} calibration_cases and cases must be disjoint"
                )
            if "oracle_metric" not in group:
                raise ConfigError(
                    f"suite group {index} oracle_calibration requires oracle_metric"
                )
            if "budget" in group:
                oracle_budget = BenchmarkBudget.from_mapping(group["budget"])
            elif isinstance(payload.get("budget"), Mapping):
                oracle_budget = BenchmarkBudget.from_mapping(payload["budget"])
            else:
                oracle_budget = BenchmarkBudget(protocol=ORACLE_CALIBRATION)
            if oracle_budget.tuning_trials <= 0:
                raise ConfigError(
                    f"suite group {index} oracle_calibration requires tuning_trials > 0"
                )
            if (
                oracle_budget.tuning_max_forward_calls is None
                and oracle_budget.tuning_max_adjoint_calls is None
                and oracle_budget.tuning_runtime_seconds is None
            ):
                raise ConfigError(
                    f"suite group {index} oracle_calibration requires a tuning "
                    "call or runtime budget"
                )
            grid_lengths = {
                len(algorithm.get("parameter_grid", []))
                for algorithm in group.get("algorithms", [])
            }
            if not grid_lengths or 0 in grid_lengths or len(grid_lengths) != 1:
                raise ConfigError(
                    f"suite group {index} oracle_calibration requires an equal non-empty "
                    "parameter_grid for every algorithm"
                )
            if next(iter(grid_lengths)) != oracle_budget.tuning_trials:
                raise ConfigError(
                    f"suite group {index} parameter_grid length must equal tuning_trials"
                )
    return payload, source


def _suite_budget(suite: Mapping[str, Any], group: Mapping[str, Any]) -> BenchmarkBudget:
    """Resolve and validate one group budget without silently changing it."""

    raw: dict[str, Any] = {}
    if isinstance(suite.get("budget"), Mapping):
        raw.update(suite["budget"])
    if isinstance(group.get("budget"), Mapping):
        raw.update(group["budget"])
    protocol = group.get("benchmark_protocol")
    if protocol is None and "protocol" not in raw:
        protocol = suite.get("benchmark_protocol")
    if protocol is not None:
        raw["protocol"] = protocol
    budget = BenchmarkBudget.from_mapping(raw)
    budget.validate()
    return budget


def _suite_record(
    result: Mapping[str, Any],
    *,
    solver: str,
    case_id: str,
    budget: BenchmarkBudget,
    protocol: str,
) -> dict[str, Any]:
    """Normalize a single run into the multi-axis benchmark schema."""

    metrics = dict(result.get("metrics", {}) or {})
    diagnostics = dict(result.get("diagnostics", {}) or {})
    convergence = dict(diagnostics.get("convergence", {}) or {})
    resources = dict(diagnostics.get("resources", {}) or {})
    status = str(result.get("status", metrics.get("status", "failed")))
    execution_status = str(
        diagnostics.get(
            "execution_status",
            result.get("execution_status", metrics.get("execution_status", status)),
        )
    )
    runtime = metrics.get("runtime_seconds", resources.get("runtime_seconds"))
    final_residual = convergence.get("final_residual", diagnostics.get("final_residual"))
    parameter_sources = diagnostics.get("parameter_sources", {})
    if not isinstance(parameter_sources, Mapping):
        parameter_sources = {}
    parameter_source = "config"
    if any(
        str(value) == "oracle_calibration_development_only"
        for value in parameter_sources.values()
    ):
        parameter_source = "oracle_calibration_development_only"
    record: dict[str, Any] = {
        **metrics,
        "algorithm": solver,
        "solver": solver,
        "case_id": case_id,
        "status": status,
        "execution_status": execution_status,
        "geometry": diagnostics.get("geometry_type"),
        "observation_domain": diagnostics.get("observation_domain"),
        "regularizer": diagnostics.get("regularizer"),
        "parameters": diagnostics.get("parameters", {}),
        "parameter_source": parameter_source,
        "parameter_sources": dict(parameter_sources),
        "parameter_overrides": diagnostics.get("parameter_overrides", {}),
        "tuning_protocol": protocol,
        "convergence_status": convergence.get("status"),
        "stopping_reason": convergence.get("stopping_reason"),
        "iterations": diagnostics.get("iterations_completed"),
        "runtime_seconds": runtime,
        "forward_calls": resources.get("forward_calls"),
        "adjoint_calls": resources.get("adjoint_calls"),
        "peak_memory_mb": resources.get("peak_memory_mb"),
        "objective": diagnostics.get("objective", convergence.get("final_objective")),
        "residual": final_residual if final_residual is not None else metrics.get("data_residual"),
        "budget": budget.to_dict(),
        "device": diagnostics.get("device"),
        "dtype": diagnostics.get("dtype"),
        "initialization": diagnostics.get("initialization"),
        "preprocessing": diagnostics.get("preprocessing"),
        "mask_id": diagnostics.get("mask_id"),
    }
    if status != "success" and result.get("message"):
        record["failure_reason"] = result["message"]
    return record


def _suite_group_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Keep fairness and quality/resource comparisons as separate axes."""

    by_case: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        case_key = str(record.get("case_id", "unknown"))
        # A held-out fold is a distinct comparison group: its fit/held-out
        # view partition must not be compared as if it were the full case or
        # another fold.
        if record.get("split_fold") is not None:
            case_key = f"{case_key}::fold_{int(record['split_fold'])}"
        by_case.setdefault(case_key, []).append(record)
    fairness: dict[str, Any] = {}
    fronts: dict[str, list[dict[str, Any]]] = {}
    for case_id, rows in by_case.items():
        fairness[case_id] = check_fairness(rows)
        fronts[case_id] = [row.to_dict() for row in pareto_front(rows)]
    return {"fairness": fairness, "pareto_front": fronts}


def _suite_split_cases(
    group: Mapping[str, Any],
    case_ids: Sequence[str],
    *,
    data_root: Path | None,
) -> dict[str, tuple[dict[str, Any], tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]]]:
    """Materialize one deterministic train/held-out split per case."""

    configuration = group.get("heldout_split")
    if configuration is None:
        return {}
    split_config = _require_mapping(configuration, "heldout_split")
    folds = int(split_config.get("folds", 3))
    protocol_version = str(split_config.get("protocol_version", "heldout_projection_cv/v1"))
    result: dict[str, tuple[dict[str, Any], tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]]] = {}
    for case_id in case_ids:
        loaded = load_ct_case(case_id, data_root=data_root, device="cpu")
        angles = loaded.geometry.get("angles_rad", int(loaded.measurement.shape[-2]))
        split = make_heldout_projection_split(
            case_id,
            angles,
            folds=folds,
            protocol_version=protocol_version,
        )
        pairs = tuple(
            (split.training_indices(index), split.validation_folds[index])
            for index in range(split.fold_count)
        )
        result[case_id] = (split.to_dict(), pairs)
    return result


_ORACLE_PARAMETER_SOURCE = "oracle_calibration_development_only"
_ORACLE_METRICS = {"psnr", "ssim", "rmse", "relative_error", "data_residual"}


def _oracle_metric_value(record: Mapping[str, Any], metric: str) -> float | None:
    value = record.get(metric)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _oracle_trial_cap(total: int | None, trial_count: int, case_count: int) -> int | None:
    """Allocate a declared tuning-call budget evenly across trial/case jobs."""

    if total is None:
        return None
    jobs = max(1, int(trial_count) * int(case_count))
    return max(0, int(total) // jobs)


def _oracle_failure_record(
    *,
    solver: str,
    case_id: str,
    budget: BenchmarkBudget,
    protocol: str,
    trial_index: int,
    parameters: Mapping[str, Any],
    reason: str,
) -> dict[str, Any]:
    return {
        "algorithm": solver,
        "solver": solver,
        "case_id": case_id,
        "status": "resource_exhausted",
        "execution_status": "resource_exhausted",
        "tuning_protocol": protocol,
        "parameter_source": _ORACLE_PARAMETER_SOURCE,
        "parameter_sources": {
            str(name): _ORACLE_PARAMETER_SOURCE for name in parameters
        },
        "parameters": dict(parameters),
        "parameter_overrides": dict(parameters),
        "oracle_phase": "calibration",
        "oracle_trial_index": int(trial_index),
        "budget": budget.to_dict(),
        "failure_reason": reason,
    }


def run_suite(path: str | Path) -> dict[str, Any]:
    suite, source = load_suite(path)
    base = source.parent
    output_root = _prepare_output(_resolve_relative(base, suite["output_root"]), bool(suite.get("overwrite", False)))
    data_root = _resolve_relative(base, suite["data_root"]) if suite.get("data_root") else None
    device = str(suite.get("device", "cpu"))
    records: list[dict[str, Any]] = []
    calibration_records: list[dict[str, Any]] = []
    oracle_selection: dict[str, dict[str, Any]] = {}
    for group_index, group in enumerate(suite["groups"]):
        budget = _suite_budget(suite, group)
        protocol_name = budget.protocol
        retry_budget = group.get("retry_budget", suite.get("retry_budget", 0))
        if retry_budget:
            raise ConfigError(
                "benchmark suite retry_budget must be zero; use independent "
                "parent-linked runs for debugging retries"
            )
        if budget.tuning_trials and protocol_name != ORACLE_CALIBRATION:
            raise ConfigError(
                "benchmark suite declares tuning_trials, but this runner only executes "
                "tuning trials under oracle_calibration."
            )
        calibration_cases = tuple(str(case_id) for case_id in group.get("calibration_cases", ()))
        oracle_metric = str(group.get("oracle_metric", "psnr"))
        if protocol_name == ORACLE_CALIBRATION:
            if not calibration_cases:
                raise ConfigError(
                    f"group {group_index} oracle_calibration requires calibration_cases"
                )
            if budget.tuning_trials <= 0:
                raise ConfigError(
                    f"group {group_index} oracle_calibration requires tuning_trials > 0"
                )
            if oracle_metric not in _ORACLE_METRICS:
                raise ConfigError(f"unsupported oracle metric {oracle_metric!r}")
            grids = [algorithm.get("parameter_grid", ()) for algorithm in group["algorithms"]]
            grid_lengths = {len(grid) for grid in grids}
            if not grid_lengths or 0 in grid_lengths or len(grid_lengths) != 1:
                raise ConfigError(
                    f"group {group_index} oracle_calibration requires an equal non-empty "
                    "parameter_grid for every algorithm"
                )
            trial_count = next(iter(grid_lengths))
            if trial_count != budget.tuning_trials:
                raise ConfigError(
                    f"group {group_index} parameter_grid length {trial_count} "
                    f"does not match tuning_trials {budget.tuning_trials}"
                )
        elif calibration_cases or any("parameter_grid" in algorithm for algorithm in group["algorithms"]):
            raise ConfigError(
                f"group {group_index} calibration_cases/parameter_grid require oracle_calibration"
            )
        split_cases = _suite_split_cases(group, group["cases"], data_root=data_root)
        for algorithm in group["algorithms"]:
            config_path = _resolve_relative(base, algorithm["config"])
            load_algorithm_config(config_path, algorithm["name"])
            selected_override: dict[str, Any] = {}
            selected_sources: dict[str, str] = {}
            selection_key = f"group_{group_index}/{algorithm['name']}"
            if protocol_name == ORACLE_CALIBRATION:
                grid = [dict(item) for item in algorithm["parameter_grid"]]
                trial_rows: dict[int, list[dict[str, Any]]] = {
                    index: [] for index in range(len(grid))
                }
                tuning_started = time.perf_counter()
                tuning_forward_calls = 0
                tuning_adjoint_calls = 0
                per_job_forward_cap = _oracle_trial_cap(
                    budget.tuning_max_forward_calls,
                    len(grid),
                    len(calibration_cases),
                )
                per_job_adjoint_cap = _oracle_trial_cap(
                    budget.tuning_max_adjoint_calls,
                    len(grid),
                    len(calibration_cases),
                )
                stop_calibration = False
                for trial_index, override in enumerate(grid):
                    for calibration_case_id in calibration_cases:
                        calibration_dir = (
                            output_root
                            / "_calibration"
                            / algorithm["name"]
                            / calibration_case_id.replace("/", "__")
                            / f"trial_{trial_index}"
                        )
                        if (
                            budget.tuning_runtime_seconds is not None
                            and time.perf_counter() - tuning_started
                            >= budget.tuning_runtime_seconds
                        ):
                            record = _oracle_failure_record(
                                solver=algorithm["name"],
                                case_id=calibration_case_id,
                                budget=budget,
                                protocol=protocol_name,
                                trial_index=trial_index,
                                parameters=override,
                                reason="tuning runtime budget exhausted",
                            )
                        else:
                            try:
                                result = run_case(
                                    algorithm["name"],
                                    calibration_case_id,
                                    config_path,
                                    calibration_dir,
                                    device=device,
                                    data_root=data_root,
                                    overwrite=False,
                                    observation_domain=group.get("observation_domain"),
                                    problem_constraints=group.get("problem_constraints"),
                                    max_iterations=group.get("max_iterations"),
                                    max_forward_calls=per_job_forward_cap,
                                    max_adjoint_calls=per_job_adjoint_cap,
                                    parameter_overrides=override,
                                    parameter_sources={
                                        str(name): _ORACLE_PARAMETER_SOURCE for name in override
                                    },
                                )
                                record = _suite_record(
                                    result,
                                    solver=algorithm["name"],
                                    case_id=calibration_case_id,
                                    budget=budget,
                                    protocol=protocol_name,
                                )
                            except Exception as error:
                                record = {
                                    "algorithm": algorithm["name"],
                                    "solver": algorithm["name"],
                                    "case_id": calibration_case_id,
                                    "status": "failed",
                                    "execution_status": "failed",
                                    "message": str(error),
                                    "tuning_protocol": protocol_name,
                                    "parameter_source": _ORACLE_PARAMETER_SOURCE,
                                    "parameter_sources": {
                                        str(name): _ORACLE_PARAMETER_SOURCE for name in override
                                    },
                                    "parameters": dict(override),
                                    "parameter_overrides": dict(override),
                                    "oracle_phase": "calibration",
                                    "oracle_trial_index": int(trial_index),
                                    "budget": budget.to_dict(),
                                }
                                if not suite.get("continue_on_error", True):
                                    stop_calibration = True
                        record.update({
                            "oracle_phase": "calibration",
                            "oracle_trial_index": int(trial_index),
                            "oracle_parameters": dict(override),
                            "tuning_budget": {
                                "trial_count": len(grid),
                                "max_forward_calls": budget.tuning_max_forward_calls,
                                "max_adjoint_calls": budget.tuning_max_adjoint_calls,
                                "max_runtime_seconds": budget.tuning_runtime_seconds,
                            },
                        })
                        tuning_forward_calls += int(record.get("forward_calls") or 0)
                        tuning_adjoint_calls += int(record.get("adjoint_calls") or 0)
                        if (
                            budget.tuning_runtime_seconds is not None
                            and time.perf_counter() - tuning_started
                            > budget.tuning_runtime_seconds
                        ):
                            record.update({
                                "status": "resource_exhausted",
                                "failure_reason": "tuning runtime budget exceeded",
                            })
                        if (
                            budget.tuning_max_forward_calls is not None
                            and tuning_forward_calls > budget.tuning_max_forward_calls
                        ):
                            record.update({
                                "status": "resource_exhausted",
                                "failure_reason": (
                                    "tuning forward-call budget exceeded: "
                                    f"{tuning_forward_calls} > {budget.tuning_max_forward_calls}"
                                ),
                            })
                        if (
                            budget.tuning_max_adjoint_calls is not None
                            and tuning_adjoint_calls > budget.tuning_max_adjoint_calls
                        ):
                            record.update({
                                "status": "resource_exhausted",
                                "failure_reason": (
                                    "tuning adjoint-call budget exceeded: "
                                    f"{tuning_adjoint_calls} > {budget.tuning_max_adjoint_calls}"
                                ),
                            })
                        calibration_records.append(record)
                        trial_rows[trial_index].append(record)
                        if stop_calibration:
                            break
                    if stop_calibration:
                        break
                valid_trials: list[dict[str, Any]] = []
                for trial_index, rows in trial_rows.items():
                    values = [
                        _oracle_metric_value(row, oracle_metric)
                        for row in rows
                        if row.get("status") == "success"
                    ]
                    if (
                        len(rows) == len(calibration_cases)
                        and len(values) == len(calibration_cases)
                    ):
                        valid_trials.append({
                            "trial_index": int(trial_index),
                            "score": float(sum(values) / len(values)),
                            "parameters": dict(grid[trial_index]),
                        })
                maximize = oracle_metric in {"psnr", "ssim"}
                if valid_trials:
                    selected = sorted(
                        valid_trials,
                        key=lambda item: (
                            -item["score"] if maximize else item["score"],
                            item["trial_index"],
                        ),
                    )[0]
                    selected_trial_index = int(selected["trial_index"])
                    selected_override = dict(grid[selected_trial_index])
                else:
                    # Keep a deterministic final attempt even when every
                    # calibration candidate failed; all such failures remain
                    # in benchmark.json.
                    selected_trial_index = 0
                    selected_override = dict(grid[0])
                selected_sources = {
                    str(name): _ORACLE_PARAMETER_SOURCE for name in selected_override
                }
                oracle_selection[selection_key] = {
                    "algorithm": algorithm["name"],
                    "calibration_cases": list(calibration_cases),
                    "metric": oracle_metric,
                    "trial_count": len(grid),
                    "valid_trials": valid_trials,
                    "selected_trial_index": selected_trial_index,
                    "selected_parameters": dict(selected_override),
                    "selected_parameter_sources": dict(selected_sources),
                    "tuning_forward_calls": tuning_forward_calls,
                    "tuning_adjoint_calls": tuning_adjoint_calls,
                    "tuning_runtime_seconds": time.perf_counter() - tuning_started,
                    "tuning_budget": {
                        "trial_count": budget.tuning_trials,
                        "max_forward_calls": budget.tuning_max_forward_calls,
                        "max_adjoint_calls": budget.tuning_max_adjoint_calls,
                        "max_runtime_seconds": budget.tuning_runtime_seconds,
                    },
                }
            for case_id in group["cases"]:
                split_payload = split_cases.get(case_id)
                runs = ((None, None, None),) if split_payload is None else tuple(
                    (fold, train, heldout)
                    for fold, (train, heldout) in enumerate(split_payload[1])
                )
                stop_after_failure = False
                for fold, fit_indices, heldout_indices in runs:
                    suffix = "" if fold is None else f"/fold_{fold}"
                    job_dir = output_root / algorithm["name"] / case_id.replace("/", "__") / suffix.strip("/")
                    try:
                        run_split_metadata = None
                        if split_payload is not None:
                            run_split_metadata = {
                                **split_payload[0],
                                "fold": int(fold),
                                "fit_view_indices": list(fit_indices),
                                "heldout_view_indices": list(heldout_indices),
                            }
                        result = run_case(
                            algorithm["name"],
                            case_id,
                            config_path,
                            job_dir,
                            device=device,
                            data_root=data_root,
                            overwrite=False,
                            observation_domain=group.get("observation_domain"),
                            problem_constraints=group.get("problem_constraints"),
                            max_iterations=group.get("max_iterations"),
                            max_forward_calls=budget.max_forward_calls,
                            max_adjoint_calls=budget.max_adjoint_calls,
                            parameter_overrides=selected_override or None,
                            parameter_sources=selected_sources or None,
                            fit_view_indices=fit_indices,
                            heldout_view_indices=heldout_indices,
                            split_metadata=run_split_metadata,
                            fixed_compute=protocol_name == EQUAL_OPERATOR_CALLS,
                        )
                        record = _suite_record(
                            result,
                            solver=algorithm["name"],
                            case_id=case_id,
                            budget=budget,
                            protocol=protocol_name,
                        )
                        if split_payload is not None:
                            record.update({
                                "split_protocol": split_payload[0]["protocol_version"],
                                "split_fold": int(fold),
                                "split_sha256": split_payload[0]["split_sha256"],
                            })
                        if protocol_name == ORACLE_CALIBRATION:
                            record.update({
                                "oracle_phase": "final",
                                "oracle_selection_key": selection_key,
                                "oracle_selected_trial_index": oracle_selection[selection_key]["selected_trial_index"],
                                "oracle_selected_parameters": dict(selected_override),
                                "oracle_metric": oracle_metric,
                                "tuning_budget": oracle_selection[selection_key]["tuning_budget"],
                            })
                        try:
                            enforce_budget(
                                forward_calls=int(record.get("forward_calls") or 0),
                                adjoint_calls=int(record.get("adjoint_calls") or 0),
                                budget=budget,
                                runtime_seconds=(
                                    float(record["runtime_seconds"])
                                    if record.get("runtime_seconds") is not None else None
                                ),
                            )
                        except RuntimeError as error:
                            record.update({"status": "resource_exhausted", "failure_reason": str(error)})
                    except Exception as error:
                        record = {
                            "algorithm": algorithm["name"],
                            "solver": algorithm["name"],
                            "case_id": case_id,
                            "status": "failed",
                            "message": str(error),
                            "tuning_protocol": protocol_name,
                            "budget": budget.to_dict(),
                        }
                        if not suite.get("continue_on_error", True):
                            stop_after_failure = True
                    records.append(record)
                    if stop_after_failure:
                        break
                if stop_after_failure:
                    break
    protocol = None
    evaluation = None
    if suite.get("protocol"):
        protocol, protocol_source = load_protocol(_resolve_relative(base, suite["protocol"]))
        evaluation = evaluate_metrics(records, protocol)
        evaluation["protocol"] = str(protocol_source)
    else:
        evaluation = {"schema_version": SCHEMA_VERSION, "passed": bool(records) and all(record.get("status") == "success" for record in records), "record_count": len(records), "checks": []}
    comparison = _suite_group_summary(records)
    _write_json(output_root / "metrics.json", {"schema_version": SCHEMA_VERSION, "records": records})
    columns = sorted({key for record in records for key in record})
    with (output_root / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    calibration_columns = sorted({key for record in calibration_records for key in record})
    if calibration_columns:
        with (output_root / "calibration_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=calibration_columns)
            writer.writeheader()
            writer.writerows(calibration_records)
    _write_json(output_root / "evaluation.json", evaluation)
    protocols_used = sorted({
        str(record.get("tuning_protocol"))
        for record in [*records, *calibration_records]
        if record.get("tuning_protocol")
    })
    benchmark_protocol = suite.get("benchmark_protocol")
    if benchmark_protocol is None:
        benchmark_protocol = protocols_used[0] if len(protocols_used) == 1 else "mixed"
    _write_json(output_root / "benchmark.json", {
        "schema_version": "inv_framework.ct_benchmark.v1",
        "protocol": benchmark_protocol,
        "protocols_used": protocols_used,
        "calibration_records": calibration_records,
        "oracle_selection": oracle_selection,
        "comparison": comparison,
        "records": records,
    })
    _write_json(output_root / "manifest.json", {"schema_version": SCHEMA_VERSION, "suite": suite, "suite_source": str(source), "device": device, "record_count": len(records), "calibration_record_count": len(calibration_records), "passed": evaluation["passed"], "benchmark": "benchmark.json", "repository_provenance": _git_provenance(Path(__file__).resolve().parents[1]), "environment": {"python": sys.version, "torch": torch.__version__, "platform": platform.platform(), "git_revision": _git_revision()}})
    report = ["# CT Benchmark 汇总", "", f"- Suite：`{suite['name']}`", f"- 结果：**{'通过' if evaluation['passed'] else '未通过'}**", f"- 记录数：{len(records)}", "", "| Solver | Case | Status | PSNR | SSIM |", "| --- | --- | --- | ---: | ---: |"]
    for record in records:
        report.append(f"| {record.get('solver', '')} | {record.get('case_id', '')} | {record.get('status', '')} | {record.get('psnr', '')} | {record.get('ssim', '')} |")
    (output_root / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    _write_checksums(output_root)
    return {
        "output_root": output_root,
        "records": records,
        "evaluation": evaluation,
        "comparison": comparison,
        "calibration_records": calibration_records,
        "oracle_selection": oracle_selection,
    }


def validate_cases(case_id: str | None = None, *, data_root: str | Path | None = None) -> list[dict[str, Any]]:
    records = list_ct_cases(data_root=data_root)
    if case_id is not None:
        records = [record for record in records if record["case_id"] == case_id]
        if not records:
            raise ConfigError(f"unknown CT case: {case_id!r}")
    results = []
    for record in records:
        loaded = load_ct_case(record["case_id"], data_root=data_root, verify_checksum=True)
        results.append({"case_id": loaded.case_id, "status": "valid", "truth_shape": list(loaded.truth.shape), "measurement_shape": list(loaded.measurement.shape)})
    return results

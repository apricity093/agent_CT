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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from inv_framework.benchmarks import list_ct_cases, load_ct_case
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
)
from inv_framework.utils.metrics import ssim


SCHEMA_VERSION = 1
RESULT_FILENAME = "reconstruction.pt"


@dataclass(frozen=True)
class SolverSpec:
    name: str
    display_name: str
    dimensions: tuple[int, ...]
    geometry_types: tuple[str, ...]
    parameter_names: tuple[str, ...]
    description: str
    backend: str | None = None


_COMMON_ITERATIVE = ("num_iterations", "min_value", "max_value")
SOLVER_SPECS: dict[str, SolverSpec] = {
    "fbp": SolverSpec("fbp", "FBP", (2,), ("parallel_2d",), ("scale",), "Filtered backprojection."),
    "sirt": SolverSpec("sirt", "SIRT", (2,), ("parallel_2d",), _COMMON_ITERATIVE, "Simultaneous iterative reconstruction."),
    "landweber": SolverSpec("landweber", "Landweber", (2,), ("parallel_2d",), _COMMON_ITERATIVE + ("step_size",), "Landweber iteration."),
    "cgls": SolverSpec("cgls", "CGLS", (2,), ("parallel_2d",), _COMMON_ITERATIVE + ("tol",), "Conjugate-gradient least squares."),
    "lsqr": SolverSpec("lsqr", "LSQR", (2,), ("parallel_2d",), _COMMON_ITERATIVE + ("damping", "atol", "btol"), "Golub-Kahan LSQR."),
    "sart": SolverSpec("sart", "SART", (2,), ("parallel_2d",), _COMMON_ITERATIVE + ("block_size", "order_strategy", "seed", "relaxation", "eps"), "Ordered row-action reconstruction."),
    "os_sart": SolverSpec("os_sart", "OS-SART", (2,), ("parallel_2d",), _COMMON_ITERATIVE + ("block_size", "subset_count", "order_strategy", "seed", "relaxation", "eps"), "Ordered-subsets SART."),
    "mlem": SolverSpec("mlem", "MLEM", (2,), ("parallel_2d",), _COMMON_ITERATIVE + ("initial_value", "eps"), "Maximum-likelihood expectation maximization."),
    "osem": SolverSpec("osem", "OSEM", (2,), ("parallel_2d",), _COMMON_ITERATIVE + ("block_size", "subset_count", "order_strategy", "seed", "initial_value", "eps"), "Ordered-subsets expectation maximization."),
    "tikhonov": SolverSpec("tikhonov", "Tikhonov", (2,), ("parallel_2d",), _COMMON_ITERATIVE + ("reg_strength", "tolerance"), "Quadratic Tikhonov reconstruction."),
    "tv_fista": SolverSpec("tv_fista", "TV-FISTA", (2,), ("parallel_2d",), _COMMON_ITERATIVE + ("reg_strength", "step_size", "tolerance", "power_iterations", "tv_mode", "tv_num_iterations", "tv_tolerance"), "TV-regularized FISTA."),
    "fdk": SolverSpec("fdk", "FDK", (3,), ("cone_3d",), ("filter_type", "short_scan", "voxel_supersampling"), "Cone-beam Feldkamp-Davis-Kress reconstruction.", "ASTRA CUDA"),
}


class ConfigError(ValueError):
    """Raised for invalid user-authored CLI configuration."""


class BackendUnavailable(RuntimeError):
    """Raised when an optional numerical backend cannot run a requested job."""


def solver_records() -> list[dict[str, Any]]:
    return [asdict(SOLVER_SPECS[name]) for name in SOLVER_SPECS]


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
    return name, parameters, source


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


def validate_solver_case(name: str, case: Any) -> SolverSpec:
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
    data_range = float(case.metadata.get("ground_truth", {}).get("data_range", 1.0))
    params.setdefault("min_value", 0.0) if name not in {"fbp", "fdk"} else None
    params.setdefault("max_value", data_range) if name not in {"fbp", "fdk"} else None
    num_views = int(case.measurement.shape[-2])
    if name in {"os_sart", "osem"} and "subset_count" in params:
        count = int(params.pop("subset_count"))
        if count <= 0:
            raise ConfigError("subset_count must be positive.")
        params.setdefault("block_size", max(num_views // count, 1))
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


def _metrics(bundle: Mapping[str, Any], runtime_seconds: float | None = None) -> dict[str, Any]:
    reconstruction = bundle["reconstruction"].float()
    truth = bundle["truth"].float()
    measurement = bundle["measurement"].float()
    predicted = bundle["predicted_measurement"].float()
    data_range = float(bundle["data_range"])
    difference = reconstruction - truth
    rmse = difference.square().mean().sqrt()
    relative_error = difference.norm() / truth.norm().clamp_min(1e-12)
    data_residual = (predicted - measurement).norm() / measurement.norm().clamp_min(1e-12)
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
    result = {
        "schema_version": SCHEMA_VERSION,
        "solver": str(bundle["solver"]),
        "case_id": str(bundle["case_id"]),
        "status": "success",
        "dimension": int(bundle["dimension"]),
        "relative_error": float(relative_error.item()),
        "rmse": float(rmse.item()),
        "psnr": float(psnr_value.item()),
        "ssim": float(ssim_value.item()),
        "ssim_definition": ssim_definition,
        "data_residual": float(data_residual.item()),
    }
    if runtime_seconds is not None:
        result["runtime_seconds"] = float(runtime_seconds)
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


def _plot_comparison(path: Path, truth: torch.Tensor, reconstruction: torch.Tensor, dimension: int) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    truth_image = truth[0, 0] if dimension == 2 else truth[0, truth.shape[1] // 2]
    reconstruction_image = reconstruction[0, 0] if dimension == 2 else reconstruction[0, reconstruction.shape[1] // 2]
    vmin = float(truth_image.min().item())
    vmax = float(truth_image.max().item())
    figure, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    for axis, image, title in zip(axes, (truth_image, reconstruction_image), ("Reference", "Reconstruction")):
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
) -> dict[str, Any]:
    destination = _prepare_output(output_dir, overwrite)
    try:
        name, parameters, config_source = load_algorithm_config(config_path, solver_name)
        case = load_ct_case(case_id, data_root=data_root, device=device)
        spec = validate_solver_case(name, case)
        operator = build_operator(case, device)
        solver = build_solver(name, parameters, case)
        started = time.perf_counter()
        reconstruction = solver.solve(case.measurement, operator)
        runtime_seconds = time.perf_counter() - started
        if tuple(reconstruction.shape) != tuple(case.truth.shape):
            raise RuntimeError(f"solver returned shape {tuple(reconstruction.shape)}, expected {tuple(case.truth.shape)}")
        if not torch.isfinite(reconstruction).all():
            raise RuntimeError("solver returned non-finite values")
        predicted = operator.forward(reconstruction).detach()
        dimension = int(case.metadata.get("dimension", 2))
        bundle = {
            "schema_version": SCHEMA_VERSION,
            "solver": name,
            "case_id": case.case_id,
            "dimension": dimension,
            "data_range": float(case.metadata.get("ground_truth", {}).get("data_range", 1.0)),
            "reconstruction": reconstruction.detach().cpu(),
            "truth": case.truth.detach().cpu(),
            "measurement": case.measurement.detach().cpu(),
            "predicted_measurement": predicted.detach().cpu(),
            "roi_mask": case.roi_mask.detach().cpu() if case.roi_mask is not None else None,
            "valid_measurement_mask": case.valid_measurement_mask.detach().cpu() if case.valid_measurement_mask is not None else None,
        }
        torch.save(bundle, destination / RESULT_FILENAME)
        metrics = _metrics(bundle, runtime_seconds)
        _write_json(destination / "metrics.json", metrics)
        _plot_comparison(destination / "comparison.png", case.truth, reconstruction, dimension)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "success",
            "solver": asdict(spec),
            "parameters": parameters,
            "case_id": case.case_id,
            "case_source": str(case.source_path) if case.source_path else None,
            "data_root": str(Path(data_root).resolve()) if data_root else None,
            "config_source": str(config_source),
            "device": str(device),
            "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "git_revision": _git_revision()},
            "artifacts": {RESULT_FILENAME: _sha256(destination / RESULT_FILENAME), "metrics.json": _sha256(destination / "metrics.json"), "comparison.png": _sha256(destination / "comparison.png")},
        }
        _write_json(destination / "manifest.json", manifest)
        _write_checksums(destination)
        return {"status": "success", "output_dir": destination, "metrics": metrics}
    except BackendUnavailable as error:
        return {"status": "unavailable", "output_dir": destination, **_failure(destination, "unavailable", error)}
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
    _check_keys(payload, {"schema_version", "name", "data_root", "output_root", "device", "overwrite", "continue_on_error", "protocol", "groups"}, "benchmark suite")
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
    groups = payload.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ConfigError("suite groups must be a non-empty list.")
    for index, group_value in enumerate(groups):
        group = _require_mapping(group_value, f"suite group {index}")
        _check_keys(group, {"algorithms", "cases"}, f"suite group {index}")
        if not isinstance(group.get("cases"), list) or not group["cases"] or not all(isinstance(v, str) for v in group["cases"]):
            raise ConfigError(f"suite group {index} cases must be a non-empty string list.")
        algorithms = group.get("algorithms")
        if not isinstance(algorithms, list) or not algorithms:
            raise ConfigError(f"suite group {index} algorithms must be a non-empty list.")
        for alg_index, algorithm_value in enumerate(algorithms):
            algorithm = _require_mapping(algorithm_value, f"suite group {index} algorithm {alg_index}")
            _check_keys(algorithm, {"name", "config"}, f"suite group {index} algorithm {alg_index}")
            if algorithm.get("name") not in SOLVER_SPECS or not isinstance(algorithm.get("config"), str):
                raise ConfigError(f"suite group {index} has invalid algorithm entry {algorithm!r}.")
    return payload, source


def run_suite(path: str | Path) -> dict[str, Any]:
    suite, source = load_suite(path)
    base = source.parent
    output_root = _prepare_output(_resolve_relative(base, suite["output_root"]), bool(suite.get("overwrite", False)))
    data_root = _resolve_relative(base, suite["data_root"]) if suite.get("data_root") else None
    device = str(suite.get("device", "cpu"))
    records: list[dict[str, Any]] = []
    for group in suite["groups"]:
        for algorithm in group["algorithms"]:
            config_path = _resolve_relative(base, algorithm["config"])
            load_algorithm_config(config_path, algorithm["name"])
            for case_id in group["cases"]:
                job_dir = output_root / algorithm["name"] / case_id.replace("/", "__")
                try:
                    result = run_case(algorithm["name"], case_id, config_path, job_dir, device=device, data_root=data_root, overwrite=False)
                    if result["status"] == "success":
                        record = dict(result["metrics"])
                    else:
                        record = {"solver": algorithm["name"], "case_id": case_id, "status": result["status"], "message": result.get("message")}
                except Exception as error:
                    record = {"solver": algorithm["name"], "case_id": case_id, "status": "failed", "message": str(error)}
                    if not suite.get("continue_on_error", True):
                        records.append(record)
                        break
                records.append(record)
    protocol = None
    evaluation = None
    if suite.get("protocol"):
        protocol, protocol_source = load_protocol(_resolve_relative(base, suite["protocol"]))
        evaluation = evaluate_metrics(records, protocol)
        evaluation["protocol"] = str(protocol_source)
    else:
        evaluation = {"schema_version": SCHEMA_VERSION, "passed": bool(records) and all(record.get("status") == "success" for record in records), "record_count": len(records), "checks": []}
    _write_json(output_root / "metrics.json", {"schema_version": SCHEMA_VERSION, "records": records})
    columns = sorted({key for record in records for key in record})
    with (output_root / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    _write_json(output_root / "evaluation.json", evaluation)
    _write_json(output_root / "manifest.json", {"schema_version": SCHEMA_VERSION, "suite": suite, "suite_source": str(source), "device": device, "record_count": len(records), "passed": evaluation["passed"], "environment": {"python": sys.version, "torch": torch.__version__, "platform": platform.platform(), "git_revision": _git_revision()}})
    report = ["# CT Benchmark 汇总", "", f"- Suite：`{suite['name']}`", f"- 结果：**{'通过' if evaluation['passed'] else '未通过'}**", f"- 记录数：{len(records)}", "", "| Solver | Case | Status | PSNR | SSIM |", "| --- | --- | --- | ---: | ---: |"]
    for record in records:
        report.append(f"| {record.get('solver', '')} | {record.get('case_id', '')} | {record.get('status', '')} | {record.get('psnr', '')} | {record.get('ssim', '')} |")
    (output_root / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    _write_checksums(output_root)
    return {"output_root": output_root, "records": records, "evaluation": evaluation}


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

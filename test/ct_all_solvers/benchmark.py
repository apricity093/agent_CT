"""Portable benchmark runner for every traditional CT solver in this project.

The public test-facing seam is :func:`run_benchmark`.  It keeps solver
construction, pilot calibration, metric collection, plotting, and artifact
serialization together so command-line and remote callers exercise the same
workflow.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from inv_framework.benchmarks import CTTestCase, load_ct_case
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
from inv_framework.utils.metrics import psnr, ssim


QUALITY_CASE_IDS = (
    "parallel_2d/tissue_breast_dense_clean_128",
    "parallel_2d/tissue_breast_sparse_poisson_128",
    "parallel_2d/tissue_breast_limited_angle_128",
)
FDK_CASE_ID = "cone_3d/spheres_astra_12"

TWO_D_ALGORITHM_IDS = (
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
)
ALL_ALGORITHM_IDS = TWO_D_ALGORITHM_IDS + ("fdk",)

DISPLAY_NAMES = {
    "fbp": "FBP",
    "sirt": "SIRT",
    "landweber": "Landweber",
    "cgls": "CGLS",
    "lsqr": "LSQR",
    "sart": "SART",
    "os_sart": "OS-SART",
    "mlem": "MLEM",
    "osem": "OSEM",
    "tikhonov": "Tikhonov",
    "tv_fista": "TV-FISTA",
    "fdk": "FDK",
}


@dataclass(frozen=True)
class BenchmarkConfig:
    """Inputs accepted by :func:`run_benchmark`.

    ``case_ids`` is configurable for small local tests, while the CLI defaults
    to the three required 128x128 quality cases.  ``require_fdk`` promotes an
    unavailable ASTRA CUDA backend from a recorded local status to an error.
    """

    output_dir: str | Path
    device: str = "cpu"
    profile_path: str | Path | None = None
    calibrate: bool = False
    require_fdk: bool = False
    include_fdk: bool = True
    seed: int = 20260804
    case_ids: tuple[str, ...] = QUALITY_CASE_IDS
    fdk_case_id: str = FDK_CASE_ID


@dataclass(frozen=True)
class PilotCase:
    case_id: str
    truth: torch.Tensor
    measurement: torch.Tensor
    angles: torch.Tensor
    data_range: float


@dataclass
class BenchmarkRun:
    output_dir: Path
    records: list[dict[str, Any]] = field(default_factory=list)
    profile: dict[str, Any] = field(default_factory=dict)


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
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _case_data_range(case: CTTestCase) -> float:
    return float(case.metadata.get("ground_truth", {}).get("data_range", 1.0))


def _parallel_operator(case: CTTestCase) -> ParallelBeamRadon2D:
    angles = torch.as_tensor(
        case.geometry["angles_rad"], dtype=torch.float32, device=case.truth.device
    )
    return ParallelBeamRadon2D(
        image_size=int(case.truth.shape[-1]),
        angles=angles,
        device=str(case.truth.device),
        in_channels=int(case.truth.shape[1]),
    )


def _psnr_2d(reconstruction: torch.Tensor, truth: torch.Tensor, data_range: float) -> float:
    return float(psnr(reconstruction, truth, data_range=data_range).mean().item())


def _ssim_2d(reconstruction: torch.Tensor, truth: torch.Tensor, data_range: float) -> float:
    return float(ssim(reconstruction, truth, data_range=data_range).mean().item())


def volume_psnr(reconstruction: torch.Tensor, truth: torch.Tensor, data_range: float) -> float:
    """PSNR over every voxel of a batched volume."""

    mse = (reconstruction - truth).square().mean()
    value = 20.0 * torch.log10(
        torch.as_tensor(data_range, dtype=mse.dtype, device=mse.device)
        / mse.sqrt().clamp_min(1e-12)
    )
    return float(value.item())


def axial_mean_ssim(
    reconstruction: torch.Tensor, truth: torch.Tensor, data_range: float
) -> float:
    """Mean two-dimensional SSIM over all axial slices of ``(B, Z, Y, X)``."""

    if reconstruction.ndim != 4 or truth.shape != reconstruction.shape:
        raise ValueError("axial_mean_ssim expects equal tensors shaped (B, Z, Y, X).")
    batch, depth, height, width = reconstruction.shape
    reconstruction_2d = reconstruction.reshape(batch * depth, 1, height, width)
    truth_2d = truth.reshape(batch * depth, 1, height, width)
    return float(ssim(reconstruction_2d, truth_2d, data_range=data_range).mean().item())


def _candidate_parameters() -> dict[str, list[dict[str, Any]]]:
    def iterations(values: Iterable[int]) -> list[dict[str, Any]]:
        return [{"num_iterations": int(value)} for value in values]

    return {
        "fbp": [{}],
        "sirt": iterations((25, 50, 100)),
        "landweber": [
            {"num_iterations": count, "step_size": step}
            for count in (50, 100, 200)
            for step in (1e-4, 5e-4, 1e-3, 5e-3)
        ],
        "cgls": iterations((25, 50, 100)),
        "lsqr": iterations((25, 50, 100)),
        "sart": [
            {"num_iterations": count, "relaxation": relaxation, "block_size": 1}
            for count in (5, 10, 20)
            for relaxation in (0.1, 0.25, 0.5)
        ],
        "os_sart": [
            {"num_iterations": count, "relaxation": relaxation, "subset_count": 10}
            for count in (5, 10, 20)
            for relaxation in (0.1, 0.25, 0.5)
        ],
        "mlem": iterations((25, 50, 100)),
        "osem": [
            {"num_iterations": count, "subset_count": 10} for count in (25, 50, 100)
        ],
        "tikhonov": [
            {"reg_strength": strength, "num_iterations": 100}
            for strength in (1e-7, 1e-6, 1e-5, 1e-4, 1e-3)
        ],
        "tv_fista": [
            {"reg_strength": strength, "num_iterations": 50}
            for strength in (1e-7, 1e-6, 1e-5, 1e-4)
        ],
        "fdk": [{}],
    }


def default_profile() -> dict[str, Any]:
    """Return a deterministic fallback profile without calibration."""

    candidates = _candidate_parameters()
    return {
        "schema_version": 1,
        "source": "fallback_first_candidate",
        "seed": 20260804,
        "solvers": {name: values[0] for name, values in candidates.items()},
    }


def _resolve_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    if "solvers" not in profile or not isinstance(profile["solvers"], Mapping):
        raise ValueError("benchmark profile must contain a solvers mapping.")
    missing = set(ALL_ALGORITHM_IDS) - set(profile["solvers"])
    if missing:
        raise ValueError(f"benchmark profile is missing algorithms: {sorted(missing)}")
    return {
        "schema_version": int(profile.get("schema_version", 1)),
        "source": str(profile.get("source", "external")),
        "seed": int(profile.get("seed", 20260804)),
        "solvers": {
            name: dict(profile["solvers"][name]) for name in ALL_ALGORITHM_IDS
        },
        **({"calibration": profile["calibration"]} if "calibration" in profile else {}),
    }


def _load_profile(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return _resolve_profile(payload)


def _pilot_truth(image_size: int, data_range: float, device: torch.device) -> torch.Tensor:
    coordinates = torch.linspace(-1.0, 1.0, image_size, device=device)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    image = torch.full((image_size, image_size), 0.18 * data_range, device=device)
    image[((xx / 0.78).square() + (yy / 0.92).square()) <= 1.0] = 0.50 * data_range
    image[((xx + 0.26).square() + (yy - 0.18).square()) <= 0.16**2] = 1.00 * data_range
    image[((xx - 0.30).square() + (yy + 0.24).square()) <= 0.20**2] = 0.34 * data_range
    image[((xx + 0.05) / 0.12).square() + ((yy + 0.34) / 0.08).square() <= 1.0] = (
        0.72 * data_range
    )
    return image.unsqueeze(0).unsqueeze(0)


def _pilot_cases(device: torch.device, seed: int) -> list[PilotCase]:
    """Create held-out, in-memory 64x64 pilot cases for parameter selection."""

    torch.manual_seed(int(seed))
    image_size = 64
    data_range = 0.004
    truth = _pilot_truth(image_size, data_range, device)
    specifications = (
        ("pilot_dense_clean", 90, math.pi, False),
        ("pilot_sparse_poisson", 24, math.pi, True),
        ("pilot_limited_angle", 60, math.radians(120.0), False),
    )
    pilot_cases = []
    for name, views, coverage, add_poisson_noise in specifications:
        angles = torch.arange(views, dtype=torch.float32, device=device) * coverage / views
        operator = ParallelBeamRadon2D(
            image_size=image_size, angles=angles, device=str(device)
        )
        measurement = operator.forward(truth).detach()
        if add_poisson_noise:
            incident_photons = 2_000_000.0
            counts = torch.poisson(torch.exp(-measurement) * incident_photons)
            measurement = -torch.log(counts.clamp_min(0.5) / incident_photons)
        pilot_cases.append(
            PilotCase(name, truth, measurement, angles, data_range)
        )
    return pilot_cases


def _subset_block_size(num_views: int, parameters: Mapping[str, Any]) -> int:
    if "block_size" in parameters:
        return int(parameters["block_size"])
    return max(int(num_views) // int(parameters.get("subset_count", 10)), 1)


def build_solver(
    algorithm: str,
    parameters: Mapping[str, Any],
    *,
    data_range: float,
    num_views: int,
):
    """Build one solver with fixed benchmark constraints and selected parameters."""

    params = dict(parameters)
    bounds = {"min_value": 0.0, "max_value": float(data_range)}
    if algorithm == "fbp":
        return FBPSolver()
    if algorithm == "sirt":
        return SIRTSolver(**params, **bounds)
    if algorithm == "landweber":
        return LandweberSolver(**params, **bounds)
    if algorithm == "cgls":
        return CGLSSolver(tol=0.0, **params, **bounds)
    if algorithm == "lsqr":
        return LSQRSolver(atol=0.0, btol=0.0, **params, **bounds)
    if algorithm == "sart":
        return SARTSolver(order_strategy="ordered", **params, **bounds)
    if algorithm == "os_sart":
        params["block_size"] = _subset_block_size(num_views, params)
        params.pop("subset_count", None)
        return OSSARTSolver(order_strategy="ordered", **params, **bounds)
    if algorithm == "mlem":
        return MLEMSolver(**params, **bounds)
    if algorithm == "osem":
        params["block_size"] = _subset_block_size(num_views, params)
        params.pop("subset_count", None)
        return OSEMSolver(order_strategy="ordered", **params, **bounds)
    if algorithm == "tikhonov":
        return TikhonovSolver(tolerance=1e-7, **params, **bounds)
    if algorithm == "tv_fista":
        regularizer = TVRegularizer(num_iterations=50, tolerance=1e-5)
        return TVFISTASolver(
            regularizer=regularizer,
            tolerance=0.0,
            power_iterations=8,
            **params,
            **bounds,
        )
    if algorithm == "fdk":
        return FDKSolver(**params)
    raise KeyError(f"unknown traditional CT algorithm {algorithm!r}")


def calibrate_profile(device: str | torch.device = "cpu", seed: int = 20260804) -> dict[str, Any]:
    """Select one fixed parameter set per 2D solver on held-out pilot cases."""

    resolved_device = torch.device(device)
    candidates = _candidate_parameters()
    pilot_cases = _pilot_cases(resolved_device, seed)
    selected: dict[str, dict[str, Any]] = {"fbp": {}, "fdk": {}}
    details: dict[str, list[dict[str, Any]]] = {}

    for algorithm in TWO_D_ALGORITHM_IDS:
        candidate_scores = []
        for parameters in candidates[algorithm]:
            scores = []
            elapsed_seconds = 0.0
            for pilot in pilot_cases:
                operator = ParallelBeamRadon2D(
                    image_size=pilot.truth.shape[-1],
                    angles=pilot.angles,
                    device=str(resolved_device),
                )
                solver = build_solver(
                    algorithm,
                    parameters,
                    data_range=pilot.data_range,
                    num_views=pilot.angles.numel(),
                )
                started = time.perf_counter()
                reconstruction = solver.solve(pilot.measurement, operator)
                elapsed_seconds += time.perf_counter() - started
                if not torch.isfinite(reconstruction).all():
                    scores = []
                    break
                scores.append(
                    (
                        _psnr_2d(reconstruction, pilot.truth, pilot.data_range),
                        _ssim_2d(reconstruction, pilot.truth, pilot.data_range),
                    )
                )
            if scores:
                average_psnr = sum(item[0] for item in scores) / len(scores)
                average_ssim = sum(item[1] for item in scores) / len(scores)
            else:
                average_psnr = float("-inf")
                average_ssim = float("-inf")
            candidate_scores.append(
                {
                    "parameters": dict(parameters),
                    "mean_psnr": average_psnr,
                    "mean_ssim": average_ssim,
                    "runtime_seconds": elapsed_seconds,
                }
            )

        best = max(
            candidate_scores,
            key=lambda item: (item["mean_psnr"], item["mean_ssim"], -item["runtime_seconds"]),
        )
        selected[algorithm] = dict(best["parameters"])
        details[algorithm] = candidate_scores

    return {
        "schema_version": 1,
        "source": "held_out_64x64_pilot",
        "seed": int(seed),
        "solvers": selected,
        "calibration": {
            "pilot_case_ids": [pilot.case_id for pilot in pilot_cases],
            "selection_order": ["mean_psnr", "mean_ssim", "lower_runtime_seconds"],
            "candidates": details,
        },
    }


def _measure_2d(
    algorithm: str,
    case: CTTestCase,
    profile_parameters: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    operator = _parallel_operator(case)
    data_range = _case_data_range(case)
    solver = build_solver(
        algorithm,
        profile_parameters,
        data_range=data_range,
        num_views=int(case.measurement.shape[-2]),
    )
    started = time.perf_counter()
    reconstruction = solver.solve(case.measurement, operator)
    runtime_seconds = time.perf_counter() - started
    if tuple(reconstruction.shape) != tuple(case.truth.shape):
        raise ValueError(
            f"{algorithm} returned {tuple(reconstruction.shape)}, expected {tuple(case.truth.shape)}"
        )
    if not torch.isfinite(reconstruction).all():
        raise ValueError(f"{algorithm} returned non-finite reconstruction values")
    record = {
        "algorithm": algorithm,
        "display_name": DISPLAY_NAMES[algorithm],
        "case_id": case.case_id,
        "dimension": 2,
        "status": "success",
        "psnr": _psnr_2d(reconstruction, case.truth, data_range),
        "ssim": _ssim_2d(reconstruction, case.truth, data_range),
        "ssim_definition": "2d_ssim",
        "runtime_seconds": runtime_seconds,
        "data_range": data_range,
        "measurement_min": float(case.measurement.min().item()),
        "parameters": dict(profile_parameters),
    }
    if algorithm in {"mlem", "osem"}:
        record["measurement_handling"] = "solver_internal_nonnegative_clamp"
    return reconstruction, record


def _astra_fdk_operator(case: CTTestCase):
    from inv_framework.operators.ct import ASTRAFDKOperator3D
    from inv_framework.operators.ct import astra_adapter as astra_backend

    if not astra_backend._HAS_ASTRA:
        raise RuntimeError("astra-toolbox is not installed")
    if not astra_backend.astra.use_cuda() or not torch.cuda.is_available():
        raise RuntimeError("ASTRA CUDA and PyTorch CUDA are required")
    astra = astra_backend.astra
    geometry = case.geometry
    size = int(geometry["domain_shape"][0])
    angles = np.asarray(geometry["angles_rad"], dtype=np.float32)
    volume_geometry = astra.create_vol_geom(size, size, size)
    projection_geometry = astra.create_proj_geom(
        "cone",
        float(geometry["detector_spacing_row"]),
        float(geometry["detector_spacing_col"]),
        int(geometry["detector_rows"]),
        int(geometry["detector_cols"]),
        angles,
        float(geometry["source_origin_distance"]),
        float(geometry["origin_detector_distance"]),
    )
    return ASTRAFDKOperator3D(volume_geometry, projection_geometry)


def _measure_fdk(
    case: CTTestCase, profile_parameters: Mapping[str, Any]
) -> tuple[torch.Tensor, dict[str, Any]]:
    operator = _astra_fdk_operator(case)
    solver = build_solver(
        "fdk",
        profile_parameters,
        data_range=_case_data_range(case),
        num_views=int(case.measurement.shape[-2]),
    )
    started = time.perf_counter()
    reconstruction = solver.solve(case.measurement, operator)
    runtime_seconds = time.perf_counter() - started
    if tuple(reconstruction.shape) != tuple(case.truth.shape):
        raise ValueError("FDK returned a reconstruction with the wrong volume shape")
    if not torch.isfinite(reconstruction).all():
        raise ValueError("FDK returned non-finite reconstruction values")
    data_range = _case_data_range(case)
    return reconstruction, {
        "algorithm": "fdk",
        "display_name": DISPLAY_NAMES["fdk"],
        "case_id": case.case_id,
        "dimension": 3,
        "status": "success",
        "psnr": volume_psnr(reconstruction, case.truth, data_range),
        "ssim": axial_mean_ssim(reconstruction, case.truth, data_range),
        "ssim_definition": "axial_mean_ssim",
        "runtime_seconds": runtime_seconds,
        "data_range": data_range,
        "measurement_min": float(case.measurement.min().item()),
        "parameters": dict(profile_parameters),
    }


def _plot_2d_case(
    output_path: Path,
    case: CTTestCase,
    reconstructions: Mapping[str, torch.Tensor],
    records: Mapping[str, Mapping[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    panels = [("Reference", case.truth)] + [
        (DISPLAY_NAMES[name], reconstructions[name]) for name in TWO_D_ALGORITHM_IDS
    ]
    columns = 4
    rows = math.ceil(len(panels) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 3.2 * rows))
    axes = np.asarray(axes).reshape(-1)
    data_range = _case_data_range(case)
    for axis, (name, image) in zip(axes, panels):
        axis.imshow(image[0, 0].detach().cpu(), cmap="gray", vmin=0.0, vmax=data_range)
        if name == "Reference":
            axis.set_title("Reference")
        else:
            record = records[name.lower().replace("-", "_")]
            axis.set_title(
                f"{name}\nPSNR {record['psnr']:.2f} dB | SSIM {record['ssim']:.3f}"
            )
        axis.axis("off")
    for axis in axes[len(panels) :]:
        axis.axis("off")
    figure.suptitle(case.case_id, fontsize=13)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure.savefig(output_path, dpi=160, facecolor="white")
    plt.close(figure)


def _plot_fdk_case(
    output_path: Path,
    case: CTTestCase,
    reconstruction: torch.Tensor,
    record: Mapping[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    data_range = _case_data_range(case)
    slice_index = case.truth.shape[1] // 2
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.6))
    axes[0].imshow(
        case.truth[0, slice_index].detach().cpu(), cmap="gray", vmin=0.0, vmax=data_range
    )
    axes[0].set_title("Reference axial slice")
    axes[1].imshow(
        reconstruction[0, slice_index].detach().cpu(),
        cmap="gray",
        vmin=0.0,
        vmax=data_range,
    )
    axes[1].set_title(
        f"FDK\nPSNR {record['psnr']:.2f} dB | SSIM {record['ssim']:.3f}"
    )
    for axis in axes:
        axis.axis("off")
    figure.suptitle(f"{case.case_id} | SSIM: axial mean", fontsize=12)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    figure.savefig(output_path, dpi=160, facecolor="white")
    plt.close(figure)


def _write_csv(path: Path, records: list[Mapping[str, Any]]) -> None:
    fieldnames = [
        "algorithm",
        "display_name",
        "case_id",
        "dimension",
        "status",
        "psnr",
        "ssim",
        "ssim_definition",
        "runtime_seconds",
        "data_range",
        "measurement_min",
        "measurement_handling",
        "parameters",
        "reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["parameters"] = json.dumps(_jsonable(row.get("parameters", {})), sort_keys=True)
            writer.writerow(row)


def _write_manifest(
    output_dir: Path,
    config: BenchmarkConfig,
    profile: Mapping[str, Any],
    records: list[Mapping[str, Any]],
) -> None:
    astra_version = None
    try:
        from inv_framework.operators.ct import astra_adapter as astra_backend

        if astra_backend._HAS_ASTRA:
            astra_version = getattr(astra_backend.astra, "__version__", "unknown")
    except ImportError:
        pass
    payload = {
        "schema_version": 1,
        "benchmark_kind": "all_traditional_ct_solvers",
        "algorithm_ids": list(ALL_ALGORITHM_IDS),
        "config": asdict(config),
        "profile": profile,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "astra": astra_version,
            "git_revision": _git_revision(),
        },
        "result_count": len(records),
        "artifact_sha256": {
            path.name: _sha256(path)
            for path in output_dir.iterdir()
            if path.is_file() and path.name != "manifest.json"
        },
    }
    _write_json(output_dir / "manifest.json", payload)


def run_benchmark(config: BenchmarkConfig) -> BenchmarkRun:
    """Run the configured 2D matrix and optional ASTRA FDK benchmark.

    Local machines without ASTRA CUDA still receive an explicit FDK
    ``unavailable`` record unless ``require_fdk`` is true.  All 2D solver
    failures are raised because they have no optional backend dependency.
    """

    destination = Path(config.output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(config.seed))
    if config.profile_path is not None:
        profile = _load_profile(config.profile_path)
    elif config.calibrate:
        profile = calibrate_profile(config.device, config.seed)
    else:
        profile = default_profile()
    profile = _resolve_profile(profile)
    _write_json(destination / "calibration_profile.json", profile)

    run = BenchmarkRun(output_dir=destination, profile=profile)
    tensor_bundle: dict[str, Any] = {"two_d": {}, "fdk": None}
    for case_id in config.case_ids:
        case = load_ct_case(case_id, device=config.device)
        if int(case.geometry.get("dimension", case.metadata.get("dimension", 2))) == 3:
            raise ValueError(f"2D benchmark case expected, got {case_id!r}")
        reconstructions: dict[str, torch.Tensor] = {}
        records_by_algorithm: dict[str, Mapping[str, Any]] = {}
        for algorithm in TWO_D_ALGORITHM_IDS:
            reconstruction, record = _measure_2d(
                algorithm, case, profile["solvers"][algorithm]
            )
            reconstructions[algorithm] = reconstruction
            records_by_algorithm[algorithm] = record
            run.records.append(record)
        plot_path = destination / f"{case_id.replace('/', '__')}_reconstructions.png"
        _plot_2d_case(plot_path, case, reconstructions, records_by_algorithm)
        tensor_bundle["two_d"][case_id] = {
            "truth": case.truth.detach().cpu(),
            "measurement": case.measurement.detach().cpu(),
            "reconstructions": {
                name: value.detach().cpu() for name, value in reconstructions.items()
            },
        }

    if config.include_fdk:
        fdk_case = load_ct_case(config.fdk_case_id, device=config.device)
        try:
            reconstruction, record = _measure_fdk(fdk_case, profile["solvers"]["fdk"])
        except RuntimeError as error:
            if config.require_fdk:
                raise RuntimeError(f"FDK is required but unavailable: {error}") from error
            record = {
                "algorithm": "fdk",
                "display_name": DISPLAY_NAMES["fdk"],
                "case_id": fdk_case.case_id,
                "dimension": 3,
                "status": "unavailable",
                "reason": str(error),
                "psnr": None,
                "ssim": None,
                "ssim_definition": "axial_mean_ssim",
                "runtime_seconds": None,
                "data_range": _case_data_range(fdk_case),
                "parameters": dict(profile["solvers"]["fdk"]),
            }
        else:
            run.records.append(record)
            _plot_fdk_case(destination / "cone_3d__spheres_astra_12_fdk.png", fdk_case, reconstruction, record)
            tensor_bundle["fdk"] = {
                "truth": fdk_case.truth.detach().cpu(),
                "measurement": fdk_case.measurement.detach().cpu(),
                "reconstruction": reconstruction.detach().cpu(),
            }
        if record not in run.records:
            run.records.append(record)

    _write_csv(destination / "metrics.csv", run.records)
    _write_json(
        destination / "metrics.json",
        {"schema_version": 1, "algorithm_ids": list(ALL_ALGORITHM_IDS), "records": run.records},
    )
    torch.save(tensor_bundle, destination / "reconstructions.pt")
    _write_manifest(destination, config, profile, run.records)
    return run

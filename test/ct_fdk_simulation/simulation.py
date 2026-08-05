"""Deterministic 3D modified Shepp-Logan data for ASTRA FDK smoke tests."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from inv_framework.operators.ct import ASTRAFDKOperator3D
from inv_framework.operators.ct import astra_adapter as astra_backend
from inv_framework.solvers import FDKSolver


@dataclass(frozen=True)
class FDKSimulationConfig:
    """Fixed geometry for the reproducible cone-beam FDK smoke case."""

    volume_size: int = 64
    num_angles: int = 180
    detector_rows: int = 128
    detector_cols: int = 128
    detector_spacing_row: float = 1.0
    detector_spacing_col: float = 1.0
    source_origin_distance: float = 320.0
    origin_detector_distance: float = 320.0
    attenuation_scale: float = 0.02


@dataclass
class FDKSimulationResult:
    """In-memory outputs from one ASTRA forward-projection and FDK run."""

    config: FDKSimulationConfig
    geometry: dict[str, Any]
    truth: torch.Tensor
    measurement: torch.Tensor
    reconstruction: torch.Tensor
    runtime_seconds: float
    operator: ASTRAFDKOperator3D


# (intensity, semi-axis-x, semi-axis-y, centre-x, centre-y, z-rotation-deg)
_SHEPP_LOGAN_2D = (
    (1.00, 0.6900, 0.9200, 0.0000, 0.0000, 0.0),
    (-0.80, 0.6624, 0.8740, 0.0000, -0.0184, 0.0),
    (-0.20, 0.1100, 0.3100, 0.2200, 0.0000, -18.0),
    (-0.20, 0.1600, 0.4100, -0.2200, 0.0000, 18.0),
    (0.10, 0.2100, 0.2500, 0.0000, 0.3500, 0.0),
    (0.10, 0.0460, 0.0460, 0.0000, 0.1000, 0.0),
    (0.10, 0.0460, 0.0460, 0.0000, -0.1000, 0.0),
    (0.10, 0.0460, 0.0230, -0.0800, -0.6050, 0.0),
    (0.10, 0.0230, 0.0230, 0.0000, -0.6060, 0.0),
    (0.10, 0.0230, 0.0460, 0.0600, -0.6050, 0.0),
)

# Keeps the phantom deliberately asymmetric in all three spatial axes.
_ORIENTATION_MARKER = (0.15, 0.0500, 0.0500, 0.0800, 0.1800, -0.1600, 0.2800)


def require_astra_cuda() -> Any:
    """Return ASTRA only when both ASTRA and PyTorch CUDA are usable."""

    if not astra_backend._HAS_ASTRA:
        raise RuntimeError("astra-toolbox is required for the simulated FDK smoke test.")
    astra = astra_backend.astra
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is required for the simulated FDK smoke test.")
    if not astra.use_cuda():
        raise RuntimeError("ASTRA CUDA is required for the simulated FDK smoke test.")
    return astra


def make_modified_shepp_logan_3d(
    size: int,
    *,
    device: str | torch.device,
    attenuation_scale: float = 0.02,
) -> torch.Tensor:
    """Return the project-defined modified Shepp-Logan 3D phantom.

    The first ten ellipsoids extend the repository's existing 2D modified
    Shepp-Logan coefficients.  The final marker makes incorrect Z or detector
    axis handling visible in the reconstruction preview.
    """

    if size < 2:
        raise ValueError("size must be at least 2.")
    if attenuation_scale <= 0.0:
        raise ValueError("attenuation_scale must be positive.")

    coordinates = -1.0 + (
        torch.arange(size, dtype=torch.float32, device=device) + 0.5
    ) * (2.0 / size)
    zz, yy, xx = torch.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    phantom = torch.zeros((size, size, size), dtype=torch.float32, device=device)

    for intensity, axis_x, axis_y, centre_x, centre_y, angle_degrees in _SHEPP_LOGAN_2D:
        angle = math.radians(angle_degrees)
        shifted_x = xx - centre_x
        shifted_y = yy - centre_y
        rotated_x = math.cos(angle) * shifted_x + math.sin(angle) * shifted_y
        rotated_y = -math.sin(angle) * shifted_x + math.cos(angle) * shifted_y
        axis_z = min(axis_x, axis_y)
        mask = (
            (rotated_x / axis_x).square()
            + (rotated_y / axis_y).square()
            + (zz / axis_z).square()
            <= 1.0
        )
        phantom = torch.where(mask, phantom + intensity, phantom)

    intensity, axis_x, axis_y, axis_z, centre_x, centre_y, centre_z = _ORIENTATION_MARKER
    marker_mask = (
        ((xx - centre_x) / axis_x).square()
        + ((yy - centre_y) / axis_y).square()
        + ((zz - centre_z) / axis_z).square()
        <= 1.0
    )
    phantom = torch.where(marker_mask, phantom + intensity, phantom)
    return phantom.clamp(0.0, 1.0).mul(attenuation_scale).unsqueeze(0)


def build_operator(
    config: FDKSimulationConfig,
) -> tuple[ASTRAFDKOperator3D, dict[str, Any]]:
    """Build the exact regular ASTRA cone geometry used for generation and FDK."""

    astra = require_astra_cuda()
    if min(config.volume_size, config.num_angles, config.detector_rows, config.detector_cols) < 1:
        raise ValueError("volume, angle, and detector dimensions must be positive.")
    if min(config.detector_spacing_row, config.detector_spacing_col) <= 0.0:
        raise ValueError("detector spacing must be positive.")
    if min(config.source_origin_distance, config.origin_detector_distance) <= 0.0:
        raise ValueError("source and detector distances must be positive.")

    angles = np.arange(config.num_angles, dtype=np.float32) * (
        2.0 * np.pi / config.num_angles
    )
    volume_geometry = astra.create_vol_geom(
        config.volume_size, config.volume_size, config.volume_size
    )
    projection_geometry = astra.create_proj_geom(
        "cone",
        config.detector_spacing_row,
        config.detector_spacing_col,
        config.detector_rows,
        config.detector_cols,
        angles,
        config.source_origin_distance,
        config.origin_detector_distance,
    )
    operator = ASTRAFDKOperator3D(volume_geometry, projection_geometry)
    geometry = {
        "type": "cone_3d",
        "domain_shape": list(operator.domain_shape),
        "range_shape": list(operator.range_shape),
        "image_layout": ["z", "y", "x"],
        "measurement_layout": ["detector_row", "angle", "detector_col"],
        "angles_rad": [float(value) for value in angles.tolist()],
        "detector_rows": config.detector_rows,
        "detector_cols": config.detector_cols,
        "detector_spacing_row": config.detector_spacing_row,
        "detector_spacing_col": config.detector_spacing_col,
        "source_origin_distance": config.source_origin_distance,
        "origin_detector_distance": config.origin_detector_distance,
        "length_unit": "astra_geometry_unit",
        "short_scan": False,
        "filter_type": "ram-lak",
    }
    return operator, geometry


def run_simulated_fdk(
    config: FDKSimulationConfig = FDKSimulationConfig(),
    *,
    device: str | torch.device = "cuda",
) -> FDKSimulationResult:
    """Generate clean cone-beam line integrals and reconstruct them with FDK."""

    target_device = torch.device(device)
    if target_device.type != "cuda":
        raise ValueError("The simulated FDK smoke test requires a CUDA device.")
    require_astra_cuda()
    operator, geometry = build_operator(config)
    truth = make_modified_shepp_logan_3d(
        config.volume_size,
        device=target_device,
        attenuation_scale=config.attenuation_scale,
    )
    measurement = operator.forward(truth).detach()
    started = time.perf_counter()
    reconstruction = FDKSolver(
        filter_type="ram-lak", short_scan=False, voxel_supersampling=1
    ).solve(measurement, operator)
    runtime_seconds = time.perf_counter() - started
    return FDKSimulationResult(
        config=config,
        geometry=geometry,
        truth=truth,
        measurement=measurement,
        reconstruction=reconstruction,
        runtime_seconds=runtime_seconds,
        operator=operator,
    )


def metrics_for(result: FDKSimulationResult) -> dict[str, Any]:
    """Return non-gating diagnostic metrics for an FDK smoke run."""

    prediction = result.operator.forward(result.reconstruction).detach()
    truth_norm = result.truth.norm().clamp_min(1e-12)
    measurement_norm = result.measurement.norm().clamp_min(1e-12)
    rmse = torch.sqrt((result.reconstruction - result.truth).square().mean())
    psnr = 20.0 * torch.log10(
        torch.as_tensor(
            result.config.attenuation_scale,
            dtype=rmse.dtype,
            device=rmse.device,
        )
        / rmse.clamp_min(1e-12)
    )
    return {
        "case_id": (
            f"cone_3d/modified_shepp_logan_{result.config.volume_size}_clean"
        ),
        "algorithm": "fdk",
        "status": "success",
        "truth_shape": list(result.truth.shape),
        "measurement_shape": list(result.measurement.shape),
        "reconstruction_shape": list(result.reconstruction.shape),
        "dtype": str(result.reconstruction.dtype).removeprefix("torch."),
        "device": str(result.reconstruction.device),
        "measurement_min": float(result.measurement.min().item()),
        "measurement_max": float(result.measurement.max().item()),
        "reconstruction_max_abs": float(result.reconstruction.abs().max().item()),
        "relative_error": float(((result.reconstruction - result.truth).norm() / truth_norm).item()),
        "data_residual": float(((prediction - result.measurement).norm() / measurement_norm).item()),
        "rmse": float(rmse.item()),
        "psnr": float(psnr.item()),
        "runtime_seconds": float(result.runtime_seconds),
    }


def validate_result(result: FDKSimulationResult) -> None:
    """Check only smoke-test correctness properties, not reconstruction quality."""

    expected_truth = (1, result.config.volume_size, result.config.volume_size, result.config.volume_size)
    expected_measurement = (
        1,
        result.config.detector_rows,
        result.config.num_angles,
        result.config.detector_cols,
    )
    if tuple(result.truth.shape) != expected_truth:
        raise ValueError(f"truth shape {tuple(result.truth.shape)} does not match {expected_truth}.")
    if tuple(result.measurement.shape) != expected_measurement:
        raise ValueError(
            f"measurement shape {tuple(result.measurement.shape)} does not match {expected_measurement}."
        )
    if tuple(result.reconstruction.shape) != expected_truth:
        raise ValueError("FDK reconstruction shape does not match the phantom shape.")
    tensors = (result.truth, result.measurement, result.reconstruction)
    if any(tensor.dtype != torch.float32 for tensor in tensors):
        raise TypeError("simulated FDK tensors must remain float32.")
    if any(tensor.device.type != "cuda" for tensor in tensors):
        raise RuntimeError("simulated FDK tensors must remain on CUDA.")
    if any(not torch.isfinite(tensor).all() for tensor in tensors):
        raise RuntimeError("simulated FDK tensors contain non-finite values.")
    if result.measurement.abs().max().item() == 0.0:
        raise RuntimeError("simulated projection is identically zero.")
    if result.reconstruction.abs().max().item() == 0.0:
        raise RuntimeError("FDK reconstruction is identically zero.")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_preview(path: Path, result: FDKSimulationResult, metrics: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    centre = result.truth.shape[1] // 2
    truth_slice = result.truth[0, centre].detach().cpu().numpy()
    reconstruction_slice = result.reconstruction[0, centre].detach().cpu().numpy()
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    for axis, image, title in (
        (axes[0], truth_slice, "Modified Shepp-Logan truth"),
        (axes[1], reconstruction_slice, f"FDK (PSNR {metrics['psnr']:.2f} dB)"),
    ):
        axis.imshow(image, cmap="gray", vmin=0.0, vmax=result.config.attenuation_scale)
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle("Simulated cone-beam FDK smoke case", fontsize=12)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    figure.savefig(path, dpi=160, facecolor="white")
    plt.close(figure)


def write_artifacts(result: FDKSimulationResult, output_dir: str | Path) -> dict[str, Any]:
    """Write small diagnostics without persisting the full simulated projection."""

    validate_result(result)
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")
    destination.mkdir(parents=True)

    metrics = metrics_for(result)
    geometry_payload = {
        "schema_version": 1,
        "case_id": metrics["case_id"],
        "generator": "modified_shepp_logan_3d_v1",
        "provenance": "model_matched_simulated_cone_beam",
        "measurement": {"kind": "line_integral", "noise_model": "none"},
        "config": asdict(result.config),
        "geometry": result.geometry,
    }
    _write_json(destination / "geometry.json", geometry_payload)
    _write_json(destination / "metrics.json", metrics)

    centre = result.truth.shape[1] // 2
    np.save(destination / "truth_axial.npy", result.truth[0, centre].detach().cpu().numpy())
    np.save(
        destination / "reconstruction_axial.npy",
        result.reconstruction[0, centre].detach().cpu().numpy(),
    )
    _write_preview(destination / "comparison.png", result, metrics)

    astra = astra_backend.astra
    files = {
        path.name: _sha256(path)
        for path in sorted(destination.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "schema_version": 1,
        "artifact_kind": "simulated_fdk_smoke",
        "case_id": metrics["case_id"],
        "config": asdict(result.config),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(result.reconstruction.device),
            "astra": getattr(astra, "__version__", "unknown"),
        },
        "files": files,
    }
    _write_json(destination / "manifest.json", manifest)
    return metrics

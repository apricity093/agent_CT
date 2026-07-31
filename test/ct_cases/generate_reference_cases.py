"""Author the small, redistributable CT cases committed under ``test/data``.

Run from the project root with the fno Python environment.  Every generated
array is deterministic; manifests record whether measurements are analytic,
model-matched, or produced by a named external backend.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inv_framework.benchmarks import CTTestCase, write_ct_case
from inv_framework.operators.ct import ASTRAFDKOperator3D, ParallelBeamRadon2D
from inv_framework.operators.ct import astra_adapter as astra_backend


DATA_ROOT = PROJECT_ROOT / "test" / "data"
CASE_ROOT = DATA_ROOT / "cases"
SEED = 20260731


def _pixel_centres(size: int) -> torch.Tensor:
    return -1.0 + (torch.arange(size, dtype=torch.float32) + 0.5) * (2.0 / size)


def _disk(size: int, radius: float = 0.65, value: float = 0.02) -> torch.Tensor:
    coordinates = _pixel_centres(size)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    image = torch.where(
        xx.square() + yy.square() <= radius**2,
        torch.full_like(xx, value),
        torch.zeros_like(xx),
    )
    return image[None, None]


def _shepp_logan(size: int, scale: float = 0.02) -> torch.Tensor:
    coordinates = _pixel_centres(size)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    image = torch.zeros((size, size), dtype=torch.float32)
    ellipses = [
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
    ]
    for intensity, a, b, cx, cy, angle_deg in ellipses:
        angle = math.radians(angle_deg)
        x = xx - cx
        y = yy - cy
        xr = math.cos(angle) * x + math.sin(angle) * y
        yr = -math.sin(angle) * x + math.cos(angle) * y
        mask = (xr / a).square() + (yr / b).square() <= 1.0
        image = torch.where(mask, image + intensity, image)
    return image.clamp(0.0, 1.0).mul(scale)[None, None]


def _parallel_geometry(size: int, angles: torch.Tensor) -> dict:
    return {
        "type": "parallel_2d",
        "domain_shape": [1, size, size],
        "range_shape": [1, int(angles.numel()), size],
        "image_layout": ["channel", "y", "x"],
        "measurement_layout": ["channel", "angle", "detector"],
        "angles_rad": [float(value) for value in angles.tolist()],
        "detector_count": size,
        "detector_spacing": 2.0 / size,
        "length_unit": "normalized_image_coordinate",
    }


def _metadata(
    *,
    case_id: str,
    noise_model: str,
    reference_kind: str,
    generator: str,
    tags: list[str],
    seed: int | None = None,
    parameters: dict | None = None,
) -> dict:
    return {
        "schema_version": "1.0",
        "case_id": case_id,
        "modality": "xray_ct",
        "dimension": 2,
        "ground_truth": {
            "quantity": "linear_attenuation",
            "unit": "normalized_inverse_length",
            "data_range": 0.02,
        },
        "measurement": {
            "kind": "line_integral",
            "noise_model": noise_model,
            "seed": seed,
            "parameters": parameters or {},
        },
        "provenance": {
            "reference_kind": reference_kind,
            "generator": generator,
            "license": "project-generated",
        },
        "capability_tags": tags,
    }


def _analytic_disk_case(size: int = 32, num_angles: int = 32) -> CTTestCase:
    case_id = "parallel_2d/disk_analytic_32"
    radius = 0.65
    value = 0.02
    angles = torch.arange(num_angles, dtype=torch.float32) * (math.pi / num_angles)
    truth = _disk(size, radius=radius, value=value)
    detector = _pixel_centres(size)
    chord_pixels = size * torch.sqrt((radius**2 - detector.square()).clamp_min(0.0))
    projection = (value * chord_pixels)[None, None, None, :].expand(
        1, 1, num_angles, size
    ).contiguous()
    roi = truth > 0.0
    return CTTestCase(
        case_id=case_id,
        truth=truth,
        measurement_clean=projection,
        measurement=projection.clone(),
        geometry=_parallel_geometry(size, angles),
        metadata=_metadata(
            case_id=case_id,
            noise_model="none",
            reference_kind="analytic_independent",
            generator="closed_form_parallel_beam_disk_chord_length",
            tags=["2d", "parallel", "analytic", "clean", "ground_truth"],
        ),
        roi_mask=roi,
    )


def _model_case(
    *,
    case_id: str,
    size: int,
    angles: torch.Tensor,
    noise_model: str,
    seed: int = SEED,
) -> CTTestCase:
    truth = _shepp_logan(size)
    operator = ParallelBeamRadon2D(image_size=size, angles=angles, device="cpu")
    clean = operator.forward(truth).detach()
    observed = clean.clone()
    parameters: dict[str, float] = {}
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if noise_model == "gaussian_relative":
        fraction = 0.01
        noise = torch.randn(clean.shape, generator=generator)
        observed = clean + fraction * clean.norm() / noise.norm().clamp_min(1e-12) * noise
        parameters = {"relative_l2_fraction": fraction}
    elif noise_model == "poisson_log":
        photon_count = 100000.0
        expected_counts = photon_count * torch.exp(-clean)
        counts = torch.poisson(expected_counts, generator=generator)
        observed = -torch.log(counts.clamp_min(1.0) / photon_count)
        parameters = {"incident_photon_count": photon_count}
    elif noise_model != "none":
        raise ValueError(f"unsupported authoring noise model: {noise_model}")
    tags = ["2d", "parallel", "ground_truth"]
    if int(angles.numel()) < 32:
        tags.append("sparse_view")
    if float(angles.max()) < math.pi * 0.9:
        tags.append("limited_angle")
    tags.append("clean" if noise_model == "none" else "noisy")
    return CTTestCase(
        case_id=case_id,
        truth=truth,
        measurement_clean=clean,
        measurement=observed,
        geometry=_parallel_geometry(size, angles),
        metadata=_metadata(
            case_id=case_id,
            noise_model=noise_model,
            reference_kind="model_matched",
            generator="inv_framework.operators.ct.ParallelBeamRadon2D",
            tags=tags,
            seed=seed if noise_model != "none" else None,
            parameters=parameters,
        ),
        roi_mask=truth > 0.0,
    )


def _cone_case() -> CTTestCase | None:
    if not astra_backend._HAS_ASTRA or not astra_backend.astra.use_cuda():
        return None
    if not torch.cuda.is_available():
        return None
    astra = astra_backend.astra
    size = 12
    num_angles = 36
    detector_rows = detector_cols = 24
    angles = np.arange(num_angles, dtype=np.float32) * (2.0 * np.pi / num_angles)
    volume_geometry = astra.create_vol_geom(size, size, size)
    projection_geometry = astra.create_proj_geom(
        "cone", 1.0, 1.0, detector_rows, detector_cols, angles, 60.0, 60.0
    )
    operator = ASTRAFDKOperator3D(volume_geometry, projection_geometry)
    coordinates = torch.linspace(-1.0, 1.0, size, device="cuda")
    zz, yy, xx = torch.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    truth = torch.zeros((1, size, size, size), dtype=torch.float32, device="cuda")
    truth[(xx.square() + yy.square() + zz.square())[None] <= 0.55**2] = 0.02
    clean = operator.forward(truth).detach()
    case_id = "cone_3d/spheres_astra_12"
    geometry = {
        "type": "cone_3d",
        "domain_shape": list(operator.domain_shape),
        "range_shape": list(operator.range_shape),
        "image_layout": ["z", "y", "x"],
        "measurement_layout": ["detector_row", "angle", "detector_col"],
        "angles_rad": [float(value) for value in angles.tolist()],
        "detector_rows": detector_rows,
        "detector_cols": detector_cols,
        "detector_spacing_row": 1.0,
        "detector_spacing_col": 1.0,
        "source_origin_distance": 60.0,
        "origin_detector_distance": 60.0,
        "length_unit": "astra_geometry_unit",
    }
    metadata = {
        "schema_version": "1.0",
        "case_id": case_id,
        "modality": "xray_ct",
        "dimension": 3,
        "ground_truth": {
            "quantity": "linear_attenuation",
            "unit": "normalized_inverse_length",
            "data_range": 0.02,
        },
        "measurement": {
            "kind": "line_integral",
            "noise_model": "none",
            "seed": None,
            "parameters": {},
        },
        "provenance": {
            "reference_kind": "backend_reference",
            "generator": f"ASTRA {getattr(astra, '__version__', 'unknown')} FDK geometry FP3D_CUDA",
            "license": "project-generated arrays; ASTRA GPL-3.0 backend",
        },
        "capability_tags": [
            "3d",
            "cone",
            "astra",
            "clean",
            "ground_truth",
            "fdk",
        ],
    }
    return CTTestCase(
        case_id=case_id,
        truth=truth.cpu(),
        measurement_clean=clean.cpu(),
        measurement=clean.cpu(),
        geometry=geometry,
        metadata=metadata,
        roi_mask=(truth > 0.0).cpu(),
    )


def main() -> None:
    size = 32
    full_angles = torch.arange(60, dtype=torch.float32) * (math.pi / 60)
    sparse_angles = torch.arange(16, dtype=torch.float32) * (math.pi / 16)
    limited_angles = torch.linspace(0.0, math.radians(120.0), 24)
    cases = [
        _analytic_disk_case(),
        _model_case(
            case_id="parallel_2d/shepp_logan_dense_clean_32",
            size=size,
            angles=full_angles,
            noise_model="none",
        ),
        _model_case(
            case_id="parallel_2d/shepp_logan_sparse_gaussian_32",
            size=size,
            angles=sparse_angles,
            noise_model="gaussian_relative",
        ),
        _model_case(
            case_id="parallel_2d/shepp_logan_sparse_poisson_32",
            size=size,
            angles=sparse_angles,
            noise_model="poisson_log",
        ),
        _model_case(
            case_id="parallel_2d/shepp_logan_limited_angle_32",
            size=size,
            angles=limited_angles,
            noise_model="none",
        ),
    ]
    cone_case = _cone_case()
    if cone_case is not None:
        cases.append(cone_case)

    records = []
    for case in cases:
        slug = case.case_id.replace("/", "__")
        record = write_ct_case(case, CASE_ROOT / slug, overwrite=True)
        record["path"] = f"cases/{slug}"
        records.append(record)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    catalog = {"schema_version": "1.0", "cases": sorted(records, key=lambda item: item["case_id"])}
    (DATA_ROOT / "catalog.json").write_text(
        json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(records)} cases to {DATA_ROOT}")


if __name__ == "__main__":
    main()

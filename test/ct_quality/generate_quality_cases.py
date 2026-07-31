"""Generate high-resolution synthetic CT cases with an independent ASTRA projector.

The committed arrays are deterministic.  ASTRA is used only while authoring
the reference sinograms; benchmark reconstruction uses invframework operators.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inv_framework.benchmarks import CTTestCase, write_ct_case


DATA_ROOT = PROJECT_ROOT / "test" / "data"
CASE_ROOT = DATA_ROOT / "quality_cases"
SEED = 20260731
IMAGE_SIZE = 128
PIXEL_SPACING_CM = 0.1
DISPLAY_SCALE_TO_CM_INVERSE = 1.0 / PIXEL_SPACING_CM


def _coordinates(size: int) -> tuple[torch.Tensor, torch.Tensor]:
    axis = -1.0 + (torch.arange(size, dtype=torch.float32) + 0.5) * (2.0 / size)
    return torch.meshgrid(axis, axis, indexing="ij")


def _ellipse(
    xx: torch.Tensor,
    yy: torch.Tensor,
    *,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    angle_deg: float = 0.0,
) -> torch.Tensor:
    angle = math.radians(angle_deg)
    x = xx - cx
    y = yy - cy
    xr = math.cos(angle) * x + math.sin(angle) * y
    yr = -math.sin(angle) * x + math.cos(angle) * y
    return (xr / rx).square() + (yr / ry).square() <= 1.0


def _smooth_noise(size: int, generator: torch.Generator) -> torch.Tensor:
    noise = torch.randn((1, 1, size, size), generator=generator)
    broad = F.avg_pool2d(noise, kernel_size=25, stride=1, padding=12)
    medium = F.avg_pool2d(noise, kernel_size=11, stride=1, padding=5)
    fine = F.avg_pool2d(noise, kernel_size=5, stride=1, padding=2)
    texture = 0.55 * broad / broad.std() + 0.30 * medium / medium.std()
    texture = texture + 0.15 * fine / fine.std()
    return texture[0, 0]


def tissue_breast_phantom(size: int = IMAGE_SIZE) -> tuple[torch.Tensor, torch.Tensor]:
    """Return voxel attenuation and an anatomical ROI for a synthetic slice."""

    yy, xx = _coordinates(size)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    outer = _ellipse(xx, yy, cx=-0.03, cy=0.02, rx=0.79, ry=0.86, angle_deg=-4.0)
    inner = _ellipse(xx, yy, cx=-0.03, cy=0.02, rx=0.755, ry=0.82, angle_deg=-4.0)
    central = _ellipse(xx, yy, cx=-0.02, cy=0.00, rx=0.61, ry=0.68, angle_deg=-4.0)

    # Values are attenuation per pixel. Divide by PIXEL_SPACING_CM for cm^-1.
    adipose_mu = 0.0190 * PIXEL_SPACING_CM
    glandular_delta = 0.0030 * PIXEL_SPACING_CM
    image = torch.zeros((size, size), dtype=torch.float32)
    texture = _smooth_noise(size, generator)
    adipose_texture = (0.00022 * PIXEL_SPACING_CM) * torch.tanh(texture / 2.5)
    image[inner] = adipose_mu + adipose_texture[inner]

    glandular_field = torch.sigmoid(2.4 * (texture - 0.12))
    glandular_field = glandular_field * central.float()
    image = image + glandular_delta * glandular_field

    # Elongated lobules and thin septa introduce directional tissue structure.
    lobules = [
        (-0.30, -0.21, 0.29, 0.075, -28.0),
        (0.22, -0.28, 0.32, 0.065, 31.0),
        (-0.18, 0.18, 0.34, 0.070, 17.0),
        (0.27, 0.22, 0.28, 0.060, -22.0),
        (0.00, 0.42, 0.24, 0.050, 4.0),
    ]
    for cx, cy, rx, ry, angle in lobules:
        mask = _ellipse(xx, yy, cx=cx, cy=cy, rx=rx, ry=ry, angle_deg=angle)
        image[mask & inner] += 0.00075 * PIXEL_SPACING_CM

    skin = outer & ~inner
    image[skin] = 0.0245 * PIXEL_SPACING_CM

    # A small deterministic calcification cluster supplies a high-contrast task.
    for cx, cy, radius in [
        (0.19, -0.04, 0.018),
        (0.23, -0.01, 0.012),
        (0.16, 0.01, 0.010),
        (0.21, 0.04, 0.009),
    ]:
        calc = (xx - cx).square() + (yy - cy).square() <= radius**2
        image[calc] = 0.0380 * PIXEL_SPACING_CM

    image = image.clamp_min(0.0)
    return image[None, None], outer[None, None]


def _astra_project(truth: torch.Tensor, angles: np.ndarray) -> tuple[torch.Tensor, str]:
    try:
        import astra
    except ImportError as error:
        raise RuntimeError(
            "Generating quality cases requires astra-toolbox; committed cases can "
            "still be loaded without ASTRA."
        ) from error

    size = int(truth.shape[-1])
    volume_geometry = astra.create_vol_geom(size, size)
    projection_geometry = astra.create_proj_geom(
        "parallel", 1.0, size, angles.astype(np.float32)
    )
    projector_id = astra.create_projector("line", projection_geometry, volume_geometry)
    sinogram_id = None
    try:
        sinogram_id, sinogram = astra.create_sino(
            truth[0, 0].detach().cpu().numpy(), projector_id
        )
        clean = torch.from_numpy(np.asarray(sinogram, dtype=np.float32))[None, None]
    finally:
        if sinogram_id is not None:
            astra.data2d.delete(sinogram_id)
        astra.projector.delete(projector_id)
    return clean, str(getattr(astra, "__version__", "unknown"))


def _geometry(size: int, angles: np.ndarray) -> dict:
    return {
        "type": "parallel_2d",
        "domain_shape": [1, size, size],
        "range_shape": [1, int(angles.size), size],
        "image_layout": ["channel", "y", "x"],
        "measurement_layout": ["channel", "angle", "detector"],
        "angles_rad": [float(value) for value in angles.tolist()],
        "detector_count": size,
        "detector_spacing_pixels": 1.0,
        "pixel_spacing_cm": PIXEL_SPACING_CM,
        "angular_coverage_deg": float(np.degrees(angles[-1] - angles[0])),
        "length_unit": "pixel",
    }


def _metadata(
    *,
    case_id: str,
    astra_version: str,
    noise_model: str,
    tags: list[str],
    parameters: dict,
) -> dict:
    return {
        "schema_version": "1.0",
        "case_id": case_id,
        "modality": "xray_ct",
        "dimension": 2,
        "ground_truth": {
            "quantity": "voxel_line_attenuation",
            "unit": "dimensionless_per_pixel",
            "data_range": 0.004,
            "pixel_spacing_cm": PIXEL_SPACING_CM,
            "display_scale_to_cm_inverse": DISPLAY_SCALE_TO_CM_INVERSE,
            "display_window_cm_inverse": [0.0175, 0.0255],
            "origin": "deterministic synthetic tissue phantom; not patient data",
        },
        "measurement": {
            "kind": "line_integral",
            "noise_model": noise_model,
            "seed": SEED if noise_model != "none" else None,
            "parameters": parameters,
        },
        "provenance": {
            "reference_kind": "backend_reference",
            "generator": f"ASTRA {astra_version} 2D line projector",
            "phantom_generator": "deterministic_multiscale_breast_tissue_v1",
            "reconstruction_operator_under_test": (
                "inv_framework.operators.ct.ParallelBeamRadon2D"
            ),
            "inverse_crime_control": (
                "reference measurements use ASTRA ray-driven line projection; "
                "reconstruction uses invframework grid-sample rotation projection"
            ),
            "license": "project-generated synthetic arrays; ASTRA GPL-3.0 backend",
        },
        "capability_tags": tags,
    }


def _case(
    *,
    case_id: str,
    angles: np.ndarray,
    noise_model: str,
) -> CTTestCase:
    truth, roi = tissue_breast_phantom()
    clean, astra_version = _astra_project(truth, angles)
    observed = clean.clone()
    parameters: dict[str, float] = {}
    if noise_model == "poisson_log":
        photon_count = 2000000.0
        generator = torch.Generator(device="cpu").manual_seed(SEED)
        expected = photon_count * torch.exp(-clean)
        counts = torch.poisson(expected, generator=generator)
        observed = -torch.log(counts.clamp_min(1.0) / photon_count)
        parameters = {"incident_photon_count": photon_count}
    elif noise_model != "none":
        raise ValueError(f"Unsupported noise model: {noise_model}")

    tags = [
        "2d",
        "parallel",
        "quality",
        "synthetic_tissue",
        "backend_reference",
        "ground_truth",
        "clean" if noise_model == "none" else "noisy",
    ]
    if angles.size < 100:
        tags.append("sparse_view")
    if float(angles[-1]) < math.radians(150.0):
        tags.append("limited_angle")
    return CTTestCase(
        case_id=case_id,
        truth=truth,
        measurement_clean=clean,
        measurement=observed,
        geometry=_geometry(IMAGE_SIZE, angles),
        metadata=_metadata(
            case_id=case_id,
            astra_version=astra_version,
            noise_model=noise_model,
            tags=tags,
            parameters=parameters,
        ),
        roi_mask=roi,
    )


def _angles(count: int, coverage_deg: float = 180.0) -> np.ndarray:
    return np.arange(count, dtype=np.float32) * math.radians(coverage_deg) / count


def main() -> None:
    cases = [
        _case(
            case_id="parallel_2d/tissue_breast_dense_clean_128",
            angles=_angles(180),
            noise_model="none",
        ),
        _case(
            case_id="parallel_2d/tissue_breast_sparse_poisson_128",
            angles=_angles(48),
            noise_model="poisson_log",
        ),
        _case(
            case_id="parallel_2d/tissue_breast_limited_angle_128",
            angles=_angles(120, coverage_deg=120.0),
            noise_model="none",
        ),
    ]

    catalog_path = DATA_ROOT / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    generated_ids = {case.case_id for case in cases}
    records = [record for record in catalog["cases"] if record["case_id"] not in generated_ids]
    for case in cases:
        slug = case.case_id.replace("/", "__")
        record = write_ct_case(case, CASE_ROOT / slug, overwrite=True)
        record["path"] = f"quality_cases/{slug}"
        records.append(record)
    catalog["cases"] = sorted(records, key=lambda item: item["case_id"])
    catalog_path.write_text(
        json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(cases)} quality cases to {CASE_ROOT}")


if __name__ == "__main__":
    main()

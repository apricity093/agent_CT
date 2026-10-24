"""Tissue-based cases for the bounded Agent showcase, separate from solvers."""

from dataclasses import replace
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .ct_cases import load_ct_case, write_ct_case
from ..operators.ct import ASTRAFDKOperator3D
from ..operators.ct.radon_torch import ParallelBeamRadon2D


SEED = 20261024
TISSUE_CASE = "parallel_2d/tissue_breast_sparse_poisson_128"


def gaussian_radius(sample_count: int, sigma: float) -> float:
    """99% Gaussian L2 noise bound; depends only on public acquisition settings."""
    t = math.log(100)
    return sigma * math.sqrt(sample_count + 2 * math.sqrt(sample_count * t) + 2 * t)


def tissue_volume(size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """A coherent 3D synthetic breast-like volume in attenuation-per-voxel units.

    This is a procedural phantom, not a patient image or a stack of 2D slices.
    Three texture scales, ellipsoidal lobules, skin and calcifications vary in Z.
    """
    axis = (torch.arange(size, dtype=torch.float32) + .5) * (2 / size) - 1
    z, y, x = torch.meshgrid(axis, axis, axis, indexing="ij")
    radius = ((x + .03) / .79)**2 + ((y - .02) / .86)**2 + (z / .81)**2
    outer = radius <= 1
    inner = radius <= .91
    noise = torch.randn((1, 1, size, size, size), generator=torch.Generator().manual_seed(SEED))
    texture = torch.zeros_like(noise)
    for kernel, weight in [(13, .55), (7, .30), (3, .15)]:
        field = F.avg_pool3d(noise, kernel, 1, kernel // 2)
        texture += weight * field / field.std()
    texture = texture[0, 0]
    image = (.0019 + .000022 * torch.tanh(texture / 2.5)) * inner
    central = (x / .61)**2 + (y / .68)**2 + (z / .65)**2 < 1
    image += .0003 * torch.sigmoid(2.4 * (texture - .12)) * central
    for cx, cy, cz, rx, ry, rz, angle in [
        (-.30, -.21, -.1, .29, .075, .33, -28),
        (.22, -.28, .12, .32, .065, .29, 31),
        (-.18, .18, .04, .34, .070, .38, 17),
        (.27, .22, -.17, .28, .060, .27, -22),
    ]:
        a = math.radians(angle)
        xr = math.cos(a) * (x - cx) + math.sin(a) * (y - cy)
        yr = -math.sin(a) * (x - cx) + math.cos(a) * (y - cy)
        lobule = (xr / rx)**2 + (yr / ry)**2 + ((z - cz) / rz)**2 <= 1
        image += .000075 * (lobule & inner)
    image[outer & ~inner] = .00245
    for cx, cy, cz in [(.19, -.04, .02), (.23, .01, -.02), (.16, .04, .05)]:
        calc = ((x - cx) / .025)**2 + ((y - cy) / .025)**2 + ((z - cz) / .04)**2 <= 1
        image[calc] = .0038
    return image[None], outer[None]


def prepare_cases(root: Path, ct_repo: Path) -> tuple[dict, float]:
    """Write one catalog with original 2D data and two explicitly new observations.

    Truth is used here only to generate observations. Public norm/radius settings
    use geometry and acquisition parameters, never reconstruction quality.
    Requires CUDA ASTRA for the 64-cubed cone case.
    """
    import astra
    template = load_ct_case(TISSUE_CASE, data_root=ct_repo / "test/data")
    op = ParallelBeamRadon2D(128, angles=torch.tensor(template.geometry["angles_rad"]))
    clean = op.forward(template.truth)
    bound = float(op.forward(torch.ones_like(template.truth)).abs().max()
                  * op.adjoint(torch.ones_like(clean)).abs().max())
    valid = torch.ones_like(clean, dtype=torch.bool)
    sigma = .00075
    radius = gaussian_radius(clean.numel(), sigma)
    gaussian = clean + sigma * torch.randn(clean.shape, generator=torch.Generator().manual_seed(SEED))
    common = {**template.metadata, "provenance": {
        **template.metadata["provenance"], "generator": "invframework parallel projector",
        "inverse_crime_control": "matched projector; flow demonstration, not independent-projector quality evidence",
    }}
    constrained = replace(template, case_id="parallel_2d/tissue_matched_gaussian_128",
        measurement=gaussian, measurement_clean=clean, valid_measurement_mask=valid,
        metadata={**common, "measurement": {"kind": "line_integral", "noise_model": "gaussian",
            "parameters": {"sigma": sigma}, "seed": SEED},
            "showcase": {"epsilon": radius, "epsilon_source": "99% Gaussian L2 upper bound from sigma=.00075 and active sample count; no truth fit"}})
    scale = 100000.
    expected = clean * scale
    emission = replace(template, case_id="parallel_2d/tissue_emission_counts_128",
        truth=template.truth * scale, measurement_clean=expected,
        measurement=torch.poisson(expected, generator=torch.Generator().manual_seed(SEED)),
        valid_measurement_mask=valid,
        metadata={**common, "ground_truth": {**template.metadata["ground_truth"],
            "quantity": "synthetic_activity", "data_range": .004 * scale,
            "unit": "arbitrary_activity_per_voxel",
            "display_note": "Activity is divided by the fixed generation scale for source-phantom-equivalent contrast; this is not a transmission attenuation measurement.",
            "display_scale_to_cm_inverse": 10 / scale},
            "measurement": {"kind": "counts", "observation_domain": "nonnegative_counts",
                "observation_model": "poisson_emission", "noise_model": "poisson",
                "parameters": {}, "seed": SEED}})
    size = 64
    angles = torch.arange(180) * (2 * math.pi / 180)
    vg = astra.create_vol_geom(size, size, size)
    pg = astra.create_proj_geom("cone", 1., 1., 128, 128, angles.numpy(), 320., 320.)
    cone = ASTRAFDKOperator3D(vg, pg)
    volume, roi = tissue_volume(size)
    measurement = cone.forward(volume.cuda()).detach().cpu()
    geometry = {"type": "cone_3d", "domain_shape": list(cone.domain_shape),
        "range_shape": list(cone.range_shape), "angles_rad": angles.tolist(),
        "image_layout": ["z", "y", "x"], "measurement_layout": ["detector_row", "angle", "detector_col"],
        "detector_rows": 128, "detector_cols": 128, "detector_spacing_row": 1.,
        "detector_spacing_col": 1., "source_origin_distance": 320., "origin_detector_distance": 320.,
        "length_unit": "astra_geometry_unit", "filter_type": "ram-lak", "short_scan": False}
    fdk = replace(template, case_id="cone_3d/tissue_volume_clean_64", truth=volume,
        measurement_clean=measurement, measurement=measurement, roi_mask=roi,
        valid_measurement_mask=torch.ones_like(measurement, dtype=torch.bool), geometry=geometry,
        metadata={**template.metadata, "dimension": 3, "capability_tags": ["3d", "cone", "synthetic_tissue", "clean"],
            "measurement": {"kind": "line_integral", "noise_model": "none", "parameters": {}},
            "provenance": {"generator": f"ASTRA {astra.__version__} FP3D_CUDA",
                "phantom_generator": "deterministic_multiscale_breast_volume_v1", "seed": SEED,
                "inverse_crime_control": "matched ASTRA projector; synthetic flow demonstration"}})
    records = []
    settings = {}
    for name, case in [("transmission", template), ("constrained_tv", constrained), ("emission", emission), ("cone_fdk", fdk)]:
        record = write_ct_case(case, root / name)
        record["path"] = name
        records.append(record)
        settings[name] = {"case_id": case.case_id, "path": name}
    import json
    (root / "catalog.json").write_text(json.dumps({"schema_version": "1.0", "cases": records}, indent=2) + "\n", encoding="utf-8")
    return settings, bound

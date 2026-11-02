from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from inv_framework.benchmarks import load_ct_case
from inv_framework.ct.backends import ASTRAFDKOperator3D, ParallelBeamRadon2D
from inv_framework.ct.backends import astra_adapter as astra_backend


DATA_ROOT = Path(__file__).resolve().parents[2] / "tests" / "data"


def _parallel_operator(case):
    angles = torch.tensor(case.geometry["angles_rad"], dtype=case.truth.dtype)
    return ParallelBeamRadon2D(
        image_size=case.truth.shape[-1], angles=angles, device=case.truth.device
    )


def test_model_matched_reference_is_exactly_reproducible():
    case = load_ct_case(
        "parallel_2d__ellipses_dense_clean", data_root=DATA_ROOT
    )
    predicted = _parallel_operator(case).forward(case.truth)
    assert torch.allclose(predicted, case.measurement_clean, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    "case_id",
    [
        "parallel_2d__ellipses_sparse_gaussian",
        "parallel_2d__ellipses_sparse_poisson",
    ],
)
def test_noisy_cases_preserve_clean_measurement(case_id):
    case = load_ct_case(case_id, data_root=DATA_ROOT)
    assert not torch.equal(case.measurement, case.measurement_clean)
    assert torch.isfinite(case.measurement).all()
    assert _parallel_operator(case).forward(case.truth).shape == case.measurement.shape


def _astra_operator_from_case(case):
    astra = astra_backend.astra
    size = int(case.geometry["domain_shape"][0])
    angles = np.asarray(case.geometry["angles_rad"], dtype=np.float32)
    volume_geometry = astra.create_vol_geom(size, size, size)
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


def test_astra_reference_reproduces_cross_version_projection_when_available():
    if not astra_backend._HAS_ASTRA:
        pytest.skip("astra-toolbox is not installed")
    if not astra_backend.astra.use_cuda() or not torch.cuda.is_available():
        pytest.skip("ASTRA CUDA and PyTorch CUDA are required")
    case = load_ct_case(
        "cone_3d/spheres_astra_12", data_root=DATA_ROOT, device="cuda"
    )
    predicted = _astra_operator_from_case(case).forward(case.truth).detach()
    relative_error = (
        predicted - case.measurement_clean
    ).norm() / case.measurement_clean.norm().clamp_min(1e-12)
    assert math.isfinite(float(relative_error.item()))
    assert relative_error < 2e-3

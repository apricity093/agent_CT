from __future__ import annotations

from pathlib import Path

import pytest
import torch

from inv_framework.benchmarks import CTTestCase, evaluate_ct_case, load_ct_case
from inv_framework.operators.ct import ParallelBeamRadon2D
from inv_framework.solvers import CGLSSolver


DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


def _operator(case):
    angles = torch.tensor(case.geometry["angles_rad"], dtype=case.truth.dtype)
    return ParallelBeamRadon2D(
        image_size=case.truth.shape[-1], angles=angles, device=case.truth.device
    )


def test_evaluate_case_accepts_injected_solver_and_operator():
    case = load_ct_case(
        "parallel_2d/shepp_logan_dense_clean_32", data_root=DATA_ROOT
    )
    result = evaluate_ct_case(
        CGLSSolver(num_iterations=24, tol=0.0),
        _operator(case),
        case,
    )
    assert result.reconstruction.shape == case.truth.shape
    assert torch.isfinite(result.reconstruction).all()
    assert result.metrics["relative_error"] < 0.26
    assert result.metrics["data_residual"] < 0.015
    assert result.metrics["psnr"] > 24.0


def test_evaluate_case_rejects_geometry_mismatch():
    case = load_ct_case("parallel_2d/disk_analytic_32", data_root=DATA_ROOT)
    wrong = ParallelBeamRadon2D(image_size=16, num_angles=8, device="cpu")
    with pytest.raises(ValueError, match="domain_shape"):
        evaluate_ct_case(CGLSSolver(num_iterations=1), wrong, case)


def test_case_batch_and_mask_validation_is_part_of_interface():
    with pytest.raises(ValueError, match="batch sizes"):
        CTTestCase(
            case_id="invalid/batch",
            truth=torch.zeros(2, 1, 4, 4),
            measurement_clean=torch.zeros(1, 1, 2, 4),
            measurement=torch.zeros(1, 1, 2, 4),
            geometry={"domain_shape": [1, 4, 4], "range_shape": [1, 2, 4]},
            metadata={},
        )

from __future__ import annotations

import math

import pytest
import torch

from inv_framework.regularizers import TVRegularizer

from .baselines import THRESHOLDS_2D
from .helpers import run_solver_2d


def _assert_thresholds(metrics, thresholds):
    assert metrics["relative_error"] <= thresholds["max_relative_error"]
    assert metrics["data_residual"] <= thresholds["max_data_residual"]
    assert metrics["psnr"] >= thresholds["min_psnr"]
    assert metrics["ssim"] >= thresholds["min_ssim"]


@pytest.mark.parametrize(
    "algorithm",
    ["cgls", "lsqr", "sart", "os_sart", "tikhonov", "tv_fista"],
)
def test_parallel_beam_reconstruction_workflow(
    algorithm,
    ct_case_2d,
    ct_solvers_2d,
    record_property,
):
    solver = ct_solvers_2d[algorithm]
    reconstruction, metrics = run_solver_2d(algorithm, solver, ct_case_2d)

    assert reconstruction.shape == ct_case_2d["truth"].shape
    assert reconstruction.dtype == ct_case_2d["measurement"].dtype
    assert reconstruction.device == ct_case_2d["measurement"].device
    assert torch.isfinite(reconstruction).all()
    assert all(math.isfinite(value) for key, value in metrics.items() if key != "algorithm")
    _assert_thresholds(metrics, THRESHOLDS_2D["clean"][algorithm])

    initial_data_term = 0.5 * ct_case_2d["measurement"].square().sum()
    residual = ct_case_2d["operator"].forward(reconstruction) - ct_case_2d["measurement"]
    if algorithm == "tikhonov":
        final_objective = 0.5 * residual.square().sum() + 0.5 * solver.reg_strength * reconstruction.square().sum()
        assert final_objective < initial_data_term
    elif algorithm == "tv_fista":
        regularizer = solver.regularizer
        assert isinstance(regularizer, TVRegularizer)
        final_objective = 0.5 * residual.square().sum() + solver.reg_strength * regularizer.value(reconstruction).sum()
        assert final_objective < initial_data_term

    for key, value in metrics.items():
        record_property(key, value)
    print(f"{algorithm}: {metrics}")


@pytest.mark.slow
@pytest.mark.parametrize(
    "algorithm",
    ["cgls", "lsqr", "sart", "os_sart", "tikhonov", "tv_fista"],
)
def test_sparse_noisy_parallel_beam_workflow(
    algorithm,
    ct_case_2d_noisy_sparse,
    ct_solvers_2d,
):
    reconstruction, metrics = run_solver_2d(
        algorithm,
        ct_solvers_2d[algorithm],
        ct_case_2d_noisy_sparse,
    )

    assert reconstruction.shape == ct_case_2d_noisy_sparse["truth"].shape
    assert torch.isfinite(reconstruction).all()
    _assert_thresholds(metrics, THRESHOLDS_2D["sparse_noisy"][algorithm])
    print(f"{algorithm}_sparse_noisy: {metrics}")

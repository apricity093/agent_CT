from __future__ import annotations

import json

import torch

from .benchmark import (
    ALL_ALGORITHM_IDS,
    TWO_D_ALGORITHM_IDS,
    BenchmarkConfig,
    axial_mean_ssim,
    default_profile,
    run_benchmark,
    volume_psnr,
)


def _fast_profile():
    profile = default_profile()
    profile["source"] = "test_fast_profile"
    profile["solvers"].update(
        {
            "sirt": {"num_iterations": 2},
            "landweber": {"num_iterations": 2, "step_size": 1e-3},
            "cgls": {"num_iterations": 2},
            "lsqr": {"num_iterations": 2},
            "sart": {"num_iterations": 1, "relaxation": 0.1, "block_size": 4},
            "os_sart": {"num_iterations": 1, "relaxation": 0.1, "subset_count": 4},
            "mlem": {"num_iterations": 2},
            "osem": {"num_iterations": 2, "subset_count": 4},
            "tikhonov": {"reg_strength": 1e-5, "num_iterations": 2},
            "tv_fista": {"reg_strength": 1e-6, "num_iterations": 2},
        }
    )
    return profile


def test_algorithm_registry_is_exactly_the_requested_traditional_set():
    assert set(ALL_ALGORITHM_IDS) == {
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
        "fdk",
    }
    assert len(TWO_D_ALGORITHM_IDS) == 11


def test_run_benchmark_writes_matrix_metrics_and_reference_reconstruction_figure(tmp_path):
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(_fast_profile()), encoding="utf-8")
    output_dir = tmp_path / "artifacts"
    run = run_benchmark(
        BenchmarkConfig(
            output_dir=output_dir,
            profile_path=profile_path,
            include_fdk=False,
            case_ids=("parallel_2d/shepp_logan_dense_clean_32",),
        )
    )

    assert len(run.records) == len(TWO_D_ALGORITHM_IDS)
    assert {record["algorithm"] for record in run.records} == set(TWO_D_ALGORITHM_IDS)
    assert all(record["status"] == "success" for record in run.records)
    assert all(torch.isfinite(torch.tensor(record["psnr"])) for record in run.records)
    assert all(torch.isfinite(torch.tensor(record["ssim"])) for record in run.records)
    assert (output_dir / "metrics.csv").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "calibration_profile.json").exists()
    assert (output_dir / "reconstructions.pt").exists()
    assert (output_dir / "manifest.json").exists()
    assert (
        output_dir / "parallel_2d__shepp_logan_dense_clean_32_reconstructions.png"
    ).exists()

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["algorithm_ids"] == list(ALL_ALGORITHM_IDS)
    assert len(metrics["records"]) == len(TWO_D_ALGORITHM_IDS)


def test_volume_metrics_use_full_volume_psnr_and_axial_slice_ssim():
    truth = torch.zeros(1, 3, 8, 8)
    truth[:, :, 2:6, 2:6] = 1.0
    reconstruction = truth.clone()
    assert volume_psnr(reconstruction, truth, data_range=1.0) > 100.0
    assert axial_mean_ssim(reconstruction, truth, data_range=1.0) > 0.99

    altered = reconstruction.clone()
    altered[:, 1] = 0.0
    assert volume_psnr(altered, truth, data_range=1.0) < 20.0
    assert axial_mean_ssim(altered, truth, data_range=1.0) < 1.0

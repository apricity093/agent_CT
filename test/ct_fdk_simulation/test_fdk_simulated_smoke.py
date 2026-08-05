from __future__ import annotations

import json

import pytest
import torch

from inv_framework.operators.ct import astra_adapter as astra_backend

from .simulation import (
    FDKSimulationConfig,
    make_modified_shepp_logan_3d,
    run_simulated_fdk,
    validate_result,
    write_artifacts,
)


def _require_astra_cuda() -> None:
    if not astra_backend._HAS_ASTRA:
        pytest.skip("astra-toolbox is not installed")
    if not torch.cuda.is_available() or not astra_backend.astra.use_cuda():
        pytest.skip("ASTRA CUDA and PyTorch CUDA are required")


def test_modified_shepp_logan_3d_is_deterministic_and_nonzero():
    phantom = make_modified_shepp_logan_3d(64, device="cpu")
    repeated = make_modified_shepp_logan_3d(64, device="cpu")
    assert phantom.shape == (1, 64, 64, 64)
    assert phantom.dtype == torch.float32
    assert torch.equal(phantom, repeated)
    assert torch.isfinite(phantom).all()
    assert phantom.max().item() == pytest.approx(0.02)
    assert phantom[..., 40, 27, 37].item() > 0.0


def test_small_simulated_cone_case_runs_fdk_and_writes_artifacts(tmp_path):
    _require_astra_cuda()
    config = FDKSimulationConfig(
        volume_size=16,
        num_angles=36,
        detector_rows=32,
        detector_cols=32,
        source_origin_distance=100.0,
        origin_detector_distance=100.0,
    )
    result = run_simulated_fdk(config, device="cuda")
    validate_result(result)

    assert result.geometry["measurement_layout"] == ["detector_row", "angle", "detector_col"]
    assert result.measurement.shape == (1, 32, 36, 32)
    assert result.reconstruction.shape == (1, 16, 16, 16)
    assert result.reconstruction.dtype == torch.float32
    assert result.reconstruction.device.type == "cuda"

    output_dir = tmp_path / "fdk_smoke"
    metrics = write_artifacts(result, output_dir)
    assert metrics["status"] == "success"
    assert metrics["reconstruction_max_abs"] > 0.0
    assert (output_dir / "comparison.png").is_file()
    assert (output_dir / "truth_axial.npy").is_file()
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["files"]) == {
        "comparison.png",
        "geometry.json",
        "metrics.json",
        "reconstruction_axial.npy",
        "truth_axial.npy",
    }

from __future__ import annotations

import json

import pytest
import torch

from inv_framework.ct.backends import astra_adapter as astra_backend

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

from __future__ import annotations

import importlib
import os
import time

import numpy as np
import pytest
import torch

from inv_framework.operators.ct import ASTRAFDKOperator3D, LEAPOperator3D
from inv_framework.operators.ct import astra_adapter as astra_backend
from inv_framework.solvers import FDKSolver

from .baselines import THRESHOLD_FDK_ASTRA
from .helpers import fdk_metrics, make_phantom_3d


def _require_astra_cuda():
    if not astra_backend._HAS_ASTRA:
        pytest.skip("astra-toolbox is not installed")
    astra = astra_backend.astra
    if not astra.use_cuda() or not torch.cuda.is_available():
        pytest.skip("ASTRA CUDA and PyTorch CUDA are required")
    return astra


@pytest.mark.astra
@pytest.mark.gpu
def test_astra_fdk_real_cone_workflow(record_property):
    astra = _require_astra_cuda()
    size = 16
    angles = np.arange(60, dtype=np.float32) * (2.0 * np.pi / 60.0)
    volume_geometry = astra.create_vol_geom(size, size, size)
    projection_geometry = astra.create_proj_geom(
        "cone", 1.0, 1.0, 32, 32, angles, 100.0, 100.0
    )
    operator = ASTRAFDKOperator3D(volume_geometry, projection_geometry)
    truth = make_phantom_3d(operator.domain_shape, device="cuda")
    measurement = operator.forward(truth).detach()

    started = time.perf_counter()
    reconstruction = FDKSolver().solve(measurement, operator)
    metrics = fdk_metrics(
        truth,
        measurement,
        reconstruction,
        operator,
        time.perf_counter() - started,
    )
    print(f"astra_fdk: {metrics}")

    assert operator.range_shape == (32, 60, 32)
    assert reconstruction.shape == truth.shape
    assert reconstruction.dtype == measurement.dtype
    assert reconstruction.device == measurement.device
    assert torch.isfinite(reconstruction).all()
    assert metrics["relative_error"] <= THRESHOLD_FDK_ASTRA["max_relative_error"]
    assert metrics["scale_aligned_data_residual"] <= THRESHOLD_FDK_ASTRA["max_scale_aligned_data_residual"]
    assert metrics["psnr"] >= THRESHOLD_FDK_ASTRA["min_psnr"]
    assert metrics["contrast_recovery"] >= THRESHOLD_FDK_ASTRA["min_contrast_recovery"]
    assert metrics["cnr"] >= THRESHOLD_FDK_ASTRA["min_cnr"]
    for key, value in metrics.items():
        record_property(key, value)


def _load_leap_factory():
    reference = os.environ.get("INVFRAMEWORK_LEAP_CONE_FACTORY")
    if not reference:
        pytest.skip("set INVFRAMEWORK_LEAP_CONE_FACTORY=module:function for real LEAP FDK")
    module_name, separator, function_name = reference.partition(":")
    if not separator:
        raise ValueError("INVFRAMEWORK_LEAP_CONE_FACTORY must use module:function syntax")
    return getattr(importlib.import_module(module_name), function_name)


@pytest.mark.leap
def test_leap_fdk_real_cone_workflow(record_property):
    model = _load_leap_factory()()
    operator = LEAPOperator3D(model)
    truth = make_phantom_3d(operator.domain_shape, device="cpu")
    measurement = operator.forward(truth).detach()

    started = time.perf_counter()
    reconstruction = FDKSolver().solve(measurement, operator)
    metrics = fdk_metrics(
        truth,
        measurement,
        reconstruction,
        operator,
        time.perf_counter() - started,
    )

    assert reconstruction.shape == truth.shape
    assert reconstruction.dtype == measurement.dtype
    assert reconstruction.device == measurement.device
    assert torch.isfinite(reconstruction).all()
    assert metrics["relative_error"] < 1.0
    assert metrics["scale_aligned_data_residual"] < 1.0
    assert metrics["psnr"] > metrics["zero_psnr"]
    assert metrics["contrast_recovery"] > 0.0
    assert metrics["cnr"] > 1.0
    for key, value in metrics.items():
        record_property(key, value)

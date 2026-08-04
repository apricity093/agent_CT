"""Shared deterministic data, solvers, and metrics for CT workflow tests."""

from __future__ import annotations

import math
import time
from typing import Any

import torch

from inv_framework.operators.ct import ParallelBeamRadon2D
from inv_framework.regularizers import TVRegularizer
from inv_framework.solvers import (
    CGLSSolver,
    LSQRSolver,
    OSSARTSolver,
    SARTSolver,
    TVFISTASolver,
    TikhonovSolver,
)
from inv_framework.utils.metrics import psnr, ssim


def make_phantom_2d(size: int, device: str | torch.device = "cpu") -> torch.Tensor:
    coordinates = torch.linspace(-1.0, 1.0, size, device=device)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    phantom = torch.zeros((size, size), dtype=torch.float32, device=device)
    phantom[((xx / 0.72) ** 2 + (yy / 0.88) ** 2) <= 1.0] = 0.55
    phantom[((xx + 0.27) ** 2 + (yy - 0.18) ** 2) <= 0.15**2] = 1.0
    phantom[((xx - 0.28) ** 2 + (yy + 0.20) ** 2) <= 0.19**2] = 0.22
    return phantom.unsqueeze(0).unsqueeze(0)


def make_case_2d(
    size: int = 32,
    num_angles: int = 24,
    noise_fraction: float = 0.0,
    seed: int = 1234,
) -> dict[str, Any]:
    angles = torch.arange(num_angles, dtype=torch.float32) * (math.pi / num_angles)
    operator = ParallelBeamRadon2D(image_size=size, angles=angles, device="cpu")
    truth = make_phantom_2d(size)
    measurement = operator.forward(truth).detach()
    if noise_fraction:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(measurement.shape, generator=generator)
        scale = float(noise_fraction) * measurement.norm() / noise.norm().clamp_min(1e-12)
        measurement = measurement + scale * noise
    return {
        "truth": truth,
        "measurement": measurement,
        "operator": operator,
        "noise_fraction": float(noise_fraction),
        "seed": int(seed),
    }


def solver_specs_2d() -> dict[str, Any]:
    return {
        "cgls": CGLSSolver(num_iterations=12, tol=0.0, min_value=0.0, max_value=1.2),
        "lsqr": LSQRSolver(
            num_iterations=12,
            atol=0.0,
            btol=0.0,
            min_value=0.0,
            max_value=1.2,
        ),
        "sart": SARTSolver(
            num_iterations=2,
            block_size=1,
            relaxation=0.35,
            min_value=0.0,
            max_value=1.2,
        ),
        "os_sart": OSSARTSolver(
            num_iterations=2,
            block_size=4,
            order_strategy="ordered",
            relaxation=0.35,
            min_value=0.0,
            max_value=1.2,
        ),
        "tikhonov": TikhonovSolver(
            reg_strength=1e-2,
            num_iterations=24,
            tolerance=1e-7,
            min_value=0.0,
            max_value=1.2,
        ),
        "tv_fista": TVFISTASolver(
            reg_strength=2e-3,
            num_iterations=12,
            tolerance=0.0,
            power_iterations=6,
            regularizer=TVRegularizer(num_iterations=24, tolerance=1e-5),
            min_value=0.0,
            max_value=1.2,
        ),
    }


def reconstruction_metrics(
    truth: torch.Tensor,
    measurement: torch.Tensor,
    reconstruction: torch.Tensor,
    operator,
    runtime_seconds: float,
) -> dict[str, float]:
    residual = operator.forward(reconstruction).detach() - measurement
    relative_error = (reconstruction - truth).norm() / truth.norm().clamp_min(1e-12)
    data_residual = residual.norm() / measurement.norm().clamp_min(1e-12)
    zero_psnr = psnr(torch.zeros_like(truth), truth, data_range=1.0).mean()
    return {
        "relative_error": float(relative_error.item()),
        "data_residual": float(data_residual.item()),
        "psnr": float(psnr(reconstruction, truth, data_range=1.0).mean().item()),
        "ssim": float(ssim(reconstruction, truth, data_range=1.0).mean().item()),
        "zero_psnr": float(zero_psnr.item()),
        "runtime_seconds": float(runtime_seconds),
    }


def run_solver_2d(name: str, solver, case: dict[str, Any]):
    started = time.perf_counter()
    reconstruction = solver.solve(case["measurement"], case["operator"])
    runtime_seconds = time.perf_counter() - started
    metrics = reconstruction_metrics(
        case["truth"],
        case["measurement"],
        reconstruction,
        case["operator"],
        runtime_seconds,
    )
    metrics["algorithm"] = name
    return reconstruction, metrics


def make_phantom_3d(
    shape: tuple[int, int, int],
    device: str | torch.device,
) -> torch.Tensor:
    z_size, y_size, x_size = shape
    zz, yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, z_size, device=device),
        torch.linspace(-1.0, 1.0, y_size, device=device),
        torch.linspace(-1.0, 1.0, x_size, device=device),
        indexing="ij",
    )
    phantom = torch.zeros(shape, dtype=torch.float32, device=device)
    phantom[(xx.square() + yy.square() + zz.square()) <= 0.52**2] = 0.65
    phantom[
        ((xx + 0.23).square() + (yy - 0.16).square() + (zz + 0.10).square())
        <= 0.17**2
    ] = 1.0
    return phantom.unsqueeze(0)


def fdk_metrics(
    truth: torch.Tensor,
    measurement: torch.Tensor,
    reconstruction: torch.Tensor,
    operator,
    runtime_seconds: float,
) -> dict[str, float]:
    predicted = operator.forward(reconstruction).detach()
    relative_error = (reconstruction - truth).norm() / truth.norm().clamp_min(1e-12)
    measurement_norm = measurement.norm().clamp_min(1e-12)
    data_residual = (predicted - measurement).norm() / measurement_norm
    projection_scale = (predicted * measurement).sum() / predicted.square().sum().clamp_min(1e-12)
    aligned_data_residual = (projection_scale * predicted - measurement).norm() / measurement_norm
    projection_norm_ratio = predicted.norm() / measurement_norm
    rmse = torch.sqrt((reconstruction - truth).square().mean()).clamp_min(1e-12)
    zero_rmse = torch.sqrt(truth.square().mean()).clamp_min(1e-12)
    foreground = truth >= 0.6
    background = truth == 0.0
    foreground_mean = reconstruction[foreground].mean()
    background_values = reconstruction[background]
    background_mean = background_values.mean()
    foreground_std = reconstruction[foreground].std(unbiased=False)
    background_std = background_values.std(unbiased=False)
    true_contrast = truth[foreground].mean() - truth[background].mean()
    contrast_recovery = (foreground_mean - background_mean) / true_contrast.clamp_min(1e-12)
    cnr = (foreground_mean - background_mean).abs() / torch.sqrt(
        foreground_std.square() + background_std.square() + 1e-12
    )
    return {
        "relative_error": float(relative_error.item()),
        "raw_data_residual": float(data_residual.item()),
        "scale_aligned_data_residual": float(aligned_data_residual.item()),
        "projection_scale": float(projection_scale.item()),
        "projection_norm_ratio": float(projection_norm_ratio.item()),
        "psnr": float((20.0 * torch.log10(1.0 / rmse)).item()),
        "zero_psnr": float((20.0 * torch.log10(1.0 / zero_rmse)).item()),
        "contrast_recovery": float(contrast_recovery.item()),
        "cnr": float(cnr.item()),
        "runtime_seconds": float(runtime_seconds),
    }

"""Regenerate the small P0 reference explicitly; never run by pytest.

Only the frozen native forward builds A. All updates below use NumPy float64
dense A/A.T and an independently assembled difference matrix. No production
solver, adjoint or regularizer generates the expected values.
"""

import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from p0_reference import explicit_matrix


def reference():
    size, views, iterations = 8, 7, 3
    angles = torch.linspace(0, math.pi, views + 1)[:-1]
    matrix = explicit_matrix(size, angles).numpy().astype(np.float64)
    truth = np.zeros((size, size))
    truth[2:6, 2:6] = 1
    truth[3:5, 4:6] = 0.5
    measurement = matrix @ truth.ravel()
    # D is assembled from signed unit entries, independent of production D.
    difference = np.zeros((2, size, size, size * size))
    for row in range(size):
        for col in range(size):
            index = row * size + col
            if row < size - 1:
                difference[0, row, col, index] = -1
                difference[0, row, col, index + size] = 1
            if col < size - 1:
                difference[1, row, col, index] = -1
                difference[1, row, col, index + 1] = 1
    difference = difference.reshape(2 * size * size, -1)

    def tv(x):
        return np.linalg.norm((difference @ x).reshape(2, -1), axis=0).sum()

    def prox(x, weight):
        dual = np.zeros(2 * size * size)
        momentum = dual.copy()
        acceleration = 1.
        for _ in range(30):
            primal = x - weight * difference.T @ momentum
            next_dual = momentum + (difference @ primal) / (8 * weight)
            next_dual = next_dual.reshape(2, -1)
            next_dual /= np.maximum(1, np.linalg.norm(next_dual, axis=0))
            next_dual = next_dual.ravel()
            next_acceleration = (1 + math.sqrt(1 + 4 * acceleration**2)) / 2
            momentum = next_dual + (acceleration - 1) / next_acceleration * (next_dual - dual)
            dual, acceleration = next_dual, next_acceleration
        return x - weight * difference.T @ dual

    records = {}
    for name in ("tikhonov", "tv_fista", "cgls", "sirt", "sart", "fbp"):
        x = np.zeros(size * size)
        trajectory = []

        def record(state):
            residual = matrix @ state - measurement
            objective = 0.5 * residual @ residual
            if name == "tikhonov":
                objective += 0.5 * 0.2 * (state @ state)
            if name == "tv_fista":
                objective += 0.08 * tv(state)
            trajectory.append({"residual": float(np.linalg.norm(residual) / np.linalg.norm(measurement)),
                               "objective": float(objective)})

        if name in ("tikhonov", "cgls"):
            normal = matrix.T @ matrix
            if name == "tikhonov":
                normal += 0.2 * np.eye(size * size)
            residual = matrix.T @ measurement
            direction = residual.copy()
            for _ in range(iterations):
                product = normal @ direction
                alpha = (residual @ residual) / (direction @ product)
                x += alpha * direction
                next_residual = residual - alpha * product
                direction = next_residual + (next_residual @ next_residual) / (residual @ residual) * direction
                residual = next_residual
                record(x)
        elif name == "tv_fista":
            momentum, acceleration = x.copy(), 1.
            for _ in range(iterations):
                # Legacy diagnostics record the accelerated input state.
                record(momentum)
                next_x = prox(momentum - 0.01 * matrix.T @ (matrix @ momentum - measurement), 0.01 * 0.08)
                next_acceleration = (1 + math.sqrt(1 + 4 * acceleration**2)) / 2
                momentum = next_x + (acceleration - 1) / next_acceleration * (next_x - x)
                x, acceleration = next_x, next_acceleration
        elif name in ("sirt", "sart"):
            for _ in range(iterations):
                blocks = [np.arange(views)] if name == "sirt" else [np.arange(i, min(i + 2, views)) for i in range(0, views, 2)]
                for block in blocks:
                    rows = np.concatenate([np.arange(i * size, (i + 1) * size) for i in block])
                    sub = matrix[rows]
                    rw, cw = sub.sum(1), sub.sum(0)
                    rw = np.divide(1., rw, out=np.zeros_like(rw), where=np.abs(rw) >= 1e-8)
                    cw = np.divide(1., cw, out=np.zeros_like(cw), where=np.abs(cw) >= 1e-8)
                    x -= cw * (sub.T @ (rw * (sub @ x - measurement[rows])))
                record(x)
        else:
            width = 2 * size
            ramp = np.zeros(width)
            ramp[0] = 0.25
            odd = np.arange(1, width, 2)
            ramp[odd] = -1 / (math.pi * np.minimum(odd, width - odd))**2
            filtered = np.fft.irfft(np.fft.rfft(measurement.reshape(views, size), n=width)
                                   * np.fft.rfft(ramp), n=width)[:, :size]
            x = math.pi / views * (matrix.T @ filtered.ravel())
            record(x)
        final_residual = matrix @ x - measurement
        records[name] = {"reconstruction": x.tolist(), "trajectory": trajectory,
                         "final_residual": float(np.linalg.norm(final_residual) / np.linalg.norm(measurement))}
    return {"reference": "frozen 011d48d forward basis matrix; NumPy float64 recurrences; independent dense D",
            "size": size, "angles": angles.tolist(), "measurement": measurement.tolist(),
            "iterations": iterations, "tuning": False,
            "parameters": {"tikhonov_strength": 0.2, "tv_strength": 0.08,
                           "tv_step": 0.01, "prox_iterations": 30, "sart_block_size": 2},
            "records": records}


if __name__ == "__main__":
    destination = ROOT / "test" / "data" / "p0_short_trajectories.json"
    destination.write_text(json.dumps(reference(), indent=2) + "\n", encoding="utf-8")
    print(destination)

from __future__ import annotations

import pytest
import torch

from inv_framework.operators.ct import ParallelBeamRadon2D
from inv_framework.solvers import (
    CGLSSolver,
    FDKSolver,
    LSQRSolver,
    OSSARTSolver,
    SARTSolver,
    TVFISTASolver,
    TikhonovSolver,
)


def test_seven_solver_wrappers_expose_solve_contract():
    solvers = [
        CGLSSolver(num_iterations=1),
        LSQRSolver(num_iterations=1),
        FDKSolver(),
        SARTSolver(num_iterations=1),
        OSSARTSolver(num_iterations=1),
        TikhonovSolver(num_iterations=1),
        TVFISTASolver(num_iterations=1, power_iterations=1),
    ]
    assert all(callable(getattr(solver, "solve", None)) for solver in solvers)


def test_fdk_contract_rejects_operator_without_backend():
    operator = ParallelBeamRadon2D(image_size=8, num_angles=4, device="cpu")
    measurement = torch.zeros(1, *operator.range_shape)
    with pytest.raises(NotImplementedError, match="operator.fdk"):
        FDKSolver().solve(measurement, operator)

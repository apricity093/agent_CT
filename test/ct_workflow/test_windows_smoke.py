from __future__ import annotations

import torch

from .helpers import make_case_2d, solver_specs_2d


def test_windows_fno_six_solver_smoke():
    case = make_case_2d(size=8, num_angles=4)
    for name, solver in solver_specs_2d().items():
        if hasattr(solver, "num_iterations"):
            solver.num_iterations = 1
        if hasattr(solver, "power_iterations"):
            solver.power_iterations = 1
        reconstruction = solver.solve(case["measurement"], case["operator"])
        assert reconstruction.shape == case["truth"].shape, name
        assert torch.isfinite(reconstruction).all(), name

"""Legacy public APIs against frozen independent dense P0 references."""

import json
from pathlib import Path

import pytest
import torch

from inv_framework.instrumentation import CountingLinearOperator
from inv_framework.operators.ct import radon_torch
from inv_framework.operators.ct.radon_torch import ParallelBeamRadon2D
from inv_framework.regularizers import TVRegularizer
from inv_framework.solvers import (TikhonovSolver, TVFISTASolver, CGLSSolver,
                                   SIRTSolver, SARTSolver, FBPSolver)
from inv_framework.solvers.base import SolveControl


@pytest.mark.parametrize("name", ["tikhonov", "tv_fista", "cgls", "sirt", "sart", "fbp"])
def test_frozen_short_trajectory(name, monkeypatch):
    baseline = json.loads((Path(__file__).resolve().parents[1] / "test/data/p0_short_trajectories.json").read_text())
    op = ParallelBeamRadon2D(baseline["size"], angles=torch.tensor(baseline["angles"]))
    y = torch.tensor(baseline["measurement"]).reshape(1, *op.range_shape)
    solvers = {
        "tikhonov": TikhonovSolver(reg_strength=0.2, num_iterations=3, tolerance=0),
        "tv_fista": TVFISTASolver(reg_strength=0.08, num_iterations=3, tolerance=0,
                                  step_size=0.01, regularizer=TVRegularizer(num_iterations=30, tolerance=0)),
        "cgls": CGLSSolver(num_iterations=3, tol=0),
        "sirt": SIRTSolver(num_iterations=3),
        "sart": SARTSolver(num_iterations=3, block_size=2),
        "fbp": FBPSolver(),
    }
    solver = solvers[name]
    expected = baseline["records"][name]
    target = torch.tensor(expected["reconstruction"]).reshape(1, *op.domain_shape)
    legacy = solver.solve(y, op)
    torch.testing.assert_close(legacy, target, atol=2e-6, rtol=1e-5)
    # Spy at the actual implementation boundary, including subset children.
    calls = {"forward": 0, "adjoint": 0}
    def spy(kind, implementation):
        def execute(*args, **kwargs):
            calls[kind] += 1
            return implementation(*args, **kwargs)
        return execute
    for kind in calls:
        attribute = f"_radon_{kind}_impl"
        monkeypatch.setattr(radon_torch, attribute, spy(kind, getattr(radon_torch, attribute)))
    counted = CountingLinearOperator(op)
    result = solver.solve_detailed(y, counted, control=SolveControl(tolerance=0, stop_on_convergence=False))
    torch.testing.assert_close(result.reconstruction, target, atol=2e-6, rtol=1e-5)
    assert torch.isfinite(result.reconstruction).all()
    assert result.status == ("completed_valid" if name == "fbp" else "max_iterations")
    assert result.actual_iterations == (0 if name == "fbp" else 3)
    residual = result.final_residual
    if name == "fbp":
        # Direct solve has no iterative diagnostics; runtime normally owns
        # this evaluation. Keep the additional endpoint forward counted.
        assert result.final_residual is None and not result.trajectory
        residual = float((counted.forward(result.reconstruction) - y).norm() / y.norm())
    assert residual == pytest.approx(expected["final_residual"], abs=2e-6, rel=1e-5)
    # Some recorders retain an initial point; the final three are updates.
    trajectory = list(result.trajectory) if name == "fbp" else [
        row for row in result.trajectory if row.iteration > 0]
    frozen_trajectory = [] if name == "fbp" else expected["trajectory"]
    assert len(trajectory) == len(frozen_trajectory)
    for actual, frozen in zip(trajectory, frozen_trajectory):
        assert actual.finite
        assert actual.normalized_data_residual == pytest.approx(frozen["residual"], abs=2e-6, rel=1e-5)
        assert actual.objective == pytest.approx(frozen["objective"], abs=2e-5, rel=1e-5)
    stats = counted.stats()
    assert calls == {"forward": stats["forward_calls"], "adjoint": stats["adjoint_calls"]}
    frozen_calls = {"tikhonov": (5, 5), "tv_fista": (6, 4), "cgls": (8, 4),
                    "sirt": (5, 4), "sart": (27, 24), "fbp": (1, 1)}
    assert (stats["forward_calls"], stats["adjoint_calls"]) == frozen_calls[name]
    if name == "tv_fista":
        assert stats["prox_calls"] == 4 and stats["prox_iterations"] == 120
        # P1: three checked candidate forwards and one returned-point mapping;
        # the accepted final candidate prediction is reused. Frozen numbers above are unchanged.
    error = float((result.reconstruction - target).norm() / target.norm())
    print(f"{name}: image_error={error:.3e}, residual={residual:.9g}, status={result.status}, "
          f"reason={result.stopping_reason}, calls={calls}")

from __future__ import annotations

import pytest
import torch

from inv_framework.convergence import ConvergenceStatus, classify_trajectory
from inv_framework.instrumentation import CountingLinearOperator, OperatorBudgetExceeded
from inv_framework.operators.base import LinearOperator
from inv_framework.solvers.specs import SOLVER_SPECS, validate_parameter_values


class IdentityOperator(LinearOperator):
    domain_shape = (1, 4, 4)
    range_shape = (1, 4, 4)

    def forward(self, x):
        return x

    def adjoint(self, y):
        return y


def test_registry_contains_complete_metadata_and_statistical_domain_gate():
    assert len(SOLVER_SPECS) == 12
    for spec in SOLVER_SPECS.values():
        assert spec.parameter_names == tuple(parameter.name for parameter in spec.parameters)
        assert spec.observation_domains
        assert spec.failure_modes
    result = validate_parameter_values("mlem", {}, observation_domain="log_projection")
    assert not result.valid
    assert any("observation domain" in error for error in result.errors)


@pytest.mark.parametrize(
    ("solver", "parameters", "message"),
    [
        ("sart", {"relaxation": 0.0}, "relaxation"),
        ("os_sart", {"subset_count": 9}, "subset_count"),
        ("tikhonov", {"reg_strength": -1.0}, "reg_strength"),
        ("cgls", {"num_iterations": 0}, "num_iterations"),
    ],
)
def test_parameter_constraints_are_checked_before_solver_construction(solver, parameters, message):
    result = validate_parameter_values(solver, parameters, views=8)
    assert not result.valid
    assert any(message in error for error in result.errors)


def test_convergence_classification_distinguishes_converged_stalled_diverged_and_budget():
    converged = classify_trajectory(
        [{"iteration": 1, "residual": 1.0, "relative_iterate_change": 0.2},
         {"iteration": 2, "residual": 1e-6, "relative_iterate_change": 1e-7}],
        tolerance=1e-5,
        max_iterations=5,
    )
    assert converged.status == ConvergenceStatus.CONVERGED

    stalled = classify_trajectory(
        [{"iteration": index, "residual": 1.0, "relative_iterate_change": 1e-10} for index in range(1, 7)],
        tolerance=1e-5,
        patience=5,
        max_iterations=10,
    )
    assert stalled.status == ConvergenceStatus.STALLED

    diverged = classify_trajectory(
        [{"iteration": index, "residual": float(index), "relative_iterate_change": 0.1} for index in range(1, 7)],
        tolerance=1e-5,
        patience=3,
    )
    assert diverged.status == ConvergenceStatus.DIVERGED

    budget = classify_trajectory(
        [{"iteration": index, "residual": 1.0 / index, "relative_iterate_change": 0.1} for index in range(1, 4)],
        tolerance=1e-8,
        max_iterations=3,
    )
    assert budget.status == ConvergenceStatus.MAX_ITERATIONS


def test_counting_operator_tracks_calls_and_enforces_budget():
    counted = CountingLinearOperator(IdentityOperator(), max_forward_calls=1)
    x = torch.ones(1, 1, 4, 4)
    assert torch.equal(counted.forward(x), x)
    with pytest.raises(OperatorBudgetExceeded):
        counted.forward(x)
    counted.adjoint(x)
    assert counted.stats()["forward_calls"] == 1
    assert counted.stats()["adjoint_calls"] == 1
    assert counted.stats()["total_operator_calls"] == 2

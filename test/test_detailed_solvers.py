from __future__ import annotations

import torch
import pytest

from inv_framework.operators.base import LinearOperator
from inv_framework.benchmarks import make_heldout_projection_split
from inv_framework.instrumentation import CountingLinearOperator
from inv_framework.regularizers import TVRegularizer
from inv_framework.solvers import CGLSSolver, SolveControl, SIRTSolver, TikhonovSolver, TVFISTASolver
from inv_framework.solvers.statistical import MLEMSolver


class IdentityOperator(LinearOperator):
    domain_shape = (1, 4, 4)
    range_shape = (1, 4, 4)

    def forward(self, x):
        return x

    def adjoint(self, y):
        return y


def test_detailed_result_reports_native_stop_and_actual_iterations():
    operator = IdentityOperator()
    measurement = torch.ones(1, *operator.range_shape)
    result = CGLSSolver(num_iterations=8, tol=1e-6).solve_detailed(measurement, operator)

    assert result.status == "converged"
    assert result.actual_iterations == 1
    assert result.stopping_reason == "normal_residual_tolerance"
    assert result.trajectory and result.trajectory[-1].finite
    assert result.final_residual is not None and result.final_residual < 1e-6


def test_detailed_result_distinguishes_iteration_budget():
    operator = IdentityOperator()
    measurement = torch.ones(1, *operator.range_shape)
    result = SIRTSolver(num_iterations=3).solve_detailed(
        measurement, operator, control=SolveControl(max_iterations=3, tolerance=0.0)
    )

    assert result.status == "max_iterations"
    assert result.actual_iterations == 3
    assert result.stopping_reason == "maximum_iterations_reached"


def test_sirt_policy_requires_five_consecutive_native_checks():
    operator = IdentityOperator()
    measurement = torch.ones(1, *operator.range_shape)
    result = SIRTSolver(num_iterations=12).solve_detailed(
        measurement,
        operator,
        control=SolveControl(
            max_iterations=12,
            min_iterations=5,
            check_every=1,
            patience=5,
            discrepancy_target=1e-6,
            relative_iterate_tolerance=1e-6,
        ),
    )
    assert result.status == "converged"
    assert result.actual_iterations == 9
    assert result.stopping_reason == "discrepancy_and_relative_iterate_change_patience"
    assert result.trajectory[-1].consecutive_criteria_count == 5


def test_sirt_policy_never_relabels_budget_as_converged():
    operator = IdentityOperator()
    measurement = torch.ones(1, *operator.range_shape)
    result = SIRTSolver(num_iterations=8).solve_detailed(
        measurement,
        operator,
        control=SolveControl(
            max_iterations=8,
            min_iterations=5,
            patience=5,
            discrepancy_target=1e-6,
            relative_iterate_tolerance=1e-6,
        ),
    )
    assert result.status == "max_iterations"
    assert result.stopping_reason == "maximum_iterations_reached"


def test_detailed_callback_can_cancel_at_checkpoint():
    operator = IdentityOperator()
    measurement = torch.ones(1, *operator.range_shape)
    seen = []

    result = SIRTSolver(num_iterations=5).solve_detailed(
        measurement,
        operator,
        callback=lambda record: seen.append(record.iteration) or False,
    )

    assert result.status == "cancelled"
    assert seen == [1]


def test_statistical_solver_rejects_signed_observations():
    operator = IdentityOperator()
    measurement = torch.zeros(1, *operator.range_shape)
    measurement[..., 0, 0] = -1.0
    with pytest.raises(ValueError, match="nonnegative"):
        MLEMSolver(num_iterations=1).solve(measurement, operator)


def test_detailed_result_marks_nonfinite_state_as_numerical_failure():
    operator = IdentityOperator()
    measurement = torch.full((1, *operator.range_shape), float("nan"))
    result = CGLSSolver(num_iterations=2, tol=0.0).solve_detailed(measurement, operator)

    assert result.status == "numerical_failure"
    assert result.stopping_reason == "non_finite_solver_state"


def test_heldout_projection_split_is_stable_balanced_and_disjoint():
    first = make_heldout_projection_split("case", [0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    second = make_heldout_projection_split("case", [0.0, 0.5, 1.0, 1.5, 2.0, 2.5])

    assert first.split_sha256 == second.split_sha256
    assert set().union(*map(set, first.validation_folds)) == set(range(6))
    assert sum(len(fold) for fold in first.validation_folds) == 6
    assert all(
        set(left).isdisjoint(right)
        for index, left in enumerate(first.validation_folds)
        for right in first.validation_folds[index + 1 :]
    )
    assert max(map(len, first.validation_folds)) - min(map(len, first.validation_folds)) <= 1


def test_cached_tikhonov_accounting_matches_equal_call_formula():
    operator = CountingLinearOperator(IdentityOperator())
    measurement = torch.ones(1, *operator.range_shape)
    result = TikhonovSolver(num_iterations=3, tolerance=0.0).solve_detailed(
        measurement,
        operator,
        control=SolveControl(max_iterations=3, tolerance=0.0),
    )

    assert result.status == "max_iterations"
    assert operator.stats()["forward_calls"] == 3 + 2
    assert operator.stats()["adjoint_calls"] == 3 + 2


def test_tv_fista_accounting_counts_one_gradient_forward_per_iteration():
    operator = CountingLinearOperator(IdentityOperator())
    measurement = torch.ones(1, *operator.range_shape)
    result = TVFISTASolver(
        num_iterations=3,
        step_size=0.1,
        tolerance=0.0,
        regularizer=TVRegularizer(num_iterations=1),
    ).solve_detailed(
        measurement,
        operator,
        control=SolveControl(max_iterations=3, tolerance=0.0),
    )

    assert result.status == "max_iterations"
    assert operator.stats()["forward_calls"] == 3 + 1
    assert operator.stats()["adjoint_calls"] == 3

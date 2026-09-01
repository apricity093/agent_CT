from __future__ import annotations

import json

import torch

from inv_framework.convergence import (
    ConvergenceReport,
    ConvergenceStatus,
    classify_trajectory,
    normalize_status,
    post_run_validation,
    sample_trajectory,
)
from inv_framework.solvers.base import SolveResult


class IdentityOperator:
    domain_shape = (4,)
    range_shape = (4,)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def adjoint(self, value: torch.Tensor) -> torch.Tensor:
        return value


def test_legacy_statuses_normalize_at_the_report_boundary():
    report = ConvergenceReport(
        status="numerical_failure",
        stopping_reason="non_finite_solver_state",
        algorithm="sirt",
    )

    payload = report.to_dict()
    assert report.status == ConvergenceStatus.NUMERICAL_ERROR
    assert payload["status"] == "numerical_error"
    assert payload["legacy_status"] == "numerical_failure"
    assert payload["reason_class"] == "numerical"
    assert ConvergenceReport.from_dict(payload).to_dict() == payload


def test_core_statuses_do_not_treat_missing_evidence_as_convergence():
    report = post_run_validation(
        torch.zeros((1, 4)),
        measurement=torch.zeros((1, 4)),
        predicted_measurement=torch.zeros((1, 4)),
        operator=IdentityOperator(),
        trajectory=None,
        iterations=2,
        max_iterations=5,
        algorithm="sirt",
    )
    assert report.status == ConvergenceStatus.NUMERICAL_ERROR
    assert report.stopping_reason == "no_iteration_trajectory"

    at_budget = post_run_validation(
        torch.zeros((1, 4)),
        measurement=torch.zeros((1, 4)),
        predicted_measurement=torch.zeros((1, 4)),
        operator=IdentityOperator(),
        trajectory=None,
        iterations=5,
        max_iterations=5,
        algorithm="sirt",
    )
    assert at_budget.status == ConvergenceStatus.MAX_ITERATIONS


def test_direct_and_invalid_parameter_reports_are_structured():
    direct = post_run_validation(
        torch.ones((1, 4)),
        measurement=torch.ones((1, 4)),
        predicted_measurement=torch.ones((1, 4)),
        operator=IdentityOperator(),
        trajectory=None,
        non_iterative=True,
        algorithm="fbp",
    )
    invalid = ConvergenceReport.invalid_parameters(
        ["num_iterations must be positive"],
        algorithm="sirt",
        parameters={"num_iterations": 0},
    )
    assert direct.status == ConvergenceStatus.COMPLETED_VALID
    assert direct.iterations == 0
    assert invalid.status == ConvergenceStatus.INVALID_PARAMETERS
    assert invalid.iterations == 0
    assert invalid.to_dict()["terminal_evidence"]["validation_errors"]


def test_max_iterations_is_preserved_even_when_terminal_residual_is_small():
    report = classify_trajectory(
        [
            {"iteration": 1, "normalized_residual": 1.0, "relative_iterate_change": 0.1},
            {"iteration": 2, "normalized_residual": 0.01, "relative_iterate_change": 0.001},
        ],
        max_iterations=2,
        tolerance=0.1,
        patience=2,
        algorithm="sirt",
    )
    assert report.status == ConvergenceStatus.MAX_ITERATIONS
    assert report.status != ConvergenceStatus.CONVERGED


def test_terminal_patience_evidence_is_retained_and_json_safe():
    rows = [
        {"iteration": index, "value": float("nan") if index == 9 else float(index)}
        for index in range(1, 11)
    ]
    sampled = sample_trajectory(rows, max_points=3, patience=4)
    assert [row["iteration"] for row in sampled[-4:]] == [7, 8, 9, 10]
    report = ConvergenceReport(
        status="stalled",
        stopping_reason="stalled_before_convergence",
        iterations=10,
        trajectory=tuple(sampled),
        terminal_evidence={"patience_window": sampled[-4:]},
        algorithm="sirt",
    )
    serialized = json.dumps(report.to_dict(), allow_nan=False)
    assert "NaN" not in serialized
    assert json.loads(serialized)["terminal_evidence"]["patience_window"][2]["value"] is None


def test_solve_result_normalizes_legacy_numerical_failure():
    result = SolveResult(
        reconstruction=torch.zeros((1, 4)),
        actual_iterations=1,
        status="numerical_failure",
        stopping_reason="non_finite_solver_state",
        metadata={"algorithm": "sirt", "max_iterations": 3},
    )
    assert result.status == "numerical_error"
    payload = result.to_dict()
    assert payload["convergence_status"] == "numerical_error"
    assert payload["legacy_status"] == "numerical_failure"


def test_status_normalization_keeps_core_budget_and_direct_states_distinct():
    assert normalize_status("partial", algorithm="sirt", iterations=2, max_iterations=5).value == "numerical_error"
    assert normalize_status("partial", algorithm="sirt", iterations=5, max_iterations=5).value == "max_iterations"
    assert normalize_status("partial", algorithm="fbp", direct=True).value == "completed_valid"

from __future__ import annotations

import pytest
import torch

from inv_framework.convergence import confirm_endpoint
from inv_framework.instrumentation import CountingLinearOperator
from inv_framework.operators.base import LinearOperator
from inv_framework.solvers import (
    OSSARTSolver,
    SARTSolver,
    SIRTSolver,
    LandweberSolver,
    SolveControl,
)


class IdentityOperator(LinearOperator):
    domain_shape = (4,)
    range_shape = (4,)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def adjoint(self, value: torch.Tensor) -> torch.Tensor:
        return value


class ZeroOperator(LinearOperator):
    domain_shape = (4,)
    range_shape = (4,)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(value)

    def adjoint(self, value: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(value)


class DenseAngleOperator(LinearOperator):
    def __init__(self, matrix: torch.Tensor, num_angles: int, detector_size: int, calls=None):
        self.matrix = matrix
        self.num_angles = int(num_angles)
        self.detector_size = int(detector_size)
        self.domain_shape = (int(matrix.shape[1]),)
        self.range_shape = (self.num_angles, self.detector_size)
        self.calls = [] if calls is None else calls

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        matrix = self.matrix.to(device=x.device, dtype=x.dtype)
        values = x.reshape(x.shape[0], -1).matmul(matrix.t())
        return values.reshape(x.shape[0], self.num_angles, self.detector_size)

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        matrix = self.matrix.to(device=y.device, dtype=y.dtype)
        return y.reshape(y.shape[0], -1).matmul(matrix)

    def subset(self, indices: torch.Tensor):
        selected = tuple(int(value) for value in indices.detach().cpu().tolist())
        self.calls.append(selected)
        rows = []
        for angle in selected:
            start = angle * self.detector_size
            rows.extend(range(start, start + self.detector_size))
        return DenseAngleOperator(
            self.matrix[rows], len(selected), self.detector_size, calls=self.calls
        )


class BadAdjointOperator(LinearOperator):
    """Finite deliberately inconsistent adjoint used to exercise divergence."""

    def __init__(self, matrix: torch.Tensor):
        self.matrix = matrix
        self.domain_shape = (int(matrix.shape[1]),)
        self.range_shape = (int(matrix.shape[0]),)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value.reshape(value.shape[0], -1)

    def adjoint(self, value: torch.Tensor) -> torch.Tensor:
        matrix = self.matrix.to(device=value.device, dtype=value.dtype)
        return value.reshape(value.shape[0], -1).matmul(matrix.t())


class BadAngleOperator(LinearOperator):
    def __init__(self, matrix: torch.Tensor, selected=None, calls=None):
        self.matrix = matrix
        self.selected = tuple(range(matrix.shape[0])) if selected is None else tuple(selected)
        self.calls = [] if calls is None else calls
        self.domain_shape = (int(matrix.shape[1]),)
        self.range_shape = (len(self.selected), 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        values = value.reshape(value.shape[0], -1).index_select(
            1, torch.as_tensor(self.selected, dtype=torch.long, device=value.device)
        )
        return values.unsqueeze(-1)

    def adjoint(self, value: torch.Tensor) -> torch.Tensor:
        matrix = self.matrix.to(device=value.device, dtype=value.dtype)
        selected = torch.as_tensor(self.selected, dtype=torch.long, device=value.device)
        return value.reshape(value.shape[0], -1).matmul(matrix.index_select(0, selected))

    def subset(self, indices: torch.Tensor):
        selected = tuple(self.selected[int(value)] for value in indices.detach().cpu().tolist())
        self.calls.append(selected)
        return BadAngleOperator(self.matrix, selected=selected, calls=self.calls)


def _control(
    maximum: int = 12,
    *,
    target: float = 1e-6,
    relative: float = 1e-6,
    patience: int = 2,
    stall: bool = False,
    divergence: bool = False,
    stop: bool = True,
    metadata=None,
) -> SolveControl:
    return SolveControl(
        max_iterations=maximum,
        min_iterations=1,
        check_every=1,
        patience=patience,
        stop_on_convergence=stop,
        discrepancy_target=target,
        relative_iterate_tolerance=relative,
        stall_enabled=stall,
        stall_relative_iterate_tolerance=1e-8,
        stall_patience=2,
        divergence_enabled=divergence,
        divergence_relative_increase_tolerance=1e-4,
        divergence_patience=2,
        metadata=metadata or {},
    )


def _angle_identity() -> DenseAngleOperator:
    return DenseAngleOperator(torch.eye(4), num_angles=4, detector_size=1)


@pytest.mark.parametrize("name", ["sirt", "landweber", "sart", "os_sart"])
def test_row_action_policy_converges_only_after_native_patience(name):
    measurement = torch.ones(1, 4) if name in {"sirt", "landweber"} else torch.ones(1, 4, 1)
    if name == "sirt":
        result = SIRTSolver(num_iterations=12).solve_detailed(
            measurement, IdentityOperator(), control=_control()
        )
    elif name == "landweber":
        result = LandweberSolver(num_iterations=12, step_size=1.0).solve_detailed(
            measurement, IdentityOperator(),
            control=_control(metadata={"operator_norm_estimate": 1.0}),
        )
    elif name == "sart":
        result = SARTSolver(num_iterations=12, block_size=2, relaxation=1.0).solve_detailed(
            measurement, _angle_identity(), control=_control()
        )
    else:
        result = OSSARTSolver(
            num_iterations=12, subset_indices=[(0, 1), (2, 3)], relaxation=1.0
        ).solve_detailed(measurement, _angle_identity(), control=_control())

    assert result.status == "converged"
    assert result.stopping_reason in {
        "discrepancy_and_relative_iterate_change_patience",
        "discrepancy_and_relative_epoch_change_patience",
    }
    assert result.actual_iterations == 3
    assert result.trajectory[-1].consecutive_criteria_count == 2
    if name in {"sart", "os_sart"}:
        assert all(record.epoch == record.iteration for record in result.trajectory)
        assert all(record.subset is None for record in result.trajectory)


@pytest.mark.parametrize("name", ["sirt", "landweber", "sart", "os_sart"])
def test_row_action_max_budget_is_not_convergence(name):
    measurement = torch.ones(1, 4) if name in {"sirt", "landweber"} else torch.ones(1, 4, 1)
    if name == "sirt":
        result = SIRTSolver(num_iterations=3).solve_detailed(
            measurement, ZeroOperator(), control=_control(3, target=0.0)
        )
    elif name == "landweber":
        result = LandweberSolver(num_iterations=3, step_size=1.0).solve_detailed(
            measurement, ZeroOperator(),
            control=_control(3, target=0.0, metadata={"operator_norm_estimate": 0.0}),
        )
    elif name == "sart":
        result = SARTSolver(num_iterations=3, block_size=2, relaxation=0.5).solve_detailed(
            measurement, DenseAngleOperator(torch.zeros(4, 4), 4, 1),
            control=_control(3, target=0.0),
        )
    else:
        result = OSSARTSolver(
            num_iterations=3, subset_indices=[(0, 1), (2, 3)], relaxation=0.5
        ).solve_detailed(
            measurement, DenseAngleOperator(torch.zeros(4, 4), 4, 1),
            control=_control(3, target=0.0),
        )
    assert result.status == "max_iterations"
    assert result.stopping_reason in {"maximum_iterations_reached", "maximum_epochs_reached"}


@pytest.mark.parametrize("name", ["sirt", "landweber", "sart", "os_sart"])
def test_row_action_stall_is_reported_before_budget(name):
    measurement = torch.ones(1, 4) if name in {"sirt", "landweber"} else torch.ones(1, 4, 1)
    if name == "sirt":
        result = SIRTSolver(num_iterations=8).solve_detailed(
            measurement, ZeroOperator(), control=_control(8, target=0.0, stall=True)
        )
    elif name == "landweber":
        result = LandweberSolver(num_iterations=8, step_size=1.0).solve_detailed(
            measurement, ZeroOperator(),
            control=_control(8, target=0.0, stall=True, metadata={"operator_norm_estimate": 0.0}),
        )
    elif name == "sart":
        result = SARTSolver(num_iterations=8, block_size=2, relaxation=0.5).solve_detailed(
            measurement, DenseAngleOperator(torch.zeros(4, 4), 4, 1),
            control=_control(8, target=0.0, stall=True),
        )
    else:
        result = OSSARTSolver(
            num_iterations=8, subset_indices=[(0, 1), (2, 3)], relaxation=0.5
        ).solve_detailed(
            measurement, DenseAngleOperator(torch.zeros(4, 4), 4, 1),
            control=_control(8, target=0.0, stall=True),
        )
    assert result.status == "stalled"
    assert result.stopping_reason == "stalled_before_discrepancy"
    assert result.actual_iterations == 2


@pytest.mark.parametrize("name", ["sirt", "landweber", "sart", "os_sart"])
def test_row_action_divergence_has_priority_over_max_budget(name):
    bad = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
    measurement = torch.tensor([[1.0, -1.0]])
    if name in {"sirt", "landweber"}:
        operator = BadAdjointOperator(bad)
        if name == "sirt":
            result = SIRTSolver(num_iterations=8).solve_detailed(
                measurement, operator, control=_control(8, target=0.0, divergence=True)
            )
        else:
            result = LandweberSolver(num_iterations=8, step_size=0.5).solve_detailed(
                measurement, operator,
                control=_control(8, target=0.0, divergence=True, metadata={"operator_norm_estimate": 1.0}),
            )
    else:
        operator = BadAngleOperator(bad)
        if name == "sart":
            result = SARTSolver(num_iterations=8, block_size=2, relaxation=1.0).solve_detailed(
                measurement.reshape(1, 2, 1), operator, control=_control(8, target=0.0, divergence=True)
            )
        else:
            result = OSSARTSolver(
                num_iterations=8, subset_indices=[(0, 1)], relaxation=1.0
            ).solve_detailed(
                measurement.reshape(1, 2, 1), operator, control=_control(8, target=0.0, divergence=True)
            )
    assert result.status == "diverged"
    assert result.stopping_reason == "persistent_residual_or_objective_increase"


@pytest.mark.parametrize("name", ["sirt", "landweber", "sart", "os_sart"])
def test_row_action_nonfinite_measurement_is_immediate_numerical_error(name):
    measurement = torch.full((1, 4), float("nan")) if name in {"sirt", "landweber"} else torch.full((1, 4, 1), float("inf"))
    if name == "sirt":
        result = SIRTSolver(num_iterations=5).solve_detailed(measurement, IdentityOperator(), control=_control(5))
    elif name == "landweber":
        result = LandweberSolver(num_iterations=5, step_size=1.0).solve_detailed(
            measurement, IdentityOperator(), control=_control(5, metadata={"operator_norm_estimate": 1.0})
        )
    elif name == "sart":
        result = SARTSolver(num_iterations=5, block_size=2).solve_detailed(
            measurement, _angle_identity(), control=_control(5)
        )
    else:
        result = OSSARTSolver(num_iterations=5, subset_indices=[(0, 1), (2, 3)]).solve_detailed(
            measurement, _angle_identity(), control=_control(5)
        )
    assert result.status == "numerical_error"
    assert result.stopping_reason == "non_finite_measurement"
    assert result.actual_iterations == 0


def test_landweber_step_bound_is_preflighted_without_solver_iterations():
    operator = CountingLinearOperator(IdentityOperator())
    result = LandweberSolver(num_iterations=5, step_size=2.0).solve_detailed(
        torch.ones(1, 4), operator,
        control=_control(5, metadata={"operator_norm_estimate": 1.0}),
    )
    assert result.status == "invalid_parameters"
    assert result.stopping_reason == "parameter_validation_failed"
    assert result.actual_iterations == 0
    assert operator.stats()["total_operator_calls"] == 0


def test_landweber_tolerance_changes_first_valid_stop():
    operator = IdentityOperator()
    measurement = torch.ones(1, 4)
    loose = LandweberSolver(num_iterations=30, step_size=0.5).solve_detailed(
        measurement, operator,
        control=_control(30, target=0.1, relative=0.1, patience=1, metadata={"operator_norm_estimate": 1.0}),
    )
    strict = LandweberSolver(num_iterations=30, step_size=0.5).solve_detailed(
        measurement, operator,
        control=_control(30, target=0.1, relative=0.001, patience=1, metadata={"operator_norm_estimate": 1.0}),
    )
    assert loose.status == strict.status == "converged"
    assert strict.actual_iterations > loose.actual_iterations


def test_subset_policy_records_only_complete_epoch_boundaries():
    operator = _angle_identity()
    measurement = torch.ones(1, 4, 1)
    seen = []
    result = SARTSolver(num_iterations=4, block_size=2).solve_detailed(
        measurement,
        operator,
        control=_control(4, patience=1),
        callback=lambda record: seen.append((record.iteration, record.epoch, record.subset)) or True,
    )
    assert result.status == "converged"
    assert seen and all(subset is None for _iteration, _epoch, subset in seen)
    assert [iteration for iteration, _epoch, _subset in seen] == [epoch for _iteration, epoch, _subset in seen]
    assert all(record.metadata["complete_sweep"] for record in result.trajectory)


def test_endpoint_native_and_trajectory_mismatch_blocks_strict_convergence():
    report = confirm_endpoint(
        algorithm="sirt",
        reconstruction=torch.ones(1, 4),
        measurement=torch.ones(1, 4),
        operator=IdentityOperator(),
        trajectory=[{
            "iteration": 1,
            "normalized_data_residual": 1.0,
            "native_criterion_value": 0.0,
            "relative_iterate_change": 0.0,
            "criteria": {"discrepancy": True, "relative_iterate_change": True},
            "consecutive_criteria_count": 1,
            "metadata": {"checked": True},
        }],
        solver_status="converged",
        solver_stopping_reason="discrepancy_and_relative_iterate_change_patience",
        iterations=1,
        max_iterations=5,
        parameters={"min_value": None, "max_value": None},
        operator_norm_estimate=1.0,
        policy={
            "min_iterations": 1,
            "check_every": 1,
            "patience": 1,
            "relative_iterate_tolerance": 1e-6,
            "effective": {"discrepancy_target": 1e-6},
            "endpoint_confirmation": {"require_trajectory_consistency": True},
        },
    )
    assert report["passed"] is False
    assert "trajectory_endpoint_metric_mismatch" in report["reasons"]


def test_endpoint_rejects_subset_level_terminal_evidence():
    report = confirm_endpoint(
        algorithm="sart",
        reconstruction=torch.ones(1, 4),
        measurement=torch.ones(1, 4, 1),
        operator=_angle_identity(),
        trajectory=[{
            "iteration": 1,
            "epoch": 1,
            "subset": 0,
            "normalized_data_residual": 0.0,
            "native_criterion_value": 0.0,
            "criteria": {"discrepancy": True, "relative_epoch_change": True},
            "consecutive_criteria_count": 1,
            "metadata": {"checked": True, "complete_sweep": False},
        }],
        solver_status="converged",
        solver_stopping_reason="discrepancy_and_relative_epoch_change_patience",
        iterations=1,
        max_iterations=5,
        parameters={"block_size": 2, "relaxation": 1.0, "eps": 1e-8},
        operator_norm_estimate=1.0,
        policy={
            "min_iterations": 1,
            "check_every": 1,
            "patience": 1,
            "relative_iterate_tolerance": 1e-6,
            "effective": {"discrepancy_target": 1e-6},
            "endpoint_confirmation": {"require_trajectory_consistency": True},
        },
    )
    assert report["passed"] is False
    assert "subset_level_termination" in report["reasons"]


@pytest.mark.parametrize(
    ("name", "expected_forward", "expected_adjoint"),
    [
        ("sirt", 6, 4),
        ("landweber", 5, 3),
        ("sart", 16, 12),
        ("os_sart", 16, 12),
    ],
)
def test_row_action_call_accounting_separates_endpoint_verification(name, expected_forward, expected_adjoint):
    n = 3
    if name == "sirt":
        operator = CountingLinearOperator(IdentityOperator())
        result = SIRTSolver(num_iterations=n).solve_detailed(
            torch.ones(1, 4), operator,
            control=_control(n, target=0.0, stop=False),
        )
    elif name == "landweber":
        operator = CountingLinearOperator(IdentityOperator())
        result = LandweberSolver(num_iterations=n, step_size=0.5).solve_detailed(
            torch.ones(1, 4), operator,
            control=_control(n, target=0.0, stop=False, metadata={"operator_norm_estimate": 1.0}),
        )
    elif name == "sart":
        operator = CountingLinearOperator(_angle_identity())
        result = SARTSolver(num_iterations=n, block_size=2).solve_detailed(
            torch.ones(1, 4, 1), operator,
            control=_control(n, target=0.0, stop=False),
        )
    else:
        operator = CountingLinearOperator(_angle_identity())
        result = OSSARTSolver(num_iterations=n, subset_indices=[(0, 1), (2, 3)]).solve_detailed(
            torch.ones(1, 4, 1), operator,
            control=_control(n, target=0.0, stop=False),
        )
    assert operator.stats()["forward_calls"] == expected_forward
    assert operator.stats()["adjoint_calls"] == expected_adjoint
    assert result.resources["verification_forward_calls"] == 1
    assert result.resources["optimization_forward_calls"] == expected_forward - 1
    assert result.resources["optimization_adjoint_calls"] == expected_adjoint

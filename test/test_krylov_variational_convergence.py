from __future__ import annotations

import pytest
import torch

from inv_framework.convergence import confirm_endpoint
from inv_framework.instrumentation import CountingLinearOperator
from inv_framework.operators.base import LinearOperator
from inv_framework.regularizers import TVRegularizer
from inv_framework.solvers import (
    CGLSSolver,
    LSQRSolver,
    SolveControl,
    TikhonovSolver,
    TVFISTASolver,
)


class DenseLinearOperator(LinearOperator):
    def __init__(self, matrix: torch.Tensor):
        self.matrix = matrix
        self.domain_shape = (matrix.shape[1],)
        self.range_shape = (matrix.shape[0],)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        matrix = self.matrix.to(device=value.device, dtype=value.dtype)
        return value.reshape(value.shape[0], -1) @ matrix.T

    def adjoint(self, value: torch.Tensor) -> torch.Tensor:
        matrix = self.matrix.to(device=value.device, dtype=value.dtype)
        return value.reshape(value.shape[0], -1) @ matrix


class IdentityImageOperator(LinearOperator):
    domain_shape = (1, 4, 4)
    range_shape = (1, 4, 4)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def adjoint(self, value: torch.Tensor) -> torch.Tensor:
        return value


class ZeroOperator(LinearOperator):
    domain_shape = (2,)
    range_shape = (2,)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (value.shape[0], *self.range_shape), dtype=value.dtype, device=value.device
        )

    def adjoint(self, value: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (value.shape[0], *self.domain_shape), dtype=value.dtype, device=value.device
        )


class ZeroImageOperator(LinearOperator):
    domain_shape = (1, 4, 4)
    range_shape = (1, 4, 4)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (value.shape[0], *self.range_shape), dtype=value.dtype, device=value.device
        )

    def adjoint(self, value: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (value.shape[0], *self.domain_shape), dtype=value.dtype, device=value.device
        )


def _policy(
    *,
    target: float | None,
    max_iterations: int,
    patience: int = 1,
    normal: float = 1e-4,
    iterate: float = 1e-4,
    objective: float = 1e-5,
    mapping: float = 1e-4,
) -> SolveControl:
    return SolveControl(
        max_iterations=max_iterations,
        min_iterations=1,
        check_every=1,
        patience=patience,
        discrepancy_target=target,
        normalized_normal_residual_tolerance=normal,
        relative_iterate_tolerance=iterate,
        relative_objective_tolerance=objective,
        prox_gradient_mapping_tolerance=mapping,
    )


def _endpoint_policy(
    *,
    target: float | None,
    normal: float = 1e-4,
    iterate: float = 1e-4,
    objective: float = 1e-5,
    mapping: float = 1e-4,
    patience: int = 1,
) -> dict[str, object]:
    return {
        "discrepancy_target": target,
        "normalized_normal_residual_tolerance": normal,
        "relative_iterate_tolerance": iterate,
        "relative_objective_tolerance": objective,
        "prox_gradient_mapping_tolerance": mapping,
        "min_iterations": 1,
        "check_every": 1,
        "patience": patience,
        "endpoint_confirmation": {"require_trajectory_consistency": True},
    }


@pytest.mark.parametrize(
    "solver",
    [
        CGLSSolver(num_iterations=3, tol=0.0),
        LSQRSolver(num_iterations=3, atol=0.0, btol=0.0),
    ],
)
def test_krylov_policy_uses_operator_norm_not_squared(solver) -> None:
    matrix = torch.diag(torch.tensor([2.0, 0.5], dtype=torch.float64))
    operator = DenseLinearOperator(matrix)
    measurement = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    result = solver.solve_detailed(
        measurement,
        operator,
        control=_policy(target=10.0, max_iterations=3, normal=10.0, iterate=1e15),
        operator_norm_squared=4.0,
    )

    assert result.status == "converged"
    assert result.actual_iterations == 1
    row = result.trajectory[-1]
    residual = operator.forward(result.reconstruction) - measurement
    expected_denominator = 2.0 * float(residual.norm().item())
    expected_value = float(operator.adjoint(residual).norm().item()) / expected_denominator

    assert result.metadata["operator_norm_estimate"] == pytest.approx(2.0)
    assert result.metadata["operator_norm_squared"] == pytest.approx(4.0)
    assert row.metadata["normal_residual_denominator"] == pytest.approx(expected_denominator)
    assert row.native_criterion_value == pytest.approx(expected_value)
    assert row.metadata["normal_residual_denominator"] != pytest.approx(4.0 * float(residual.norm().item()))


def test_krylov_budget_boundary_is_not_reported_as_convergence() -> None:
    operator = DenseLinearOperator(torch.eye(2, dtype=torch.float64))
    measurement = torch.ones(1, 2, dtype=torch.float64)
    result = CGLSSolver(num_iterations=1, tol=0.0).solve_detailed(
        measurement,
        operator,
        control=_policy(target=10.0, max_iterations=1, normal=10.0, iterate=1e15),
        operator_norm_estimate=1.0,
    )

    assert result.status == "max_iterations"
    assert result.stopping_reason == "maximum_iterations_reached"


def test_native_policy_without_discrepancy_target_is_not_downgraded() -> None:
    operator = DenseLinearOperator(torch.eye(2, dtype=torch.float64))
    measurement = torch.ones(1, 2, dtype=torch.float64)
    control = SolveControl(
        max_iterations=3,
        min_iterations=1,
        check_every=1,
        patience=1,
        relative_iterate_tolerance=1e15,
        normalized_normal_residual_tolerance=10.0,
        metadata={"effective_stopping_policy": {}},
    )
    result = CGLSSolver(num_iterations=3, tol=0.0).solve_detailed(
        measurement,
        operator,
        control=control,
        operator_norm_estimate=1.0,
    )

    assert result.status == "converged"
    assert result.trajectory[-1].criteria == {"krylov_native": True}


def test_krylov_endpoint_recomputes_scale_from_squared_norm() -> None:
    operator = DenseLinearOperator(torch.diag(torch.tensor([2.0, 0.5], dtype=torch.float64)))
    measurement = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    result = CGLSSolver(num_iterations=3, tol=0.0).solve_detailed(
        measurement,
        operator,
        control=_policy(target=10.0, max_iterations=3, normal=10.0, iterate=1e15),
        operator_norm_squared=4.0,
    )
    residual = result.predicted_measurement - measurement
    report = confirm_endpoint(
        algorithm="cgls",
        reconstruction=result.reconstruction,
        measurement=measurement,
        operator=operator,
        predicted_measurement=result.predicted_measurement,
        policy=_endpoint_policy(target=10.0, normal=10.0, iterate=1e15),
        trajectory=[row.to_dict() for row in result.trajectory],
        solver_status=result.status,
        solver_stopping_reason=result.stopping_reason,
        iterations=result.actual_iterations,
        max_iterations=3,
        parameters={},
        operator_norm_squared=4.0,
    )

    assert report["passed"]
    assert report["operator_norm_estimate"] == pytest.approx(2.0)
    assert report["operator_norm_squared"] == pytest.approx(4.0)
    assert report["normal_residual_denominator"] == pytest.approx(2.0 * float(residual.norm().item()))
    assert report["trajectory_endpoint_native_consistent"]


def test_generalized_tikhonov_detailed_and_endpoint_use_supplied_L() -> None:
    dtype = torch.float64
    reconstruction_operator = DenseLinearOperator(torch.eye(2, dtype=dtype))
    regularization_operator = DenseLinearOperator(torch.diag(torch.tensor([2.0, 1.0], dtype=dtype)))
    measurement = torch.tensor([[1.0, 2.0]], dtype=dtype)
    strength = 0.5
    normal = torch.eye(2, dtype=dtype) + strength * torch.diag(torch.tensor([4.0, 1.0], dtype=dtype))
    expected = torch.linalg.solve(normal, measurement.T).T

    fixed_result = TikhonovSolver(
        reg_strength=strength,
        num_iterations=4,
        tolerance=0.0,
        regularization_operator=regularization_operator,
    ).solve_detailed(
        measurement,
        reconstruction_operator,
        control=SolveControl(max_iterations=4, tolerance=0.0),
    )
    assert torch.allclose(fixed_result.reconstruction, expected, atol=1e-12, rtol=1e-12)
    assert fixed_result.metadata["regularization_operator"] == "linear_operator"

    policy_result = TikhonovSolver(
        reg_strength=strength,
        num_iterations=3,
        tolerance=0.0,
        regularization_operator=regularization_operator,
    ).solve_detailed(
        measurement,
        reconstruction_operator,
        control=_policy(target=10.0, max_iterations=3, normal=10.0, iterate=1e15),
    )
    report = confirm_endpoint(
        algorithm="tikhonov",
        reconstruction=policy_result.reconstruction,
        measurement=measurement,
        operator=reconstruction_operator,
        predicted_measurement=policy_result.predicted_measurement,
        policy=_endpoint_policy(target=10.0, normal=10.0, iterate=1e15),
        trajectory=[row.to_dict() for row in policy_result.trajectory],
        solver_status=policy_result.status,
        solver_stopping_reason=policy_result.stopping_reason,
        iterations=policy_result.actual_iterations,
        max_iterations=3,
        parameters={"reg_strength": strength},
        regularization_operator=regularization_operator,
    )
    endpoint_x = policy_result.reconstruction
    endpoint_residual = endpoint_x - measurement
    regularization_gradient = regularization_operator.adjoint(
        regularization_operator.forward(endpoint_x)
    )
    expected_denominator = float(measurement.norm().item()) + strength * float(regularization_gradient.norm().item())
    expected_objective = float(
        0.5 * endpoint_residual.square().sum().item()
        + strength * 0.5 * regularization_operator.forward(endpoint_x).square().sum().item()
    )
    assert report["passed"]
    assert report["regularized_normal_residual_denominator"] == pytest.approx(expected_denominator)
    assert report["composite_objective"] == pytest.approx(expected_objective)
    assert report["trajectory_endpoint_native_consistent"]


def test_tv_policy_records_returned_state_and_endpoint_objective() -> None:
    base_operator = IdentityImageOperator()
    operator = CountingLinearOperator(base_operator)
    measurement = torch.zeros((1, 1, 4, 4), dtype=torch.float64)
    measurement[..., 1:3, 1:3] = 1.0
    regularizer = TVRegularizer(num_iterations=2, tolerance=0.0)
    result = TVFISTASolver(
        reg_strength=0.05,
        num_iterations=3,
        step_size=0.1,
        regularizer=regularizer,
        tolerance=0.0,
    ).solve_detailed(
        measurement,
        operator,
        control=_policy(
            target=0.0,
            max_iterations=3,
            patience=10,
            normal=0.0,
            iterate=0.0,
            objective=0.0,
            mapping=0.0,
        ),
    )

    assert result.status == "max_iterations"
    assert operator.stats()["forward_calls"] == 2 * 3 + 1
    assert operator.stats()["adjoint_calls"] == 2 * 3
    assert result.trajectory
    assert all(
        (row.metadata.get("objective_state"), row.metadata.get("mapping_state"))
        == ("returned", "returned")
        for row in result.trajectory
    )
    assert result.trajectory[-1].objective == pytest.approx(result.final_objective)

    report = confirm_endpoint(
        algorithm="tv_fista",
        reconstruction=result.reconstruction,
        measurement=measurement,
        operator=operator,
        predicted_measurement=result.predicted_measurement,
        policy=_endpoint_policy(target=0.0, normal=0.0, iterate=0.0, objective=0.0, mapping=0.0, patience=10),
        trajectory=[row.to_dict() for row in result.trajectory],
        solver_status=result.status,
        solver_stopping_reason=result.stopping_reason,
        iterations=result.actual_iterations,
        max_iterations=3,
        parameters={
            "reg_strength": 0.05,
            "step_size": 0.1,
            "tv_mode": "isotropic",
            "tv_num_iterations": 2,
            "tv_tolerance": 0.0,
        },
    )
    assert report["passed"]
    assert report["returned_state_trajectory"]
    assert report["objective_endpoint_consistent"]
    assert report["trajectory_endpoint_native_consistent"]


def test_tv_policy_stalls_on_native_plateau_when_discrepancy_is_unreachable() -> None:
    operator = ZeroImageOperator()
    measurement = torch.ones((1, 1, 4, 4), dtype=torch.float64)
    result = TVFISTASolver(
        reg_strength=0.05,
        num_iterations=6,
        step_size=0.1,
        regularizer=TVRegularizer(num_iterations=1, tolerance=0.0),
        tolerance=0.0,
    ).solve_detailed(
        measurement,
        operator,
        control=SolveControl(
            max_iterations=6,
            min_iterations=1,
            check_every=1,
            patience=10,
            discrepancy_target=0.1,
            prox_gradient_mapping_tolerance=0.0,
            relative_objective_tolerance=0.0,
            stall_enabled=True,
            stall_relative_iterate_tolerance=1e-8,
            stall_patience=2,
        ),
    )

    assert result.status == "stalled"
    assert result.stopping_reason == "stalled_before_discrepancy"
    assert result.actual_iterations == 2
    assert result.status != "diverged"
    assert result.trajectory[-1].metadata["native_plateau"]
    assert result.trajectory[-1].metadata["native_plateau_consecutive"] == 2
    assert result.metadata["native_plateau_stall"]["consecutive"] == 2


@pytest.mark.parametrize("solver", [CGLSSolver(num_iterations=2), LSQRSolver(num_iterations=2)])
def test_krylov_breakdown_and_input_validation_are_structured(solver) -> None:
    zero_operator = ZeroOperator()
    measurement = torch.ones(1, 2, dtype=torch.float64)
    stalled = solver.solve_detailed(
        measurement,
        zero_operator,
        control=_policy(target=0.0, max_iterations=2),
    )
    assert stalled.status == "stalled"
    assert stalled.stopping_reason == "stalled_before_discrepancy"

    invalid = solver.solve_detailed(measurement, zero_operator, num_iterations=0)
    assert invalid.status == "invalid_parameters"
    assert invalid.stopping_reason == "parameter_validation_failed"

    nonfinite = solver.solve_detailed(
        torch.full_like(measurement, float("nan")),
        zero_operator,
    )
    assert nonfinite.status == "numerical_error"

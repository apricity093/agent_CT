from __future__ import annotations

from types import SimpleNamespace

import torch

from inv_framework.convergence import confirm_endpoint, normalize_status
from inv_framework.ct_runtime import ConfigError, _discrepancy_target
from inv_framework.operators.base import LinearOperator
from inv_framework.solvers import SolveControl
from inv_framework.solvers.detailed import solve_fbp_detailed, solve_fdk_detailed
from inv_framework.solvers.statistical import MLEMSolver, OSEMSolver


class AngleIdentityOperator(LinearOperator):
    domain_shape = (1, 4)
    range_shape = (1, 4, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value.unsqueeze(-1)

    def adjoint(self, value: torch.Tensor) -> torch.Tensor:
        return value.squeeze(-1)

    def subset(self, indices: torch.Tensor) -> LinearOperator:
        return _AngleIdentitySubset(indices)


class _AngleIdentitySubset(LinearOperator):
    domain_shape = (1, 4)

    def __init__(self, indices: torch.Tensor):
        self.indices = indices
        self.range_shape = (1, int(indices.numel()), 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value.index_select(2, self.indices).unsqueeze(-1)

    def adjoint(self, value: torch.Tensor) -> torch.Tensor:
        result = torch.zeros(
            (value.shape[0], *self.domain_shape),
            dtype=value.dtype,
            device=value.device,
        )
        return result.index_add(2, self.indices, value.squeeze(-1))


def _count_control(**kwargs) -> SolveControl:
    return SolveControl(
        max_iterations=kwargs.pop("max_iterations", 10),
        min_iterations=kwargs.pop("min_iterations", 2),
        patience=kwargs.pop("patience", 2),
        relative_iterate_tolerance=kwargs.pop("relative_iterate_tolerance", 1e-8),
        relative_objective_tolerance=kwargs.pop("relative_objective_tolerance", 1e-8),
        **kwargs,
    )


def test_mlem_uses_complete_epoch_poisson_patience_and_rejects_bad_counts():
    operator = AngleIdentityOperator()
    measurement = torch.ones(1, *operator.range_shape)
    x_init = torch.ones(1, *operator.domain_shape)

    result = MLEMSolver(num_iterations=10).solve_detailed(
        measurement,
        operator,
        x_init=x_init,
        control=_count_control(),
    )

    assert result.status == "converged", result.stopping_reason
    assert result.actual_iterations == 3
    assert result.stopping_reason == "poisson_deviance_and_relative_iterate_change_patience"
    assert result.trajectory
    assert all(row.epoch == row.iteration for row in result.trajectory)
    assert all(row.metadata.get("complete_epoch") for row in result.trajectory)
    assert result.trajectory[-1].criteria == {
        "normalized_poisson_deviance_change": True,
        "relative_iterate_change": True,
    }
    assert result.metadata["poisson_deviance_normalization"] == "2*sum_observed"

    budget_limited = MLEMSolver(num_iterations=2).solve_detailed(
        measurement,
        operator,
        x_init=x_init,
        control=_count_control(max_iterations=2, min_iterations=1, patience=10),
    )
    assert budget_limited.status == "max_iterations"
    assert budget_limited.actual_iterations == 2
    assert budget_limited.stopping_reason == "maximum_epochs_reached"

    endpoint = confirm_endpoint(
        algorithm="mlem",
        reconstruction=result.reconstruction,
        measurement=measurement,
        predicted_measurement=result.predicted_measurement,
        operator=operator,
        policy={
            "effective": {"discrepancy_target": None},
            "min_iterations": 2,
            "check_every": 1,
            "patience": 2,
            "relative_iterate_tolerance": 1e-8,
            "relative_objective_tolerance": 1e-8,
            "endpoint_confirmation": {"require_trajectory_consistency": True},
        },
        trajectory=[row.to_dict() for row in result.trajectory],
        solver_status=result.status,
        solver_stopping_reason=result.stopping_reason,
        iterations=result.actual_iterations,
        max_iterations=10,
        parameters={"eps": 1e-8},
    )
    assert endpoint["passed"], endpoint

    negative = measurement.clone()
    negative[..., 0, 0] = -1.0
    invalid = MLEMSolver(num_iterations=2).solve_detailed(negative, operator)
    assert invalid.status == "invalid_parameters"

    nonfinite = measurement.clone()
    nonfinite[..., 0, 0] = float("nan")
    failed = MLEMSolver(num_iterations=2).solve_detailed(nonfinite, operator)
    assert failed.status == "numerical_error"


def test_osem_records_cycle_free_complete_sweep_evidence():
    operator = AngleIdentityOperator()
    measurement = torch.ones(1, *operator.range_shape)
    result = OSEMSolver(num_iterations=10, block_size=2).solve_detailed(
        measurement,
        operator,
        x_init=torch.ones(1, *operator.domain_shape),
        control=_count_control(),
    )
    assert result.status == "converged", result.stopping_reason
    assert result.actual_iterations == 3
    assert result.stopping_reason == "poisson_deviance_and_relative_epoch_change_patience"
    assert all(row.metadata.get("complete_epoch") for row in result.trajectory)
    assert all(row.metadata.get("cycle_free") for row in result.trajectory)
    assert all(row.subset is None for row in result.trajectory)


class _CyclingSubset(LinearOperator):
    domain_shape = (1, 1)
    range_shape = (1, 1, 1)

    def __init__(self, subset_index: int):
        self.subset_index = subset_index
        self.last_x = None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.last_x = value.detach().clone()
        return torch.ones(
            (value.shape[0], *self.range_shape), dtype=value.dtype, device=value.device
        )

    def adjoint(self, value: torch.Tensor) -> torch.Tensor:
        if self.subset_index != 0 or self.last_x is None:
            return torch.ones(
                (value.shape[0], *self.domain_shape), dtype=value.dtype, device=value.device
            )
        target = torch.where(
            self.last_x < 1.5,
            torch.full_like(self.last_x, 2.0),
            torch.full_like(self.last_x, 0.5),
        )
        return target / self.last_x.clamp_min(1e-6)


class _CyclingEmissionOperator(LinearOperator):
    domain_shape = (1, 1)
    range_shape = (1, 2, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.ones(
            (value.shape[0], *self.range_shape), dtype=value.dtype, device=value.device
        )

    def adjoint(self, value: torch.Tensor) -> torch.Tensor:
        return torch.ones(
            (value.shape[0], *self.domain_shape), dtype=value.dtype, device=value.device
        )

    def subset(self, indices: torch.Tensor) -> LinearOperator:
        return _CyclingSubset(int(indices[0].item()))


def test_osem_reports_a_repeating_subset_cycle_as_stalled():
    operator = _CyclingEmissionOperator()
    result = OSEMSolver(num_iterations=10, block_size=1).solve_detailed(
        torch.ones(1, *operator.range_shape),
        operator,
        x_init=torch.ones(1, *operator.domain_shape),
        control=_count_control(
            max_iterations=10,
            min_iterations=1,
            patience=10,
            stall_enabled=True,
            metadata={"osem_cycle_patience": 2, "osem_cycle_tolerance": 1e-8},
        ),
    )
    assert result.status == "stalled"
    assert result.stopping_reason == "osem_subset_cycle_detected"
    assert result.metadata["cycle_detected"] is True
    assert result.metadata["cycle_run"] >= 2
    assert any(row.metadata.get("cycle_detected") for row in result.trajectory)


def test_count_domain_policy_does_not_invent_transmission_discrepancy_target():
    count_case = SimpleNamespace(
        metadata={
            "measurement": {
                "kind": "counts",
                "observation_model": "poisson_emission",
                "parameters": {},
            }
        },
        measurement=torch.ones(1, 4, 1),
        valid_measurement_mask=None,
    )
    assert _discrepancy_target(
        count_case, {"discrepancy": {"enabled": True}}
    ) is None
    assert _discrepancy_target(
        count_case,
        {"discrepancy": {"normalized_poisson_deviance_target": 0.25}},
    ) == 0.25

    transmission_case = SimpleNamespace(
        metadata={"measurement": {"kind": "log_projection", "parameters": {}}},
        measurement=torch.ones(1, 4, 1),
        valid_measurement_mask=None,
    )
    try:
        _discrepancy_target(
            transmission_case, {"discrepancy": {"enabled": True}}
        )
    except ConfigError as error:
        assert "incident_photon_count" in str(error)
    else:
        raise AssertionError("transmission discrepancy target must require incident photons")


class _StuckEmissionOperator(AngleIdentityOperator):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.ones(
            (value.shape[0], *self.range_shape),
            dtype=value.dtype,
            device=value.device,
        )

    def adjoint(self, value: torch.Tensor) -> torch.Tensor:
        return torch.ones(
            (value.shape[0], *self.domain_shape),
            dtype=value.dtype,
            device=value.device,
        )


def test_mlem_distinguishes_plateau_and_poisson_deviance_growth():
    operator = _StuckEmissionOperator()
    measurement = torch.full((1, *operator.range_shape), 2.0)
    stalled = MLEMSolver(num_iterations=10).solve_detailed(
        measurement,
        operator,
        x_init=torch.ones(1, *operator.domain_shape),
        control=_count_control(
            max_iterations=10,
            min_iterations=1,
            patience=2,
            discrepancy_target=0.0,
            stall_enabled=True,
            stall_patience=2,
        ),
    )
    assert stalled.status == "stalled"
    assert stalled.stopping_reason == "poisson_deviance_plateau_before_discrepancy"

    class GrowingOperator(AngleIdentityOperator):
        def __init__(self):
            self.calls = 0

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return torch.full(
                (value.shape[0], *self.range_shape),
                float(self.calls),
                dtype=value.dtype,
                device=value.device,
            )

        def adjoint(self, value: torch.Tensor) -> torch.Tensor:
            return torch.ones(
                (value.shape[0], *self.domain_shape),
                dtype=value.dtype,
                device=value.device,
            )

    diverged = MLEMSolver(num_iterations=10).solve_detailed(
        torch.ones(1, *operator.range_shape),
        GrowingOperator(),
        x_init=torch.ones(1, *operator.domain_shape),
        control=_count_control(
            min_iterations=1,
            patience=10,
            divergence_enabled=True,
            divergence_patience=2,
        ),
    )
    assert diverged.status == "diverged"
    assert diverged.stopping_reason == "persistent_poisson_deviance_increase"


class _DirectOperator(LinearOperator):
    domain_shape = (1, 2)
    range_shape = (1, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def adjoint(self, value: torch.Tensor) -> torch.Tensor:
        return value


class _FDKOperator(_DirectOperator):
    def __init__(self, mode="valid"):
        self.mode = mode

    def fdk(self, value: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.mode == "nan":
            return torch.full(
                (value.shape[0], *self.domain_shape), float("nan"), dtype=value.dtype
            )
        if self.mode == "wrong_shape":
            return torch.zeros((value.shape[0], 1), dtype=value.dtype)
        if self.mode == "failure":
            raise RuntimeError("synthetic FDK backend failure")
        return torch.ones((value.shape[0], *self.domain_shape), dtype=value.dtype)


def test_direct_solvers_have_validity_not_convergence_states():
    operator = _DirectOperator()
    measurement = torch.ones(1, *operator.range_shape)
    valid = solve_fbp_detailed(operator, measurement)
    assert valid.status == "completed_valid"
    assert valid.actual_iterations == 0
    assert valid.stopping_reason == "direct_reconstruction_valid"
    assert normalize_status("converged", algorithm="fbp", direct=True).value == "completed_valid"

    invalid = solve_fbp_detailed(operator, measurement, scale=0.0)
    assert invalid.status == "invalid_parameters"
    nonfinite = solve_fbp_detailed(operator, torch.full_like(measurement, float("nan")))
    assert nonfinite.status == "numerical_error"

    fdk_valid = solve_fdk_detailed(_FDKOperator(), measurement)
    assert fdk_valid.status == "completed_valid"
    fdk_invalid_filter = solve_fdk_detailed(
        _FDKOperator(), measurement, filter_type="hann"
    )
    assert fdk_invalid_filter.status == "invalid_parameters"
    fdk_invalid = solve_fdk_detailed(_FDKOperator(), measurement, voxel_supersampling=0)
    assert fdk_invalid.status == "invalid_parameters"
    fdk_nonfinite = solve_fdk_detailed(_FDKOperator("nan"), measurement)
    assert fdk_nonfinite.status == "numerical_error"
    fdk_wrong_shape = solve_fdk_detailed(_FDKOperator("wrong_shape"), measurement)
    assert fdk_wrong_shape.status == "numerical_error"
    fdk_unavailable = solve_fdk_detailed(_DirectOperator(), measurement)
    assert fdk_unavailable.status == "unavailable"

    endpoint = confirm_endpoint(
        algorithm="fbp",
        reconstruction=valid.reconstruction,
        measurement=measurement,
        operator=operator,
        policy=None,
        trajectory=(),
        solver_status=valid.status,
        solver_stopping_reason=valid.stopping_reason,
        iterations=0,
        max_iterations=1,
        parameters={"scale": None},
    )
    assert endpoint["passed"]
    assert endpoint["shape_valid"]
    assert endpoint["parameter_valid"]

    invalid_endpoint = confirm_endpoint(
        algorithm="fdk",
        reconstruction=fdk_valid.reconstruction,
        measurement=measurement,
        operator=_FDKOperator(),
        policy=None,
        trajectory=(),
        solver_status=fdk_valid.status,
        solver_stopping_reason=fdk_valid.stopping_reason,
        iterations=0,
        max_iterations=1,
        parameters={"filter_type": "hann"},
    )
    assert not invalid_endpoint["passed"]
    assert not invalid_endpoint["parameter_valid"]

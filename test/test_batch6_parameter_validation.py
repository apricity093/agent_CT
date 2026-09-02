from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from inv_framework.ct_runtime import (
    estimate_lipschitz_squared,
    load_algorithm_config,
    run_case,
    validate_parameters,
)
from inv_framework.instrumentation import CountingLinearOperator
from inv_framework.operators.base import LinearOperator
from inv_framework.solvers.specs import (
    CANONICAL_ALGORITHM_IDS,
    PARAMETER_SOURCE_VOCABULARY,
    SOLVER_SPECS,
    compatibility_diagnostics,
    validate_parameter_values,
)


class IdentityOperator(LinearOperator):
    domain_shape = (1, 4, 4)
    range_shape = (1, 4, 4)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def adjoint(self, value: torch.Tensor) -> torch.Tensor:
        return value


def _emission_context() -> dict[str, object]:
    return {
        "observation_domain": "nonnegative_counts",
        "observation_model": "poisson_emission",
        "observation_min": 0.0,
        "observation_finite": True,
        "views": 8,
    }


@pytest.mark.parametrize("solver", CANONICAL_ALGORITHM_IDS)
def test_all_checked_in_algorithm_configs_validate(solver: str) -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "algorithms" / f"{solver}.yaml"
    name, parameters, source = load_algorithm_config(path, solver)
    assert name == solver
    assert source == path.resolve()
    assert set(parameters) <= set(SOLVER_SPECS[solver].parameter_names)
    for parameter in SOLVER_SPECS[solver].parameters:
        record = parameter.to_dict()
        assert record["canonical_name"] == parameter.name
        assert set(PARAMETER_SOURCE_VOCABULARY) <= set(record["provenance_vocabulary"])


@pytest.mark.parametrize(
    ("solver", "valid_parameters", "invalid_parameters", "context", "reason"),
    [
        ("fbp", {"scale": 0.5}, {"scale": 0.0}, {}, "parameter_out_of_range"),
        ("sirt", {"num_iterations": 1}, {"num_iterations": 0}, {}, "parameter_out_of_range"),
        (
            "landweber",
            {"num_iterations": 1, "step_size": 0.499999},
            {"num_iterations": 1, "step_size": 0.5},
            {"estimated_lipschitz": 4.0},
            "spectral_step_invalid",
        ),
        ("cgls", {"num_iterations": 1, "tol": 0.0}, {"num_iterations": 0}, {}, "parameter_out_of_range"),
        ("lsqr", {"num_iterations": 1, "damping": 0.0, "atol": 0.0, "btol": 0.0}, {"damping": -1.0}, {}, "parameter_out_of_range"),
        ("sart", {"block_size": 8, "relaxation": 1.0}, {"block_size": 9}, {"views": 8}, "block_size_exceeds_views"),
        (
            "os_sart",
            {"block_size": 4, "subset_count": 2, "relaxation": 1.0},
            {"subset_count": 9},
            {"views": 8},
            "subset_count_exceeds_views",
        ),
        (
            "mlem",
            {"initial_value": 0.1, "eps": 0.1},
            {"initial_value": 0.0},
            _emission_context(),
            "parameter_out_of_range",
        ),
        (
            "osem",
            {"subset_count": 8, "initial_value": 0.1, "eps": 0.1},
            {"subset_count": 9},
            _emission_context(),
            "subset_count_exceeds_views",
        ),
        (
            "tikhonov",
            {"reg_strength": 0.0, "tolerance": 0.0},
            {"min_value": 2.0, "max_value": 1.0},
            {},
            "parameter_constraint_violation",
        ),
        (
            "tv_fista",
            {"reg_strength": 0.0, "step_size": 0.25, "power_iterations": 1, "tv_num_iterations": 1, "tv_tolerance": 0.0},
            {"step_size": 0.250001},
            {"estimated_lipschitz": 4.0},
            "spectral_step_invalid",
        ),
        (
            "fdk",
            {"filter_type": "ramp", "voxel_supersampling": 1},
            {"filter_type": "hann"},
            {"geometry_type": "cone_3d", "dimension": 3, "observation_domain": "line_integral", "domain_shape": (1, 1, 1)},
            "parameter_choice_invalid",
        ),
    ],
)
def test_each_algorithm_has_valid_boundary_and_invalid_parameter_paths(
    solver: str,
    valid_parameters: dict[str, object],
    invalid_parameters: dict[str, object],
    context: dict[str, object],
    reason: str,
) -> None:
    valid_context = dict(context)
    estimated_lipschitz = valid_context.pop("estimated_lipschitz", None)
    valid = validate_parameter_values(
        solver,
        valid_parameters,
        estimated_lipschitz=estimated_lipschitz,
        **valid_context,
    )
    assert valid.valid, (solver, valid.errors)

    invalid_context = dict(context)
    invalid_estimate = invalid_context.pop("estimated_lipschitz", None)
    invalid = validate_parameter_values(
        solver,
        invalid_parameters,
        estimated_lipschitz=invalid_estimate,
        **invalid_context,
    )
    assert not invalid.valid, (solver, invalid.parameters)
    assert reason in invalid.reason_codes, (solver, invalid.errors, invalid.reason_codes)


def test_theory_boundaries_and_request_only_warnings() -> None:
    alias = validate_parameter_values("landweber", {"learning_rate": 0.1}, estimated_lipschitz=4.0)
    assert alias.valid and alias.parameters["step_size"] == pytest.approx(0.1)

    landweber = validate_parameter_values(
        "landweber", {"step_size": 0.1}, estimated_lipschitz=4.0
    )
    assert landweber.valid
    assert landweber.estimates["step_size_source"] == "user_override"

    strict_landweber = validate_parameter_values(
        "landweber", {"step_size": 0.5}, estimated_lipschitz=4.0
    )
    assert not strict_landweber.valid

    fista_boundary = validate_parameter_values(
        "tv_fista", {"step_size": 0.25}, estimated_lipschitz=4.0
    )
    assert fista_boundary.valid

    request_only = validate_parameter_values("tv_fista", {})
    assert request_only.valid
    assert "spectral_bound_unavailable" in request_only.warning_codes
    assert "operator_norm_squared" not in request_only.estimates


@pytest.mark.parametrize("dimension", [2.5, True, 0, -1, "2"])
def test_dimension_context_requires_strict_positive_integer(dimension: object) -> None:
    issues = compatibility_diagnostics(
        "sirt",
        dimension=dimension,
        geometry_type="parallel_2d",
        observation_domain="line_integral",
    )
    assert "dimension_invalid" in {issue["code"] for issue in issues}

    result = validate_parameter_values(
        "sirt",
        {"num_iterations": 1},
        dimension=dimension,
        geometry_type="parallel_2d",
        observation_domain="line_integral",
    )
    assert not result.valid
    assert "dimension_invalid" in result.reason_codes


def test_request_only_dimension_none_remains_compatible() -> None:
    assert validate_parameter_values("sirt", {}, dimension=None).valid


@pytest.mark.parametrize(
    "shape",
    [(0, 0, 0), (-1, 1, 1), (True, 1, 1), (1.0, 1, 1), (1, 2)],
)
def test_fdk_requires_strict_positive_cubic_integer_shape(shape: object) -> None:
    result = validate_parameter_values(
        "fdk",
        {},
        geometry_type="cone_3d",
        dimension=3,
        observation_domain="line_integral",
        domain_shape=shape,
    )
    assert not result.valid
    assert "geometry_shape_invalid" in result.reason_codes


def test_fdk_rejects_non_cubic_or_invalid_image_shape() -> None:
    non_cubic = validate_parameter_values(
        "fdk",
        {},
        geometry_type="cone_3d",
        dimension=3,
        observation_domain="line_integral",
        domain_shape=(1, 2, 3),
    )
    assert not non_cubic.valid
    assert "cubic_volume_required" in non_cubic.reason_codes

    invalid_image_shape = validate_parameter_values(
        "fdk",
        {},
        geometry_type="cone_3d",
        dimension=3,
        observation_domain="line_integral",
        image_shape=(2.0, 2, 2),
    )
    assert not invalid_image_shape.valid
    assert "geometry_shape_invalid" in invalid_image_shape.reason_codes


def test_deterministic_power_iteration_has_explicit_count_contract() -> None:
    first_operator = CountingLinearOperator(IdentityOperator())
    second_operator = CountingLinearOperator(IdentityOperator())
    reference = torch.ones(1, 4, 4)
    first = estimate_lipschitz_squared(first_operator, reference, num_iterations=3)
    second = estimate_lipschitz_squared(second_operator, reference, num_iterations=3)
    assert first == second == pytest.approx(1.0)
    assert first_operator.stats()["forward_calls"] == 4
    assert first_operator.stats()["adjoint_calls"] == 4

    case = SimpleNamespace(
        measurement=reference,
        geometry={"type": "parallel_2d", "domain_shape": (1, 4, 4)},
        metadata={"dimension": 2},
    )
    counted = CountingLinearOperator(IdentityOperator())
    result = validate_parameters(
        "landweber",
        {"num_iterations": 1},
        case=case,
        operator=counted,
    )
    assert result.valid
    assert result.parameters["step_size"] == pytest.approx(0.9)
    contract = result.estimates["spectral_estimator_contract"]
    assert contract["iterations"] == 12
    assert contract["forward_calls"] == contract["adjoint_calls"] == 13
    assert counted.stats()["forward_calls"] == 13
    assert counted.stats()["adjoint_calls"] == 13


def test_invalid_parameters_are_rejected_before_any_spectral_operator_call() -> None:
    case = SimpleNamespace(
        measurement=torch.ones(1, 4, 4),
        geometry={"type": "parallel_2d", "domain_shape": (1, 4, 4)},
        metadata={"dimension": 2},
    )
    counted = CountingLinearOperator(IdentityOperator())
    result = validate_parameters(
        "tv_fista",
        {"power_iterations": 0, "step_size": 0.1},
        case=case,
        operator=counted,
    )
    assert not result.valid
    assert "parameter_out_of_range" in result.reason_codes
    assert counted.stats()["total_operator_calls"] == 0


def test_observation_and_non_applicable_parameter_gates_are_explicit() -> None:
    negative_counts = validate_parameter_values(
        "mlem",
        {"initial_value": 0.1, "eps": 0.1},
        observation_domain="nonnegative_counts",
        observation_model="poisson_emission",
        observation_min=-1.0,
    )
    assert not negative_counts.valid
    assert "observation_not_nonnegative" in negative_counts.reason_codes

    transmission = validate_parameter_values(
        "osem",
        {},
        observation_domain="line_integral",
        observation_model="xray_transmission",
    )
    assert not transmission.valid
    assert "emission_observation_model_required" in transmission.reason_codes

    for parameter, reason in (
        ("admm_rho", "parameter_not_applicable"),
        ("primal_step", "parameter_not_applicable"),
        ("dual_step", "parameter_not_applicable"),
        ("free_momentum", "free_momentum_not_applicable"),
        ("momentum", "free_momentum_not_applicable"),
    ):
        result = validate_parameter_values("tv_fista", {parameter: 0.5})
        assert not result.valid
        assert reason in result.reason_codes


def test_budget_limits_are_validated_before_execution() -> None:
    iterations = validate_parameter_values("sirt", {"num_iterations": 2}, max_iterations=1)
    assert not iterations.valid
    assert "iteration_budget_exceeded" in iterations.reason_codes

    calls = validate_parameter_values(
        "cgls",
        {"num_iterations": 1},
        max_forward_calls=1,
        expected_operator_calls={"forward": 2, "adjoint": 1},
    )
    assert not calls.valid
    assert "forward_call_budget_exceeded" in calls.reason_codes

    invalid_budget = validate_parameter_values("sirt", {}, max_adjoint_calls=-1)
    assert not invalid_budget.valid
    assert "budget_invalid" in invalid_budget.reason_codes


def test_run_case_persists_invalid_parameters_without_solver_iterations(tmp_path: Path) -> None:
    config = tmp_path / "invalid_sirt.yaml"
    config.write_text(
        "schema_version: 1\nname: sirt\nparameters:\n  num_iterations: 0\n",
        encoding="utf-8",
    )
    result = run_case(
        "sirt",
        "parallel_2d/disk_analytic_32",
        config,
        tmp_path / "run",
        data_root=Path(__file__).resolve().parent / "data",
    )
    assert result["status"] == "invalid_parameters"
    diagnostics = result["diagnostics"]
    assert diagnostics["iterations_completed"] == 0
    assert diagnostics["parameter_validation"]["valid"] is False
    assert "parameter_out_of_range" in diagnostics["parameter_validation"]["reason_codes"]
    assert diagnostics["resources"].get("total_operator_calls", 0) == 0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from inv_framework.convergence import ConvergenceStatus, classify_trajectory
from inv_framework.solvers import ConsecutiveStoppingMonitor, SolveControl
from inv_framework.ct_runtime import (
    SOLVER_BUILDERS,
    algorithm_config_ids,
    build_solver,
    run_case,
    validate_runtime_registry,
)
from pathlib import Path
from inv_framework.instrumentation import CountingLinearOperator, OperatorBudgetExceeded
from inv_framework.operators.base import LinearOperator
from inv_framework.solvers.specs import (
    ALGORITHM_ALIASES,
    CANONICAL_ALGORITHM_IDS,
    COMPATIBILITY_REASON_CODES,
    NON_APPLICABLE_PARAMETER_CATEGORIES,
    SOLVER_SPECS,
    compatibility_diagnostics,
    registry_contract,
    registry_digest,
    validate_parameter_values,
)


class IdentityOperator(LinearOperator):
    domain_shape = (1, 4, 4)
    range_shape = (1, 4, 4)

    def forward(self, x):
        return x

    def adjoint(self, y):
        return y


def test_registry_contains_complete_metadata_and_statistical_domain_gate():
    assert tuple(SOLVER_SPECS) == CANONICAL_ALGORITHM_IDS
    for spec in SOLVER_SPECS.values():
        assert spec.parameter_names == tuple(parameter.name for parameter in spec.parameters)
        assert spec.observation_domains
        assert spec.failure_modes
    result = validate_parameter_values("mlem", {}, observation_domain="log_projection")
    assert not result.valid
    assert any("observation domain" in error for error in result.errors)
    missing_model = validate_parameter_values("mlem", {}, observation_domain="nonnegative_counts")
    assert not missing_model.valid
    assert "emission_observation_model_required" in missing_model.reason_codes


def test_registry_build_config_bijection_is_json_stable():
    report = validate_runtime_registry()
    assert report["canonical_algorithm_ids"] == list(CANONICAL_ALGORITHM_IDS)
    assert set(report["registry_ids"]) == set(CANONICAL_ALGORITHM_IDS)
    assert set(report["build_ids"]) == set(CANONICAL_ALGORITHM_IDS)
    assert set(report["config_ids"]) == set(CANONICAL_ALGORITHM_IDS)
    assert set(SOLVER_BUILDERS) == set(CANONICAL_ALGORITHM_IDS)
    assert algorithm_config_ids() == CANONICAL_ALGORITHM_IDS
    assert report["all_build_reachable"] is True
    assert report["all_configured"] is True
    assert report["aliases"] == ALGORITHM_ALIASES == {}
    assert registry_contract()["schema_version"] == "ct.algorithm_registry.v1"
    assert report["registry_digest"] == registry_digest()


def test_every_canonical_builder_constructs_from_registry_defaults():
    case = SimpleNamespace(
        measurement=torch.zeros(1, 16, 4),
        geometry={"type": "parallel_2d"},
    )
    constructed = {
        name: type(build_solver(name, validate_parameter_values(name, {}).parameters, case)).__name__
        for name in CANONICAL_ALGORITHM_IDS
    }
    assert set(constructed) == set(CANONICAL_ALGORITHM_IDS)


def test_registry_exposes_fixed_pairings_and_non_applicable_categories():
    assert SOLVER_SPECS["tikhonov"].regularizer_pairing == "tikhonov"
    assert SOLVER_SPECS["tv_fista"].regularizer_pairing == "tv"
    assert SOLVER_SPECS["tikhonov"].regularizer_pairing_policy == "fixed"
    assert SOLVER_SPECS["tv_fista"].regularizer_pairing_policy == "fixed"
    for spec in SOLVER_SPECS.values():
        for category in NON_APPLICABLE_PARAMETER_CATEGORIES:
            assert spec.parameter_applicability[category] == "not_applicable"
    assert {
        "parameter_not_applicable",
        "free_momentum_not_applicable",
    } <= set(COMPATIBILITY_REASON_CODES)


def test_structured_compatibility_diagnostics_explain_emission_boundary():
    issues = compatibility_diagnostics(
        "mlem",
        observation_domain="line_integral",
        observation_model="xray_transmission",
    )
    codes = {issue["code"] for issue in issues}
    assert "observation_domain_unsupported" in codes
    assert "observation_model_unsupported" in codes
    assert "emission_observation_model_required" in codes
    assert all({"code", "severity", "message", "details"} <= set(issue) for issue in issues)


def test_fdk_metadata_advertises_backend_and_volume_requirements():
    spec = SOLVER_SPECS["fdk"]
    assert {"ASTRA CUDA", "CUDA", "ASTRA", "cone_3d", "cubic_voxels"} <= set(spec.requirements)
    assert "backend_capabilities" in spec.required_metadata


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


def test_consecutive_stopping_policy_requires_min_patience_and_resets():
    control = SolveControl(
        max_iterations=30, min_iterations=5, check_every=1, patience=5,
        discrepancy_target=0.1, relative_iterate_tolerance=1e-4,
    )
    monitor = ConsecutiveStoppingMonitor(control)
    ok = {"discrepancy": True, "native": True}
    bad = {"discrepancy": False, "native": True}
    for iteration in range(1, 9):
        decision = monitor.observe(iteration, criteria=ok, relative_change=1e-5, monitor_value=0.01)
        assert not decision.converged
    reset = monitor.observe(9, criteria=bad, relative_change=1e-5, monitor_value=0.2)
    assert reset.consecutive == 0
    for iteration in range(10, 14):
        assert not monitor.observe(iteration, criteria=ok, relative_change=1e-5, monitor_value=0.01).converged
    assert monitor.observe(14, criteria=ok, relative_change=1e-5, monitor_value=0.01).converged


def test_consecutive_stopping_policy_honors_check_every_and_classifies_trends():
    control = SolveControl(
        max_iterations=20, min_iterations=2, check_every=2, patience=2,
        discrepancy_target=0.1, stall_enabled=True, stall_patience=2,
        divergence_enabled=True, divergence_patience=2,
    )
    monitor = ConsecutiveStoppingMonitor(control)
    unchecked = monitor.observe(3, criteria={"discrepancy": True, "native": True}, relative_change=0.0, monitor_value=1.0)
    assert not unchecked.checked and unchecked.consecutive == 0
    monitor.observe(2, criteria={"discrepancy": False, "native": False}, relative_change=1e-10, monitor_value=1.0)
    stalled = monitor.observe(4, criteria={"discrepancy": False, "native": False}, relative_change=1e-10, monitor_value=1.1)
    assert stalled.stalled


def test_runtime_applies_policy_and_independent_endpoint(tmp_path):
    root = Path(__file__).resolve().parents[1]
    policy = {
        "schema_version": "1.0", "mode": "solver_native_and_endpoint",
        "stop_on_convergence": True, "min_iterations": 5, "check_every": 1, "patience": 5,
        "relative_iterate_tolerance": 1e-4,
        "normalized_normal_residual_tolerance": 1e-4,
        "relative_objective_tolerance": 1e-5,
        "prox_gradient_mapping_tolerance": 1e-4,
        "discrepancy": {"enabled": True, "tau": 1.05, "count_floor": 1.0, "epsilon": 1e-12, "use_valid_measurement_mask": True},
        "stalled": {"enabled": True, "relative_iterate_tolerance": 1e-8, "patience": 5, "require_discrepancy_unmet": True},
        "divergence": {"enabled": True, "relative_increase_tolerance": 1e-4, "patience": 5},
        "endpoint_confirmation": {"enabled": True, "absolute_tolerance": 1e-7, "relative_tolerance": 1e-4, "require_finite": True, "require_trajectory_consistency": True},
    }
    result = run_case(
        "sirt", "parallel_2d/shepp_logan_sparse_poisson_32",
        root / "configs" / "algorithms" / "sirt.yaml", tmp_path / "run",
        data_root=root / "test" / "data", max_iterations=10,
        parameter_overrides={"num_iterations": 10}, stopping_policy=policy,
    )
    assert result["status"] == "success"
    diagnostics = result["diagnostics"]
    assert diagnostics["execution_status"] == "completed"
    assert diagnostics["convergence"]["status"] != "converged" or diagnostics["endpoint_confirmation"]["passed"]
    assert diagnostics["resources"]["phases"]["endpoint_confirmation"]["forward_calls"] >= 1


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

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inv_framework.benchmarks import (
    BENCHMARK_PROTOCOLS,
    BenchmarkBudget,
    BenchmarkResult,
    ComparisonProtocol,
    FairComparisonContext,
    check_fairness,
    enforce_budget,
    environment_snapshot,
    pareto_front,
    per_axis_rankings,
)
from inv_framework.cli import main
from inv_framework.ct_runtime import load_comparison_protocol


ROOT = Path(__file__).parents[1]


def _context(**changes):
    environment = environment_snapshot(dependencies={"torch": "test"}, hardware={"cpu": "synthetic"})
    values = {
        "input_data_id": "case/a", "projection_data_sha256": "a" * 64,
        "geometry": {"type": "parallel_2d", "views": 8},
        "preprocessing": {"log": True}, "normalization": {"scale": 1.0},
        "reconstruction_resolution": [16, 16], "mask_id": "circle/v1",
        "initialization": {"rule": "zeros"}, "seed": 17, "device": "cpu",
        "dtype": "float32", "precision": "fp32",
        "warmup_policy": {"runs": 1, "excluded_from_timing": True},
        "timing_policy": {"clock": "perf_counter", "includes_evaluation": False},
        "environment": environment, "tuning_budget": {"trials": 4},
        "observation_stratum": "transmission", "observation_model": "line_integral",
        "validation": {"split_sha256": "split", "folds": 3},
    }
    values.update(changes)
    return FairComparisonContext(**values)


def _record(algorithm="sirt", **changes):
    context = changes.pop("context", _context())
    values = {
        "algorithm": algorithm, "solver": algorithm, "case_id": "case/a",
        "observation_domain": "line_integral", "observation_stratum": "transmission",
        "tuning_protocol": "equal_operator_calls/v1",
        "budget": {"protocol": "equal_operator_calls/v1", "max_forward_calls": 10, "max_adjoint_calls": 10},
        "forward_calls": 5, "adjoint_calls": 4, "runtime_seconds": 1.0,
        "psnr": 30.0, "ssim": .9, "rmse": .03, "residual": .1,
        "shared_context": context.to_dict(), "context_digest": context.digest,
        "protocol_digest": "protocol", "environment_digest": context.environment["digest"],
        "tuning_usage": {"completed_trials": 0},
    }
    values.update(changes)
    return BenchmarkResult(**values)


def test_all_six_versioned_protocol_configs_parse_and_cli_smokes(capsys):
    paths = sorted((ROOT / "configs" / "fair_protocols").glob("*.yaml"))
    assert len(paths) == 6
    parsed = [load_comparison_protocol(path)[0] for path in paths]
    assert {item["protocol_id"] for item in parsed} == set(BENCHMARK_PROTOCOLS)
    assert all(len(item["protocol_digest"]) == 64 for item in parsed)
    assert main(["protocol-check", "--protocol", str(paths[0])]) == 0
    assert "protocol_digest" in json.loads(capsys.readouterr().out)


def test_budget_ceiling_enforces_final_and_tuning_usage():
    budget = BenchmarkBudget(protocol="equal_operator_calls/v1", max_forward_calls=5, tuning_trials=2, tuning_max_forward_calls=3)
    with pytest.raises(RuntimeError, match="forward-call budget exceeded"):
        enforce_budget(forward_calls=6, adjoint_calls=0, budget=budget)
    with pytest.raises(RuntimeError, match="tuning forward-call budget exceeded"):
        enforce_budget(forward_calls=5, adjoint_calls=0, budget=budget, tuning_forward_calls=4)


def test_context_environment_and_result_round_trip_preserve_multi_axis_schema():
    context = _context()
    restored = FairComparisonContext.from_mapping(json.loads(json.dumps(context.to_dict())))
    assert restored.digest == context.digest
    payload = _record(context=restored, robustness={"seed": 17, "noise_level": .01}).to_dict()
    round_trip = BenchmarkResult.from_mapping(json.loads(json.dumps(payload))).to_dict()
    assert round_trip["context_digest"] == context.digest
    assert round_trip["environment_digest"] == context.environment["digest"]
    assert set(round_trip["axes"]) == {"reconstruction_quality", "data_consistency", "optimization_behavior", "computational_efficiency", "robustness"}
    assert "score" not in round_trip
    assert per_axis_rankings([_record(), _record("cgls", psnr=29.0)])["quality.psnr"] == ["sirt", "cgls"]


def test_fairness_rejects_context_mismatch_invalid_exception_and_cross_stratum_ranking():
    first = _record()
    other_context = _context(seed=29)
    second = _record("cgls", context=other_context)
    assert not check_fairness([first, second])["fair"]
    invalid_exception = _record(fairness_exceptions=({"code": "native_backend"},))
    assert "missing required fields" in " ".join(check_fairness([invalid_exception])["issues"])
    emission = _record("mlem", observation_domain="emission_counts", observation_stratum="emission_count")
    report = check_fairness([first, emission])
    assert not report["rankable"]
    with pytest.raises(ValueError, match="incompatible observation strata"):
        pareto_front([first, emission])


def test_justified_backend_exception_is_serialized():
    exception = {"code": "native_backend_reconstruction", "reason": "FDK uses a native backend", "basis": "No separable forward/adjoint calls are exposed", "affected_fields": ["forward_calls", "adjoint_calls"], "fairness_impact": "Rank only in the fdk_backend stratum and report backend calls"}
    row = _record(fairness_exceptions=(exception,))
    assert check_fairness([row])["fair"]
    assert row.to_dict()["fairness_exceptions"] == [exception]


def test_protocol_digest_and_environment_digest_fail_closed():
    protocol = ComparisonProtocol.from_mapping({"schema_version": "ct.fair_benchmark_protocol.v1", "protocol_id": "fixed_defaults/v1", "budget": {"tuning_trials": 0}})
    with pytest.raises(ValueError, match="protocol_digest"):
        ComparisonProtocol.from_mapping({**protocol.to_dict(), "protocol_digest": "bad"})
    damaged = _context().to_dict()
    damaged["environment"]["python"] = "changed"
    with pytest.raises(ValueError, match="environment digest"):
        FairComparisonContext.from_mapping(damaged)

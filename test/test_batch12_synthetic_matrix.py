from __future__ import annotations

import hashlib
import json
from pathlib import Path

from batch12_synthetic_matrix import (
    DIRECT_ALGORITHMS,
    ITERATIVE_ALGORITHMS,
    SCHEMA_VERSION,
    run_matrix,
)


ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "data" / "batch12_synthetic" / "protocol.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_batch12_matrix_covers_all_algorithms_and_required_scenarios(tmp_path: Path):
    evidence = run_matrix(PROTOCOL, tmp_path)
    assert evidence["schema_version"] == SCHEMA_VERSION
    assert evidence["ordinary_ct_only"] is True
    assert evidence["summary"]["algorithm_count"] == 12
    assert evidence["summary"]["scenario_count"] == evidence["summary"]["passed_scenario_count"]
    assert set(evidence["algorithms"]) == set(ITERATIVE_ALGORITHMS) | set(DIRECT_ALGORITHMS)

    iterative_required = {
        "normal", "max_iterations", "stalled", "diverged", "nonfinite",
        "invalid_parameters", "tolerance_sensitivity", "trajectory_consistency",
    }
    direct_required = {
        "valid", "nonfinite", "invalid_parameters", "unavailable", "max_iterations",
        "stalled", "diverged", "tolerance_sensitivity", "trajectory_consistency",
    }
    for name in ITERATIVE_ALGORITHMS:
        records = evidence["algorithms"][name]["scenarios"]
        assert {record["scenario"] for record in records} == iterative_required
        assert all(record["passed"] for record in records)
    for name in DIRECT_ALGORITHMS:
        records = evidence["algorithms"][name]["scenarios"]
        assert {record["scenario"] for record in records} == direct_required
        assert all(record["passed"] for record in records)
        not_applicable = {
            record["scenario"] for record in records if record["actual_status"] == "not_applicable"
        }
        assert {"max_iterations", "stalled", "diverged", "tolerance_sensitivity", "trajectory_consistency"} <= not_applicable


def test_batch12_evidence_schema_and_checksums_are_finite_and_valid(tmp_path: Path):
    evidence = run_matrix(PROTOCOL, tmp_path)
    result_text = (tmp_path / "results.json").read_text(encoding="utf-8")
    assert "NaN" not in result_text and "Infinity" not in result_text
    assert json.loads(result_text) == evidence
    checksums = json.loads((tmp_path / "checksums.json").read_text(encoding="utf-8"))
    assert checksums["schema_version"] == "ct.batch12_checksums.v1"
    assert checksums["files"]["protocol.json"] == _sha256(PROTOCOL)
    assert checksums["files"]["results.json"] == _sha256(tmp_path / "results.json")

    checked_in_root = PROTOCOL.parent
    checked_in = json.loads((checked_in_root / "results.json").read_text(encoding="utf-8"))
    checked_in_checksums = json.loads(
        (checked_in_root / "checksums.json").read_text(encoding="utf-8")
    )
    assert checked_in == evidence
    assert checked_in_checksums == checksums
    assert checked_in_checksums["files"]["results.json"] == _sha256(
        checked_in_root / "results.json"
    )

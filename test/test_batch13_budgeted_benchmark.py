from __future__ import annotations

import json
from pathlib import Path

from batch13_budgeted_benchmark import estimate, load_config, validate_evidence


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "benchmarks" / "batch13_budgeted_32.yaml"
EVIDENCE = ROOT / "artifacts" / "ct_agent_trustworthy_v1"


def test_batch13_frozen_matrix_estimate_stays_within_budget() -> None:
    value = estimate(load_config(CONFIG))
    assert value["within_budget"] is True
    assert value["fixed_jobs"] == 29
    assert value["unique_tuning_jobs"] == 84
    assert value["robustness_jobs"] == 176
    assert value["total_numerical_jobs"] == 289


def test_batch13_compact_evidence_schema_checksums_fairness_and_budgets() -> None:
    validation = validate_evidence(EVIDENCE)
    assert validation == {
        "schema_records_validated": 205,
        "checksum_files_validated": 8,
        "fairness_groups": 40,
        "fixed_records": 29,
        "robustness_records": 176,
        "tuning_algorithms": 7,
        "oracle_records": 7,
    }


def test_batch13_keeps_strata_axes_and_oracle_separate_without_total_score() -> None:
    summary = json.loads((EVIDENCE / "summary.json").read_text(encoding="utf-8"))
    parameter = json.loads((EVIDENCE / "parameter_study.json").read_text(encoding="utf-8"))
    assert summary["fixed_defaults"]["transmission_records"] == 27
    assert summary["fixed_defaults"]["count_records"] == 2
    assert summary["robustness"]["transmission_records"] == 144
    assert summary["robustness"]["count_records"] == 32
    assert len(summary["axes"]) == 5
    assert summary["aggregate_score"] is None
    assert parameter["oracle_excluded_from_agent_and_normal_ranking"] is True
    assert all(
        record["agent_available"] is False
        and record["include_in_normal_ranking"] is False
        for record in parameter["oracle_records"]
    )

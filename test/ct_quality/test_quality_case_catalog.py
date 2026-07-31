from __future__ import annotations

from inv_framework.benchmarks import list_ct_cases, load_ct_case

from .benchmark import QUALITY_CASE_IDS


def test_quality_cases_are_catalogued_as_independent_synthetic_references():
    records = list_ct_cases({"tag": "quality"})
    assert {record["case_id"] for record in records} == set(QUALITY_CASE_IDS)
    for record in records:
        assert record["reference_kind"] == "backend_reference"
        assert "synthetic_tissue" in record["capability_tags"]


def test_quality_case_manifest_and_arrays_are_operator_ready():
    case = load_ct_case(QUALITY_CASE_IDS[0])
    assert case.truth.shape == (1, 1, 128, 128)
    assert case.measurement.shape == (1, 1, 180, 128)
    assert case.geometry["pixel_spacing_cm"] == 0.1
    assert case.metadata["ground_truth"]["origin"].endswith("not patient data")
    assert case.metadata["provenance"]["reference_kind"] == "backend_reference"
    assert "ASTRA" in case.metadata["provenance"]["generator"]
    assert "ParallelBeamRadon2D" in case.metadata["provenance"][
        "reconstruction_operator_under_test"
    ]

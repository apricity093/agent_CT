from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from inv_framework.benchmarks import CTTestCase, load_ct_case, write_ct_case


DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
REQUIRED_MANIFEST_FIELDS = {
    "schema_version",
    "case_id",
    "modality",
    "dimension",
    "geometry",
    "ground_truth",
    "measurement",
    "provenance",
    "capability_tags",
    "sha256",
}


def test_manifests_follow_v1_schema_and_checksums_are_present():
    catalog = json.loads((DATA_ROOT / "catalog.json").read_text(encoding="utf-8"))
    assert catalog["schema_version"] == "1.0"
    for record in catalog["cases"]:
        manifest_path = DATA_ROOT / record["path"] / "case.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert REQUIRED_MANIFEST_FIELDS <= set(manifest)
        assert manifest["case_id"] == record["case_id"]
        assert manifest["modality"] == "xray_ct"
        assert manifest["dimension"] in {2, 3}
        assert len(manifest["sha256"]["arrays.h5"]) == 64
        assert set(manifest["sha256"]["arrays.h5"]) <= set("0123456789abcdef")


def test_case_write_and_load_round_trip(tmp_path):
    case = load_ct_case("parallel_2d/disk_analytic_32", data_root=DATA_ROOT)
    case_dir = tmp_path / "roundtrip"
    record = write_ct_case(case, case_dir)
    record["path"] = "roundtrip"
    (tmp_path / "catalog.json").write_text(
        json.dumps({"schema_version": "1.0", "cases": [record]}),
        encoding="utf-8",
    )
    loaded = load_ct_case(case.case_id, data_root=tmp_path)
    assert torch.equal(loaded.truth, case.truth)
    assert torch.equal(loaded.measurement_clean, case.measurement_clean)
    assert torch.equal(loaded.measurement, case.measurement)
    assert torch.equal(loaded.roi_mask, case.roi_mask)


def test_case_rejects_manifest_shape_drift():
    with pytest.raises(ValueError, match="domain_shape"):
        CTTestCase(
            case_id="invalid/shape",
            truth=torch.zeros(1, 1, 4, 4),
            measurement_clean=torch.zeros(1, 1, 2, 4),
            measurement=torch.zeros(1, 1, 2, 4),
            geometry={"domain_shape": [1, 5, 5], "range_shape": [1, 2, 4]},
            metadata={},
        )


def test_checksum_verification_detects_modified_arrays(tmp_path):
    case = load_ct_case("parallel_2d/disk_analytic_32", data_root=DATA_ROOT)
    case_dir = tmp_path / "tampered"
    record = write_ct_case(case, case_dir)
    record["path"] = "tampered"
    (tmp_path / "catalog.json").write_text(
        json.dumps({"schema_version": "1.0", "cases": [record]}),
        encoding="utf-8",
    )
    with (case_dir / "arrays.h5").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        load_ct_case(case.case_id, data_root=tmp_path)

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from inv_framework.benchmarks import list_ct_cases, load_ct_case


DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


def test_catalog_is_agent_readable_and_filterable():
    records = list_ct_cases(data_root=DATA_ROOT)
    assert len(records) >= 6
    assert records == sorted(records, key=lambda item: item["case_id"])
    assert len({record["case_id"] for record in records}) == len(records)
    assert all(record["capability_tags"] for record in records)
    assert all(record["reference_kind"] for record in records)
    json.dumps(records)

    independent = list_ct_cases(
        {"reference_kind": "analytic_independent"}, data_root=DATA_ROOT
    )
    assert [record["case_id"] for record in independent] == [
        "parallel_2d/disk_analytic_32"
    ]
    noisy_2d = list_ct_cases(
        {"dimension": 2, "tags": ["parallel", "noisy"]}, data_root=DATA_ROOT
    )
    assert {record["noise_model"] for record in noisy_2d} == {
        "gaussian_relative",
        "poisson_log",
    }


@pytest.mark.parametrize(
    "case_id",
    [record["case_id"] for record in list_ct_cases(data_root=DATA_ROOT)],
)
def test_every_catalog_case_loads_with_declared_shapes(case_id):
    case = load_ct_case(case_id, data_root=DATA_ROOT)
    assert tuple(case.geometry["domain_shape"]) == tuple(case.truth.shape[1:])
    assert tuple(case.geometry["range_shape"]) == tuple(case.measurement.shape[1:])
    assert case.measurement.shape == case.measurement_clean.shape
    assert case.truth.dtype == torch.float32
    assert case.measurement.dtype == torch.float32
    assert torch.isfinite(case.truth).all()
    assert torch.isfinite(case.measurement).all()
    assert case.source_path is not None


def test_load_case_migrates_float_dtype_without_changing_masks():
    case = load_ct_case(
        "parallel_2d/disk_analytic_32",
        data_root=DATA_ROOT,
        dtype=torch.float64,
    )
    assert case.truth.dtype == torch.float64
    assert case.measurement.dtype == torch.float64
    assert case.roi_mask is not None
    assert case.roi_mask.dtype == torch.bool


def test_unknown_case_lists_available_ids():
    with pytest.raises(KeyError, match="available"):
        load_ct_case("missing/case", data_root=DATA_ROOT)

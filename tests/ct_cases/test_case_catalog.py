from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from inv_framework.benchmarks import list_ct_cases, load_ct_case


DATA_ROOT = Path(__file__).resolve().parents[2] / "tests" / "data"


def test_catalog_is_agent_readable_and_filterable():
    records = list_ct_cases(data_root=DATA_ROOT)
    assert len(records) == 5
    assert len({record["case_id"] for record in records}) == len(records)
    assert len(list_ct_cases({"dimension": 2}, data_root=DATA_ROOT)) == 4
    assert len(list_ct_cases({"geometry_type": "cone_3d"}, data_root=DATA_ROOT)) == 1
    assert all(record["reference_kind"] for record in records)
    json.dumps(records)


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
    if case.geometry["type"] == "parallel_2d":
        assert case.truth.shape == (1, 1, 64, 64)
        assert case.metadata["provenance"]["dataset"] == "ct-ellipses-new-20261101"
        assert case.metadata["provenance"]["reference_kind"] == "model_matched"


def test_load_case_migrates_float_dtype_without_changing_masks():
    case = load_ct_case(
        "parallel_2d__ellipses_dense_clean",
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

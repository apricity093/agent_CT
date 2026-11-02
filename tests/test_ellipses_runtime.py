import json
from pathlib import Path

import numpy as np
import pytest
import torch

from inv_framework.benchmarks import load_ct_case
from inv_framework.api.outputs import run_case, run_suite

ROOT = Path(__file__).resolve().parents[1]


def test_fixed_parameter_suite_records_failure(tmp_path):
    suite = {
        "name": "ellipses_smoke",
        "output_root": str(tmp_path / "runs"),
        "continue_on_error": True,
        "groups": [{
            "algorithms": [{"name": "fbp", "config": str(ROOT / "configs/algorithms/fbp.yaml")}],
            "cases": ["parallel_2d__ellipses_dense_clean", "missing_case"],
        }],
    }
    path = tmp_path / "suite.yaml"
    path.write_text(json.dumps(suite))
    result = run_suite(path)
    assert [record["status"] for record in result["records"]] == ["success", "failed"]
    assert not result["evaluation"]["passed"]


def test_ellipses_source_and_photon_observations():
    source = np.load(ROOT / "tests/data/reference_images/test.npy")[0]
    case = load_ct_case("parallel_2d__ellipses_sparse_poisson")
    np.testing.assert_array_equal(case.truth.numpy()[0, 0], source * np.float32(0.02))
    counts, incident = case.public_measurement["counts"], case.public_measurement["I0"]
    torch.testing.assert_close(
        case.measurement, -(counts.clamp_min(1) / incident).log()
    )
    assert counts.shape == case.measurement.shape == incident.shape


@pytest.mark.parametrize("algorithm", ["sart", "os_sart", "fista", "pdhg"])
def test_raw_operator_runtime(tmp_path, algorithm):
    result = run_case(
        algorithm,
        "parallel_2d__ellipses_sparse_gaussian",
        ROOT / "configs/algorithms" / (algorithm + ".yaml"),
        tmp_path / algorithm,
        parameter_overrides={"num_iterations": 2},
    )
    assert result["status"] == "success", result
    assert result["diagnostics"]["iterations_completed"] <= 2
    assert np.isfinite(result["metrics"]["rmse"])
    assert (tmp_path / algorithm / "diagnostics.json").is_file()

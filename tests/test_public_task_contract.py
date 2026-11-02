"""Public execution, split isolation and termination contracts."""
import json
import shutil
import h5py
import numpy as np
import pytest
from inv_framework.api import CTTaskRequest, run_ct_task, validate_ct_request

CASE = "parallel_2d__ellipses_sparse_gaussian"


@pytest.mark.parametrize("device", [None, "cuda"])
def test_fdk_validation_uses_requested_device(device):
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for FDK")
    request = CTTaskRequest("cone_3d/spheres_astra_12", "fdk", backend="astra", device=device)
    assert validate_ct_request(request)["filter_type"] == "ram-lak"
    request.device = "cpu"
    with pytest.raises(ValueError, match="CUDA device"):
        validate_ct_request(request)


@pytest.mark.parametrize("reason,state", [
    ("joint_prox_inner_max_iterations", "max_iterations"),
    ("joint_prox_returned_precision", "stalled"),
])
def test_fista_proximal_stop_preserves_reason_and_point(monkeypatch, reason, state):
    from inv_framework.core import regularizer
    def stop(*args, **kwargs):
        raise regularizer.ProximalAccuracyError(reason)
    monkeypatch.setattr(regularizer, "joint_prox", stop)
    result = run_ct_task(CTTaskRequest(CASE, "fista", fidelity="gaussian_nll",
        regularizer="tv", params={"num_iterations": 2, "min_value": 0.},
        stopping_policy={"discrepancy": {"enabled": False}}))
    assert result.status == "success", result.error_message
    assert result.convergence_status == state
    assert result.reason_code == reason
    assert not result.converged
    assert np.isfinite(result.reconstruction).all()


def test_public_without_truth_and_heldout_isolation(tmp_path):
    source = __import__("pathlib").Path(__file__).parent / "data"
    shutil.copytree(source / "cases" / CASE, tmp_path / "cases" / CASE)
    catalog = json.loads((source / "catalog.json").read_text())
    (tmp_path / "catalog.json").write_text(json.dumps(catalog))
    with h5py.File(tmp_path / "cases" / CASE / "arrays.h5", "a") as handle:
        del handle["truth"]
    request = CTTaskRequest(CASE, "sirt", data_root=str(tmp_path), params={"max_iter": 2},
                            fit_view_indices=list(range(16)), heldout_view_indices=list(range(16, 32)))
    validate_ct_request(request)
    first = run_ct_task(request)
    assert first.status == "success", first.error_message
    assert "psnr" not in first.metrics
    assert "held_out_projection_residual" in first.metrics
    with h5py.File(tmp_path / "cases" / CASE / "arrays.h5", "a") as handle:
        data = handle["measurement/y_observed"]
        values = data[:]
        values[..., 16:, :] += 1
        data[:] = values
    second = run_ct_task(request)
    np.testing.assert_array_equal(first.reconstruction, second.reconstruction)
    assert first.metrics["held_out_projection_residual"] != second.metrics["held_out_projection_residual"]


def test_cancel_preserves_last_iteration():
    result = run_ct_task(CTTaskRequest(CASE, "sirt", params={"max_iter": 8},
                                      callback=lambda record: False))
    assert result.convergence_status == "cancelled"
    assert result.status == "failed"
    assert result.iterations == 1
    assert len(result.history) == 1
    assert np.isfinite(result.reconstruction).all()


def test_stopping_and_invalid_combination():
    request = CTTaskRequest(CASE, "sirt", params={"max_iter": 2},
                           stopping_policy={"discrepancy": {"enabled": False}})
    result = run_ct_task(request)
    assert result.convergence_status == "max_iterations", result.error_message
    assert result.reason_code
    with pytest.raises(ValueError):
        validate_ct_request(CTTaskRequest(CASE, "mlem"))


@pytest.mark.parametrize("algorithm,params", [("fbp", {}), ("sart", {"max_iter": 1}),
                                               ("os_sart", {"max_iter": 1, "num_subsets": 4})])
def test_astra_parallel_execution(algorithm, params):
    pytest.importorskip("astra")
    result = run_ct_task(CTTaskRequest(CASE, algorithm, backend="astra", params=params))
    assert result.status == "success", result.error_message
    assert np.isfinite(result.reconstruction).all()


def test_physical_bounds_override_defaults():
    resolved = validate_ct_request(CTTaskRequest(CASE, "sirt",
        problem_constraints={"min_value": 0.001, "max_value": 0.02}))
    assert resolved["min_value"] == 0.001
    assert resolved["max_value"] == 0.02


def test_orthogonal_metrics_use_case_range():
    from inv_framework.api import orthogonal_slices, orthogonal_metrics
    from inv_framework.ct.cases import load_ct_case
    case = load_ct_case("cone_3d/spheres_astra_12")
    planes = orthogonal_slices(case.truth)
    assert set(planes) == {"axial", "coronal", "sagittal"}
    metrics = orthogonal_metrics(case.truth, case.truth, case.metadata["ground_truth"]["data_range"])
    assert all(row["rmse"] == 0 and abs(row["ssim"] - 1) < 1e-6 for row in metrics.values())


@pytest.mark.parametrize("raises", [False, True])
def test_astra_release_on_cancel_or_exception(monkeypatch, raises):
    pytest.importorskip("astra")
    from inv_framework.api import execution
    built = []
    original = execution.build_operator
    def capture(*args, **kwargs):
        operator = original(*args, **kwargs)
        built.append(operator)
        return operator
    monkeypatch.setattr(execution, "build_operator", capture)
    def stop(record):
        if raises:
            raise RuntimeError("callback failure")
        return False
    request = CTTaskRequest(CASE, "os_sart", backend="astra",
        params={"max_iter": 2, "num_subsets": 4}, callback=stop)
    if raises:
        with pytest.raises(RuntimeError, match="callback failure"):
            run_ct_task(request)
    else:
        assert run_ct_task(request).convergence_status == "cancelled"
    assert built[0].projector is None
    assert built[0].subsets == {}

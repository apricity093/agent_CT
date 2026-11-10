import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from inv_framework import CTTaskRequest, CTTaskResult, list_capabilities, run_ct_task
from inv_framework.api.schema import normalize_request
from inv_framework.api import execution

DENSE = "parallel_2d__ellipses_dense_clean"
POISSON = "parallel_2d__ellipses_sparse_poisson"


def test_discovery_without_loading_cases(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("discovery loaded case arrays")
    monkeypatch.setattr(execution, "load_ct_case", forbidden)
    capabilities = list_capabilities()
    json.dumps(capabilities, allow_nan=False)
    assert {r["name"] for r in capabilities["algorithms"]} == set(execution.SOLVER_SPECS)
    assert "cases" not in capabilities
    assert capabilities["backends"]["torch"]["geometries"] == ["parallel_2d"]
    assert capabilities["backends"]["leap"]["runnable"] is False
    assert "transmission_poisson_nll" in capabilities["fidelities"]
    for record in capabilities["algorithms"]:
        _, defaults, _ = execution.load_algorithm_config(Path(__file__).resolve().parents[1] / "configs/algorithms" / record["config_filename"])
        assert record["default_parameters"] == defaults


def test_merge_and_request_ownership():
    request = CTTaskRequest(POISSON, "pdhg", fidelity="transmission_poisson_nll", regularizer="tv",
                            params={"max_iter": 2, "min_value": 0})
    _, _, params = normalize_request(request)
    assert params["num_iterations"] == 2
    assert params["data_fidelity"] == request.fidelity
    assert params["regularizer"] == "tv"
    assert params["strength"] == .01
    assert request.params == {"max_iter": 2, "min_value": 0}


@pytest.mark.parametrize("fields", [
    {"params": {"typo": 1}}, {"params": {"max_iter": 2, "num_iterations": 3}},
    {"algorithm": "pdhg", "fidelity": "l2", "params": {"data_fidelity": "wls"}},
    {"backend": "leap"}, {"algorithm": "mlem"},
    {"algorithm": "pdhg", "fidelity": "transmission_poisson_nll"},
    {"extra": True}, {"algorithm": "missing"}, {"case_id": "missing"},
    {"algorithm": "sart", "params": {"max_iter": 0}},
])
def test_expected_request_failures(fields):
    result = run_ct_task({"case_id": DENSE, "algorithm": "fbp", **fields})
    assert result.status == "failed"
    assert result.error_message
    assert not result.converged
    json.dumps(asdict(result), allow_nan=False)


@pytest.mark.parametrize("task_request", [
    {"case_id": DENSE, "algorithm": "fbp"},
    CTTaskRequest(DENSE, "sart", params={"max_iter": 2}),
    CTTaskRequest(POISSON, "pdhg", fidelity="transmission_poisson_nll", regularizer="tv",
                  params={"max_iter": 2, "min_value": 0}),
])
def test_in_memory_execution(task_request, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = run_ct_task(task_request)
    assert isinstance(result, CTTaskResult)
    assert result.status == "success", result.error_message
    assert np.isfinite(result.reconstruction).all()
    assert all(np.isfinite(v) for v in result.metrics.values())
    assert list(tmp_path.iterdir()) == []
    if result.algorithm_summary != "fbp":
        assert result.iterations == 2
        assert result.converged is False


@pytest.mark.parametrize("failure", [RuntimeError("implementation bug"), KeyboardInterrupt()])
def test_unexpected_exception_propagation(monkeypatch, failure):
    def broken(*args):
        raise failure
    monkeypatch.setattr(execution, "build_solver", broken)
    with pytest.raises(type(failure)):
        run_ct_task(CTTaskRequest(DENSE, "fbp"))


def test_fdk_dispatch_uses_backend_gate_without_discovery(monkeypatch):
    from inv_framework.api import dispatcher
    def forbidden():
        raise AssertionError("dispatch must not enumerate capabilities")
    monkeypatch.setattr(dispatcher, "list_capabilities", forbidden)
    monkeypatch.setattr(execution.torch.cuda, "is_available", lambda: False)
    result = run_ct_task(CTTaskRequest("cone_3d/spheres_astra_12", "fdk", backend="astra"))
    assert result.status == "failed"
    assert "PyTorch CUDA" in result.error_message


@pytest.mark.parametrize("cuda_available", ["torch_unavailable", "astra_unavailable", "available"])
def test_cuda_fdk(monkeypatch, request, cuda_available):
    import importlib.util
    # Exercise the installed-but-no-CUDA branch without changing dependencies.
    if cuda_available == 'torch_unavailable':
        monkeypatch.setattr(execution.torch.cuda, "is_available", lambda: False)
    if importlib.util.find_spec("astra") is None:
        pytest.skip("ASTRA is not installed")
    if not execution.torch.cuda.is_available():
        pytest.skip("PyTorch CUDA is unavailable")
    import astra
    if cuda_available == "astra_unavailable":
        monkeypatch.setattr(astra, "use_cuda", lambda: False)
    if not astra.use_cuda():
        pytest.skip("ASTRA CUDA is unavailable")
    result = run_ct_task(CTTaskRequest("cone_3d/spheres_astra_12", "fdk", backend="astra"))
    assert result.status == "success", result.error_message
    assert result.reconstruction.shape == (1, 12, 12, 12)
    assert np.isfinite(result.reconstruction).all()
    assert np.isfinite(result.metrics["ssim"])


def test_numerical_failure_preserves_actual_iterations(monkeypatch, tmp_path):
    from dataclasses import replace
    from inv_framework.api.outputs import run_case
    original = execution.build_solver
    def failing(*args):
        solver = original(*args)
        solve = solver.solve_detailed
        def numerical(*args, **kwargs):
            result = solve(*args, **kwargs)
            return replace(result, status="numerical_error", stopping_reason="synthetic numerical failure")
        solver.solve_detailed = numerical
        return solver
    monkeypatch.setattr(execution, "build_solver", failing)
    result = run_ct_task(CTTaskRequest(DENSE, "sart", params={"max_iter": 2}))
    assert result.status == "failed"
    assert result.iterations == 2
    assert "synthetic numerical failure" in result.error_message
    legacy = run_case("sart", DENSE, Path(__file__).resolve().parents[1] / "configs/algorithms/sart.yaml",
                      tmp_path / "run", parameter_overrides={"num_iterations": 2})
    assert legacy["diagnostics"]["iterations_completed"] == 2
    assert (tmp_path / "run/partial_reconstruction.pt").is_file()

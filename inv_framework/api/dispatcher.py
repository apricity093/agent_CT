"""Capability discovery and in-memory CT task dispatch."""
import importlib.util
import time
from pathlib import Path
import torch

from inv_framework.api import execution
from inv_framework.ct.cases import list_ct_cases
from inv_framework.ct.registry import ALGORITHM_ALIASES, regularizer_records
from .schema import CTTaskRequest, CTTaskResult, normalize_request


def list_capabilities(data_root=None):
    cuda = torch.cuda.is_available()
    astra_installed = importlib.util.find_spec("astra") is not None
    astra_cuda = False
    if astra_installed:
        import astra
        astra_cuda = bool(astra.use_cuda())
    backends = {
        "torch": {"implemented": True, "dependency_installed": True, "runnable": True,
                  "geometries": ["parallel_2d"], "device": "cpu", "cuda_available": cuda},
        "astra": {"implemented": True, "dependency_installed": astra_installed,
                  "runnable": astra_installed, "geometries": ["parallel_2d", "cone_3d"], "cuda_available": cuda and astra_cuda, "device": "cpu/cuda"},
        "leap": {"implemented": True, "dependency_installed": importlib.util.find_spec("leapctype") is not None,
                 "runnable": importlib.util.find_spec("leapctype") is not None, "geometries": ["parallel_2d"], "reason": "Requires LLNL LEAP"},
    }
    algorithms = execution.solver_records()
    for record in algorithms:
        name = record["name"]
        _, defaults, _ = execution.load_algorithm_config(
            Path(__file__).resolve().parents[2] / "configs/algorithms" / f"{name}.yaml", name)
        backend = "astra" if name == "fdk" else "torch"
        record.update(backends=["astra"] if name == "fdk" else ["torch", "astra", "leap"], default_parameters=defaults, implemented=record["build_reachable"],
                      execution_backend=backend, dependency_installed=backends[backend]["dependency_installed"],
                      runnable=(cuda and astra_cuda) if name == "fdk" else backends[backend]["runnable"])
    fidelities = {}
    for record in algorithms:
        for fidelity, requirements in record["metadata"].get("fidelity_requirements", {}).items():
            fidelities.setdefault(fidelity, {})[record["name"]] = requirements
    return {"fidelities": fidelities, "algorithms": algorithms, "aliases": dict(ALGORITHM_ALIASES),
            "regularizers": regularizer_records(), "backends": backends,
            "cases": list_ct_cases(data_root=data_root), "default_backend": "torch",
            "deep_priors": {"dip": "library only; no catalog task builder",
                            "inr": "library only; no catalog task builder",
                            "dps": "library only; requires a configured diffusion model"}}


def run_ct_task(request: CTTaskRequest | dict) -> CTTaskResult:
    """Return expected request/backend/numerical failures; propagate implementation errors."""
    started = time.perf_counter()
    case_id = request.get("case_id", "") if isinstance(request, dict) else getattr(request, "case_id", "")
    summary = request.get("algorithm", "") if isinstance(request, dict) else getattr(request, "algorithm", "")
    try:
        request, name, parameters = normalize_request(request)
        summary = name
        if "data_fidelity" in parameters or "regularizer" in parameters:
            summary += f"(fidelity={parameters.get('data_fidelity', 'l2')}, reg={parameters.get('regularizer', 'none')})"
        cases = {row["case_id"]: row for row in list_ct_cases(data_root=request.data_root)}
        if request.case_id not in cases:
            raise execution.ConfigError(f"unknown CT case: {request.case_id}")
        geometry = cases[request.case_id]["geometry_type"]
        if request.backend == "leap" and importlib.util.find_spec("leapctype") is None:
            raise execution.BackendUnavailable("LLNL LEAP is not installed")
        if geometry == "cone_3d" and request.backend != "astra":
            raise execution.ConfigError("cone_3d requires backend=astra")
        device = request.device or ("cuda" if geometry == "cone_3d" else "cpu")
        if request.backend == "astra":
            if importlib.util.find_spec("astra") is None:
                raise execution.BackendUnavailable("astra-toolbox is not installed")
            if geometry == "cone_3d" or torch.device(device).type == "cuda":
                execution.require_astra_cuda(device)
        if torch.device(device).type == "cuda" and not torch.cuda.is_available():
            raise execution.BackendUnavailable("PyTorch CUDA unavailable")
        result = execution.execute_case(name, request.case_id, parameters,
            device=device, data_root=request.data_root, backend=request.backend,
            fit_view_indices=request.fit_view_indices, heldout_view_indices=request.heldout_view_indices,
            stopping_policy=request.stopping_policy, problem_constraints=request.problem_constraints,
            callback=request.callback)
        diagnostics = result.get("diagnostics", {})
        status = result.get("convergence_status")
        success = result["status"] == "success" and status not in {"cancelled", "resource_exhausted", "diverged", "numerical_error", "invalid_parameters"}
        bundle = result.get("bundle")
        metrics = {key: float(value) for key, value in result.get("metrics", {}).items()
                   if key in {"psnr", "ssim", "rmse", "relative_error", "data_residual", "measurement_norm", "held_out_projection_residual", "held_out_measurement_norm"}}
        return CTTaskResult("success" if success else "failed", request.case_id, summary, metrics,
                            time.perf_counter() - started, int(diagnostics.get("iterations_completed", 0)),
                            success and status == "converged",
                            bundle["reconstruction"].numpy() if bundle is not None else (result["partial_reconstruction"].numpy() if result.get("partial_reconstruction") is not None else None),
                            None if success else str(result.get("message", result.get("stopping_reason", status))),
                            convergence_status=status,
                            stopping_reason=diagnostics.get("stopping_reason", result.get("stopping_reason", status)),
                            reason_code=diagnostics.get("reason_code", status),
                            history=diagnostics.get("convergence", {}).get("trajectory", []))
    except (execution.ConfigError, execution.BackendUnavailable, execution.NumericalFailure, FileNotFoundError, ValueError, torch.cuda.OutOfMemoryError) as error:
        status = "unavailable" if isinstance(error, execution.BackendUnavailable) else "resource_exhausted" if isinstance(error, torch.cuda.OutOfMemoryError) else "numerical_error" if isinstance(error, execution.NumericalFailure) else "invalid_parameters"
        return CTTaskResult("failed", str(case_id), str(summary), {}, time.perf_counter() - started, 0, False,
                            error_message=str(error), convergence_status=status, stopping_reason=str(error), reason_code=status)

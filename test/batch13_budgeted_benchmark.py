"""Run and validate the bounded Batch 13 ordinary-CT benchmark.

Raw reconstructions and trajectories are resumable under the ignored
``.batch13-runtime`` tree.  Only compact, checksummed evidence is written to
``artifacts/ct_agent_trustworthy_v1``.  This file is benchmark orchestration;
it does not alter solver, regularizer, operator, or convergence behavior.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import yaml


CT_ROOT = Path(__file__).resolve().parents[1]
MAIN_ROOT = CT_ROOT.parents[1]
for candidate in (CT_ROOT, MAIN_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from inv_framework.benchmarks import CTTestCase, load_ct_case, write_ct_case
from inv_framework.benchmarks.protocol import (
    BenchmarkBudget,
    BenchmarkResult,
    ComparisonProtocol,
    canonical_digest,
    check_fairness,
    make_heldout_projection_split,
)
from inv_framework.ct_runtime import run_case
from inv_framework.operators.ct import ParallelBeamRadon2D


SCHEMA = "ct.batch13_budgeted_evidence.v1"
ALLOWED_NUMERICAL_STATUSES = {
    "completed_valid",
    "converged",
    "max_iterations",
    "stalled",
}
PROTOCOL_FILES = {
    "fixed_defaults/v1": "fixed_defaults_v1.yaml",
    "equal_trials/v1": "equal_trials_v1.yaml",
    "equal_tuning_time/v1": "equal_tuning_time_v1.yaml",
    "equal_operator_calls/v1": "equal_operator_calls_v1.yaml",
    "common_validation/v1": "common_validation_v1.yaml",
    "oracle_upper_bound/v1": "oracle_upper_bound_v1.yaml",
}


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(canonical_bytes(value))
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def peak_working_set_bytes() -> int | None:
    if os.name != "nt":
        return None

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    query = ctypes.windll.psapi.GetProcessMemoryInfo
    query.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
    query.restype = wintypes.BOOL
    ok = query(
        handle, ctypes.byref(counters), counters.cb
    )
    return int(counters.PeakWorkingSetSize) if ok else None


def git_revision(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-c", f"safe.directory={root}", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def tensor_sha256(value: torch.Tensor) -> str:
    return sha256_bytes(value.detach().cpu().contiguous().numpy().tobytes(order="C"))


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != "ct.batch13_budgeted_protocol.v1":
        raise ValueError("unsupported Batch 13 benchmark config")
    validate_config(value)
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("ordinary_ct_only") is not True:
        raise ValueError("Batch 13 config must remain ordinary-CT only")
    if int(config.get("resolution", 0)) != 32:
        raise ValueError("Batch 13 is frozen at 32x32")
    fixed = config["fixed_defaults"]
    if len(fixed["transmission_algorithms"]) != 9 or len(fixed["transmission_cases"]) != 3:
        raise ValueError("fixed-default transmission matrix must be 9 x 3")
    if len(fixed["count_algorithms"]) != 2:
        raise ValueError("fixed-default count matrix must contain MLEM and OSEM")
    if set(fixed["transmission_algorithms"]) & {"mlem", "osem", "fdk"}:
        raise ValueError("transmission stratum contains an incompatible algorithm")
    if set(fixed["count_algorithms"]) != {"mlem", "osem"}:
        raise ValueError("count stratum must contain exactly MLEM and OSEM")
    global_budget = config["global_budget"]
    ceilings = {
        "wall_seconds": 14400,
        "peak_memory_bytes": 8 * 1024**3,
        "result_bytes": 1024**3,
        "tuning_trials_per_algorithm": 6,
        "tuning_runtime_seconds_per_algorithm": 180,
        "tuning_forward_calls_per_algorithm": 360,
        "tuning_adjoint_calls_per_algorithm": 360,
    }
    for name, ceiling in ceilings.items():
        if float(global_budget[name]) > ceiling:
            raise ValueError(f"{name} expands the frozen Batch 13 budget")
    study = config["protocol_study"]
    for name, values in study["algorithms"].items():
        if len(values["values"]) != 4:
            raise ValueError(f"{name} must retain exactly four predeclared candidates")
    if set(study["protocols"]) != {
        "equal_trials/v1",
        "equal_tuning_time/v1",
        "equal_operator_calls/v1",
        "common_validation/v1",
    }:
        raise ValueError("protocol study does not match the accepted protocols")


def load_protocols() -> dict[str, ComparisonProtocol]:
    root = CT_ROOT / "configs" / "fair_protocols"
    protocols: dict[str, ComparisonProtocol] = {}
    for protocol_id, filename in PROTOCOL_FILES.items():
        raw = yaml.safe_load((root / filename).read_text(encoding="utf-8"))
        protocol = ComparisonProtocol.from_mapping(raw)
        if protocol.protocol_id != protocol_id:
            raise ValueError(f"protocol id mismatch in {filename}")
        protocols[protocol_id] = protocol
    return protocols


def pixel_centres(size: int) -> torch.Tensor:
    return -1.0 + (torch.arange(size, dtype=torch.float32) + 0.5) * (2.0 / size)


def shepp_logan(size: int) -> torch.Tensor:
    coordinates = pixel_centres(size)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    image = torch.zeros((size, size), dtype=torch.float32)
    ellipses = [
        (1.00, .6900, .9200, .0000, .0000, 0.0),
        (-.80, .6624, .8740, .0000, -.0184, 0.0),
        (-.20, .1100, .3100, .2200, .0000, -18.0),
        (-.20, .1600, .4100, -.2200, .0000, 18.0),
        (.10, .2100, .2500, .0000, .3500, 0.0),
        (.10, .0460, .0460, .0000, .1000, 0.0),
        (.10, .0460, .0460, .0000, -.1000, 0.0),
        (.10, .0460, .0230, -.0800, -.6050, 0.0),
        (.10, .0230, .0230, .0000, -.6060, 0.0),
        (.10, .0230, .0460, .0600, -.6050, 0.0),
    ]
    for intensity, a, b, cx, cy, angle_deg in ellipses:
        angle = math.radians(angle_deg)
        x, y = xx - cx, yy - cy
        xr = math.cos(angle) * x + math.sin(angle) * y
        yr = -math.sin(angle) * x + math.cos(angle) * y
        mask = (xr / a).square() + (yr / b).square() <= 1.0
        image = torch.where(mask, image + intensity, image)
    return image.clamp(0.0, 1.0)[None, None]


def angles_for(view_count: int, coverage_deg: float) -> torch.Tensor:
    coverage = math.radians(coverage_deg)
    if coverage_deg == 180.0:
        return torch.arange(view_count, dtype=torch.float32) * (math.pi / view_count)
    return torch.linspace(0.0, coverage, view_count, dtype=torch.float32)


def parallel_geometry(size: int, angles: torch.Tensor) -> dict[str, Any]:
    return {
        "type": "parallel_2d",
        "domain_shape": [1, size, size],
        "range_shape": [1, int(angles.numel()), size],
        "image_layout": ["channel", "y", "x"],
        "measurement_layout": ["channel", "angle", "detector"],
        "angles_rad": [float(value) for value in angles],
        "detector_count": size,
        "detector_spacing": 2.0 / size,
        "length_unit": "normalized_image_coordinate",
    }


def synthetic_case(
    *,
    case_id: str,
    size: int,
    seed: int,
    view_count: int,
    coverage_deg: float,
    noise_level: float | None = None,
    count_level: float | None = None,
) -> CTTestCase:
    angles = angles_for(view_count, coverage_deg)
    operator = ParallelBeamRadon2D(size, angles=angles, device="cpu")
    base_truth = shepp_logan(size)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if count_level is None:
        truth = base_truth.mul(0.02)
        clean = operator.forward(truth).detach()
        if noise_level:
            noise = torch.randn(clean.shape, generator=generator)
            observed = clean + noise_level * clean.norm() / noise.norm().clamp_min(1e-12) * noise
        else:
            observed = clean.clone()
        measurement = {
            "kind": "line_integral",
            "observation_model": "xray_transmission",
            "noise_model": "gaussian_relative" if noise_level else "none",
            "seed": seed,
            "parameters": {"relative_l2_fraction": float(noise_level or 0.0)},
        }
        observation_domain = "line_integral"
        data_range = 0.02
    else:
        unit_projection = operator.forward(base_truth).detach()
        scale = float(count_level) / float(unit_projection.mean().clamp_min(1e-6))
        truth = base_truth.mul(scale)
        clean = operator.forward(truth).detach().clamp_min(1e-6)
        observed = torch.poisson(clean, generator=generator)
        measurement = {
            "kind": "counts",
            "observation_model": "poisson_emission",
            "noise_model": "poisson_counts",
            "seed": seed,
            "parameters": {"mean_expected_count": float(count_level)},
        }
        observation_domain = "nonnegative_counts"
        data_range = float(truth.max())
    metadata = {
        "schema_version": "1.0",
        "case_id": case_id,
        "modality": "emission_ct" if count_level is not None else "xray_ct",
        "dimension": 2,
        "observation_domain": observation_domain,
        "ground_truth": {"quantity": "activity" if count_level is not None else "linear_attenuation", "data_range": data_range},
        "measurement": measurement,
        "provenance": {
            "reference_kind": "model_matched",
            "generator": "batch13_budgeted_benchmark/ParallelBeamRadon2D",
            "license": "project-generated",
        },
        "capability_tags": [
            "ordinary_ct",
            "2d",
            "parallel",
            "count_domain" if count_level is not None else "transmission",
        ],
    }
    return CTTestCase(
        case_id=case_id,
        truth=truth,
        measurement_clean=clean,
        measurement=observed,
        geometry=parallel_geometry(size, angles),
        metadata=metadata,
        roi_mask=base_truth > 0.0,
    )


def ensure_runtime_cases(config: Mapping[str, Any], runtime_root: Path) -> Path:
    data_root = runtime_root / "data"
    catalog_path = data_root / "catalog.json"
    expected_count = 33
    if catalog_path.exists():
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
        if len(payload.get("cases", ())) == expected_count:
            return data_root
    temporary = runtime_root / f"data.tmp.{os.getpid()}"
    if temporary.exists():
        temporary = runtime_root / f"data.tmp.{os.getpid()}.{time.time_ns()}"
    case_root = temporary / "cases"
    records: list[dict[str, Any]] = []

    def add(case: CTTestCase) -> None:
        slug = case.case_id.replace("/", "__")
        record = write_ct_case(case, case_root / slug, overwrite=False)
        record["path"] = f"cases/{slug}"
        records.append(record)

    add(synthetic_case(
        case_id=config["fixed_defaults"]["count_case"], size=32, seed=17,
        view_count=32, coverage_deg=180.0, count_level=50.0,
    ))
    robustness = config["robustness"]
    for seed in config["seeds"]:
        for noise in robustness["noise_levels"]:
            for views in robustness["view_counts"]:
                for coverage in robustness["angle_coverages_deg"]:
                    case_id = f"batch13_transmission/s{seed}_n{noise:g}_v{views}_a{coverage:g}_32"
                    add(synthetic_case(
                        case_id=case_id, size=32, seed=int(seed), view_count=int(views),
                        coverage_deg=float(coverage), noise_level=float(noise),
                    ))
        for count in robustness["count_levels"]:
            for views in robustness["view_counts"]:
                for coverage in robustness["angle_coverages_deg"]:
                    case_id = f"batch13_count/s{seed}_c{count:g}_v{views}_a{coverage:g}_32"
                    add(synthetic_case(
                        case_id=case_id, size=32, seed=int(seed), view_count=int(views),
                        coverage_deg=float(coverage), count_level=float(count),
                    ))
    temporary.mkdir(parents=True, exist_ok=True)
    atomic_json(temporary / "catalog.json", {
        "schema_version": "1.0",
        "cases": sorted(records, key=lambda item: item["case_id"]),
    })
    if data_root.exists():
        raise RuntimeError("incomplete Batch 13 runtime data root already exists; preserve it for diagnosis")
    os.replace(temporary, data_root)
    return data_root


@dataclass
class BudgetMonitor:
    started: float
    runtime_root: Path
    limits: Mapping[str, Any]
    elapsed_offset: float = 0.0

    def snapshot(self) -> dict[str, Any]:
        return {
            "wall_seconds": self.elapsed_offset + time.perf_counter() - self.started,
            "peak_working_set_bytes": peak_working_set_bytes(),
            "result_bytes": directory_bytes(self.runtime_root),
        }

    def guard(self) -> dict[str, Any]:
        value = self.snapshot()
        if value["wall_seconds"] >= float(self.limits["wall_seconds"]):
            raise RuntimeError("Batch 13 wall-clock budget exhausted")
        if value["result_bytes"] >= int(self.limits["result_bytes"]):
            raise RuntimeError("Batch 13 result-size budget exhausted")
        memory = value["peak_working_set_bytes"]
        if memory is not None and memory >= int(self.limits["peak_memory_bytes"]):
            raise RuntimeError("Batch 13 memory budget exhausted")
        return value


def load_run(path: Path) -> dict[str, Any]:
    required = ("metrics.json", "diagnostics.json", "manifest.json", "artifacts.sha256")
    if not all((path / name).is_file() for name in required):
        raise ValueError(f"incomplete finalized job: {path}")
    return {
        "status": json.loads((path / "metrics.json").read_text(encoding="utf-8"))["status"],
        "metrics": json.loads((path / "metrics.json").read_text(encoding="utf-8")),
        "diagnostics": json.loads((path / "diagnostics.json").read_text(encoding="utf-8")),
        "manifest": json.loads((path / "manifest.json").read_text(encoding="utf-8")),
    }


def run_atomic(
    *,
    monitor: BudgetMonitor,
    job_dir: Path,
    algorithm: str,
    case_id: str,
    config_path: Path,
    data_root: Path,
    max_iterations: int | None = None,
    max_forward_calls: int | None = None,
    max_adjoint_calls: int | None = None,
    parameter_overrides: Mapping[str, Any] | None = None,
    parameter_sources: Mapping[str, str] | None = None,
    fit_view_indices: Sequence[int] | None = None,
    heldout_view_indices: Sequence[int] | None = None,
    split_metadata: Mapping[str, Any] | None = None,
    fixed_compute: bool = False,
) -> tuple[dict[str, Any], bool]:
    monitor.guard()
    if job_dir.exists():
        return load_run(job_dir), True
    job_dir.parent.mkdir(parents=True, exist_ok=True)
    digest_payload = {
        "algorithm": algorithm,
        "case_id": case_id,
        "config_sha256": sha256_file(config_path),
        "max_iterations": max_iterations,
        "max_forward_calls": max_forward_calls,
        "max_adjoint_calls": max_adjoint_calls,
        "parameter_overrides": dict(parameter_overrides or {}),
        "fit_view_indices": list(fit_view_indices or ()),
        "heldout_view_indices": list(heldout_view_indices or ()),
        "fixed_compute": fixed_compute,
    }
    job_digest = canonical_digest(digest_payload)
    pending = job_dir.with_name(job_dir.name + ".pending.json")
    atomic_json(pending, {"state": "pending", "job_digest": job_digest, **digest_payload})
    temporary = job_dir.with_name(job_dir.name + f".tmp.{os.getpid()}.{time.time_ns()}")
    result = run_case(
        algorithm,
        case_id,
        config_path,
        temporary,
        device="cpu",
        data_root=data_root,
        overwrite=False,
        max_iterations=max_iterations,
        max_forward_calls=max_forward_calls,
        max_adjoint_calls=max_adjoint_calls,
        parameter_overrides=parameter_overrides,
        parameter_sources=parameter_sources,
        fit_view_indices=fit_view_indices,
        heldout_view_indices=heldout_view_indices,
        split_metadata=split_metadata,
        fixed_compute=fixed_compute,
    )
    os.replace(temporary, job_dir)
    atomic_json(job_dir / "batch13_state.json", {"state": "finalized", "job_digest": job_digest})
    pending.unlink(missing_ok=True)
    monitor.guard()
    return result, False


def compact_record(
    result: Mapping[str, Any],
    *,
    algorithm: str,
    case_id: str,
    stratum: str,
    protocol: ComparisonProtocol,
    case_hash: str,
    robustness: Mapping[str, Any] | None = None,
    tuning_usage: Mapping[str, Any] | None = None,
    context_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = dict(result.get("metrics", {}))
    diagnostics = dict(result.get("diagnostics", {}))
    convergence = dict(diagnostics.get("convergence", {}))
    resources = dict(diagnostics.get("resources", {}))
    context = {
        "case_id": case_id,
        "case_sha256": case_hash,
        "resolution": 32,
        "device": "cpu",
        "dtype": diagnostics.get("dtype", "torch.float32"),
        "observation_stratum": stratum,
        **dict(context_extra or {}),
    }
    context_digest = canonical_digest(context)
    budget = protocol.budget.to_dict()
    record = BenchmarkResult(
        algorithm=algorithm,
        solver=algorithm,
        case_id=case_id,
        geometry=diagnostics.get("geometry_type", "parallel_2d"),
        observation_domain=diagnostics.get("observation_domain"),
        observation_stratum=stratum,
        regularizer=diagnostics.get("regularizer"),
        parameters=diagnostics.get("parameters", {}),
        parameter_sources=diagnostics.get("parameter_sources", {}),
        tuning_protocol=protocol.protocol_id,
        convergence_status=convergence.get("status", metrics.get("convergence_status")),
        stopping_reason=convergence.get("stopping_reason", metrics.get("stopping_reason")),
        iterations=diagnostics.get("iterations_completed"),
        runtime_seconds=metrics.get("runtime_seconds", resources.get("runtime_seconds")),
        forward_calls=resources.get("forward_calls"),
        adjoint_calls=resources.get("adjoint_calls"),
        peak_memory_mb=resources.get("peak_memory_mb"),
        objective=convergence.get("final_objective"),
        residual=metrics.get("data_residual", convergence.get("final_residual")),
        psnr=metrics.get("psnr"),
        ssim=metrics.get("ssim"),
        rmse=metrics.get("rmse"),
        relative_l2_error=metrics.get("relative_error"),
        normalized_residual=metrics.get("normalized_residual"),
        data_fidelity=metrics.get("data_fidelity", metrics.get("deviance")),
        status=str(metrics.get("status", result.get("status", "failed"))),
        budget=budget,
        device="cpu",
        dtype=diagnostics.get("dtype", "torch.float32"),
        resources={
            key: resources.get(key)
            for key in (
                "total_operator_calls", "operator_runtime_seconds", "wall_time_seconds",
                "peak_gpu_memory_mb", "gpu_runtime_seconds", "prox_calls",
                "prox_iterations", "backend_reconstruction_calls", "phase_sum_consistent",
            )
        },
        tuning_usage=dict(tuning_usage or {}),
        robustness=dict(robustness or {}),
        context_digest=context_digest,
        protocol_digest=protocol.digest,
    ).to_dict()
    record.update({
        "case_sha256": case_hash,
        "context": context,
        "quality_metrics_private_evaluator": bool(metrics.get("ground_truth_available")),
        "execution_status": metrics.get("execution_status", diagnostics.get("execution_status")),
    })
    return record


def case_hash(data_root: Path, case_id: str) -> str:
    case = load_ct_case(case_id, data_root=data_root, verify_checksum=True)
    source = case.source_path
    if source is not None:
        manifest = json.loads((source / "case.json").read_text(encoding="utf-8"))
        return str(manifest["sha256"]["arrays.h5"])
    return tensor_sha256(case.measurement)


def algorithm_config(name: str) -> Path:
    return CT_ROOT / "configs" / "algorithms" / f"{name}.yaml"


def status_is_accepted(record: Mapping[str, Any]) -> bool:
    if record.get("status") != "success":
        return False
    return record.get("convergence_status") in ALLOWED_NUMERICAL_STATUSES


def fixed_default_matrix(
    config: Mapping[str, Any], protocols: Mapping[str, ComparisonProtocol], monitor: BudgetMonitor,
    runtime_root: Path, generated_data: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    fixed = config["fixed_defaults"]
    protocol = protocols["fixed_defaults/v1"]
    public_data = CT_ROOT / "test" / "data"
    for algorithm in fixed["transmission_algorithms"]:
        for case_id in fixed["transmission_cases"]:
            result, _ = run_atomic(
                monitor=monitor,
                job_dir=runtime_root / "jobs" / "fixed_defaults" / "transmission" / algorithm / case_id.replace("/", "__"),
                algorithm=algorithm, case_id=case_id, config_path=algorithm_config(algorithm),
                data_root=public_data,
            )
            records.append(compact_record(
                result, algorithm=algorithm, case_id=case_id, stratum="transmission",
                protocol=protocol, case_hash=case_hash(public_data, case_id),
            ))
    count_case = fixed["count_case"]
    for algorithm in fixed["count_algorithms"]:
        result, _ = run_atomic(
            monitor=monitor,
            job_dir=runtime_root / "jobs" / "fixed_defaults" / "count" / algorithm / count_case.replace("/", "__"),
            algorithm=algorithm, case_id=count_case, config_path=algorithm_config(algorithm),
            data_root=generated_data,
        )
        records.append(compact_record(
            result, algorithm=algorithm, case_id=count_case, stratum="emission_count",
            protocol=protocol, case_hash=case_hash(generated_data, count_case),
        ))
    return records


def fdk_capability_record(config: Mapping[str, Any]) -> dict[str, Any]:
    torch_cuda = bool(torch.cuda.is_available())
    astra_spec = importlib.util.find_spec("astra")
    astra_available = astra_spec is not None
    astra_cuda = False
    astra_version = None
    if astra_available:
        import astra
        astra_version = getattr(astra, "__version__", None)
        try:
            astra_cuda = bool(astra.use_cuda())
        except Exception:
            astra_cuda = False
    available = torch_cuda and astra_available and astra_cuda
    return {
        "schema_version": "ct.fdk_capability.v1",
        "algorithm": "fdk",
        "observation_stratum": "fdk_backend",
        "geometry": "cone_3d",
        "status": "available" if available else "unavailable",
        "reason": None if available else "FDK requires PyTorch CUDA, astra-toolbox, and ASTRA CUDA",
        "checks": {
            "torch_cuda": torch_cuda,
            "astra_module": astra_available,
            "astra_cuda": astra_cuda,
            "astra_version": astra_version,
        },
        "run_performed": False,
        "required_capabilities": list(config["fdk"]["required_capabilities"]),
    }


def trial_overrides(name: str, parameter: str, value: float) -> dict[str, Any]:
    if parameter == "atol_btol":
        return {"atol": value, "btol": value}
    result: dict[str, Any] = {parameter: value}
    if name == "tv_fista":
        result["power_iterations"] = 2
    return result


def private_oracle_psnr(job_dir: Path) -> float:
    """Offline reference metric; never used by the normal tuning selector."""

    bundle = torch.load(job_dir / "reconstruction.pt", map_location="cpu", weights_only=True)
    reconstruction = bundle["reconstruction"].float()
    truth = bundle["truth"].float()
    data_range = float(bundle.get("data_range") or (truth.max() - truth.min()).clamp_min(1e-12))
    rmse = float(torch.mean((reconstruction - truth).square()).sqrt())
    return float(20.0 * math.log10(data_range / max(rmse, 1e-12)))


def protocol_history(
    config: Mapping[str, Any], protocols: Mapping[str, ComparisonProtocol], monitor: BudgetMonitor,
    runtime_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    study = config["protocol_study"]
    public_data = CT_ROOT / "test" / "data"
    case_id = str(study["case"])
    case = load_ct_case(case_id, data_root=public_data, verify_checksum=True)
    split = make_heldout_projection_split(case_id, case.geometry["angles_rad"], folds=3)
    case_digest = case_hash(public_data, case_id)
    algorithm_summaries: dict[str, Any] = {}
    logical_records: list[dict[str, Any]] = []
    global_budget = config["global_budget"]
    for algorithm, declaration in study["algorithms"].items():
        started = time.perf_counter()
        trial_rows: list[dict[str, Any]] = []
        total_forward = total_adjoint = 0
        for trial_index, value in enumerate(declaration["values"]):
            fold_rows: list[dict[str, Any]] = []
            overrides = trial_overrides(algorithm, declaration["parameter"], float(value))
            for fold in range(3):
                fit = split.training_indices(fold)
                heldout = split.validation_folds[fold]
                max_iterations = 1 if algorithm in {"sart", "os_sart"} else int(study["max_iterations_per_trial"])
                overrides["num_iterations"] = max_iterations
                job_dir = runtime_root / "jobs" / "protocol_history_v3" / algorithm / f"trial_{trial_index}" / f"fold_{fold}"
                result, resumed = run_atomic(
                    monitor=monitor,
                    job_dir=job_dir,
                    algorithm=algorithm, case_id=case_id, config_path=algorithm_config(algorithm),
                    data_root=public_data, max_iterations=max_iterations,
                    max_forward_calls=30, max_adjoint_calls=30,
                    parameter_overrides=overrides,
                    parameter_sources={name: "batch13_predeclared_candidate" for name in overrides},
                    fit_view_indices=fit, heldout_view_indices=heldout,
                    split_metadata={
                        "protocol_version": split.protocol_version,
                        "split_sha256": split.split_sha256,
                        "fold": fold,
                        "fit_view_indices": list(fit),
                        "heldout_view_indices": list(heldout),
                    },
                    fixed_compute=False,
                )
                resources = result["diagnostics"].get("resources", {})
                forward = int(resources.get("forward_calls") or 0)
                adjoint = int(resources.get("adjoint_calls") or 0)
                total_forward += forward
                total_adjoint += adjoint
                metrics = result["metrics"]
                fold_rows.append({
                    "fold": fold,
                    "status": metrics.get("status"),
                    "convergence_status": metrics.get("convergence_status"),
                    "heldout_residual": metrics.get("held_out_projection_residual"),
                    "private_oracle_psnr": private_oracle_psnr(job_dir),
                    "runtime_seconds": metrics.get("runtime_seconds"),
                    "forward_calls": forward,
                    "adjoint_calls": adjoint,
                    "resumed": resumed,
                })
            valid = [row for row in fold_rows if row["status"] == "success" and row["heldout_residual"] is not None]
            trial_rows.append({
                "trial_index": trial_index,
                "parameter_overrides": overrides,
                "folds": fold_rows,
                "completed_valid": len(valid) == 3,
                "mean_heldout_residual": (sum(float(row["heldout_residual"]) for row in valid) / 3) if len(valid) == 3 else None,
                "mean_psnr_private_oracle": (sum(float(row["private_oracle_psnr"]) for row in valid) / 3) if len(valid) == 3 else None,
            })
        runtime_seconds = time.perf_counter() - started
        completed_trials = sum(bool(row["completed_valid"]) for row in trial_rows)
        if completed_trials > int(global_budget["tuning_trials_per_algorithm"]):
            raise RuntimeError(f"{algorithm} exceeded the global tuning-trial budget")
        if runtime_seconds > float(global_budget["tuning_runtime_seconds_per_algorithm"]):
            raise RuntimeError(f"{algorithm} exceeded the global tuning-time budget")
        if total_forward > int(global_budget["tuning_forward_calls_per_algorithm"]):
            raise RuntimeError(f"{algorithm} exceeded the global tuning forward-call budget")
        if total_adjoint > int(global_budget["tuning_adjoint_calls_per_algorithm"]):
            raise RuntimeError(f"{algorithm} exceeded the global tuning adjoint-call budget")
        valid_trials = [row for row in trial_rows if row["completed_valid"]]
        bounded = min(valid_trials, key=lambda row: (row["mean_heldout_residual"], row["trial_index"])) if valid_trials else None
        oracle = max(valid_trials, key=lambda row: (row["mean_psnr_private_oracle"], -row["trial_index"])) if valid_trials else None
        views: dict[str, Any] = {}
        for protocol_id in study["protocols"]:
            protocol = protocols[protocol_id]
            declaration_protocol = study["protocols"][protocol_id]
            if protocol_id == "equal_operator_calls/v1":
                selected_rows = []
                used_forward = used_adjoint = 0
                for row in trial_rows:
                    fold = row["folds"][0]
                    next_forward = used_forward + int(fold["forward_calls"])
                    next_adjoint = used_adjoint + int(fold["adjoint_calls"])
                    if next_forward > 60 or next_adjoint > 60:
                        break
                    selected_rows.append(row["trial_index"])
                    used_forward, used_adjoint = next_forward, next_adjoint
            else:
                selected_rows = [row["trial_index"] for row in trial_rows]
                used_forward = total_forward if protocol_id == "common_validation/v1" else sum(int(row["folds"][0]["forward_calls"]) for row in trial_rows)
                used_adjoint = total_adjoint if protocol_id == "common_validation/v1" else sum(int(row["folds"][0]["adjoint_calls"]) for row in trial_rows)
            if len(selected_rows) > int(declaration_protocol["max_trials"]):
                raise RuntimeError(f"{algorithm}/{protocol_id} exceeded trial ceiling")
            if protocol_id == "equal_tuning_time/v1" and runtime_seconds > float(declaration_protocol["max_seconds"]):
                raise RuntimeError(f"{algorithm}/{protocol_id} exceeded time ceiling")
            views[protocol_id] = {
                "trial_indices": selected_rows,
                "completed_trials": len(selected_rows),
                "runtime_seconds": runtime_seconds,
                "forward_calls": used_forward,
                "adjoint_calls": used_adjoint,
                "protocol_digest": protocol.digest,
                "budget": protocol.budget.to_dict(),
                "history_reused_without_extra_solver_calls": True,
            }
            for row in trial_rows:
                if row["trial_index"] not in selected_rows:
                    continue
                fold = row["folds"][0]
                logical = BenchmarkResult(
                    algorithm=algorithm, solver=algorithm, case_id=case_id,
                    geometry="parallel_2d", observation_domain="line_integral",
                    observation_stratum="transmission", parameters=row["parameter_overrides"],
                    parameter_source="batch13_predeclared_candidate",
                    tuning_protocol=protocol_id,
                    convergence_status=fold["convergence_status"],
                    runtime_seconds=fold["runtime_seconds"],
                    forward_calls=fold["forward_calls"], adjoint_calls=fold["adjoint_calls"],
                    psnr=None, residual=fold["heldout_residual"],
                    status=fold["status"], budget=protocol.budget.to_dict(),
                    context_digest=canonical_digest({"case": case_id, "split": split.split_sha256, "fold": 0}),
                    protocol_digest=protocol.digest,
                    tuning_usage={
                        "completed_trials": 1,
                        "runtime_seconds": fold["runtime_seconds"],
                        "forward_calls": fold["forward_calls"],
                        "adjoint_calls": fold["adjoint_calls"],
                    },
                ).to_dict()
                logical.update({"trial_index": row["trial_index"], "split_sha256": split.split_sha256})
                logical_records.append(logical)
        algorithm_summaries[algorithm] = {
            "candidate_parameter": declaration["parameter"],
            "trials": trial_rows,
            "unique_completed_trials": completed_trials,
            "unique_runtime_seconds": runtime_seconds,
            "unique_forward_calls": total_forward,
            "unique_adjoint_calls": total_adjoint,
            "bounded_selection": bounded,
            "oracle_selection": oracle,
            "protocol_views": views,
        }
    return {
        "schema_version": SCHEMA,
        "case_id": case_id,
        "case_sha256": case_digest,
        "split_protocol": split.protocol_version,
        "split_sha256": split.split_sha256,
        "fold_count": split.fold_count,
        "algorithms": algorithm_summaries,
    }, logical_records


def recommendation_plans(
    config: Mapping[str, Any], history: Mapping[str, Any], main_sha: str, ct_sha: str,
) -> dict[str, Any]:
    from inverse_agent.backends.ct import CTBackend
    from inverse_agent.orchestration.ct_selection import CTTaskProfile, recommend_ct_parameter_plan

    backend = CTBackend(CT_ROOT)
    profile = CTTaskProfile(
        geometry_type="parallel_2d", dimension=2, num_views=16,
        observation_domain="line_integral", noise_model="gaussian_relative",
        noise_level=0.01, image_shape=(32, 32), device="cpu",
    )
    output: dict[str, Any] = {}
    for algorithm in config["protocol_study"]["algorithms"]:
        plan = recommend_ct_parameter_plan(
            backend.describe_algorithm(algorithm), profile, seed=17,
            main_commit=main_sha, ct_commit=ct_sha,
        )
        output[algorithm] = json.loads(plan.canonical_json())
    return output


def parameter_study(
    config: Mapping[str, Any], fixed: Sequence[Mapping[str, Any]], history: Mapping[str, Any],
    main_sha: str, ct_sha: str,
) -> dict[str, Any]:
    recommendations = recommendation_plans(config, history, main_sha, ct_sha)
    case_id = config["protocol_study"]["case"]
    fixed_by_algorithm = {
        row["algorithm"]: {"parameters": row["parameters"], "case_id": row["case_id"]}
        for row in fixed if row["case_id"] == case_id
    }
    algorithms: dict[str, Any] = {}
    oracle_records: list[dict[str, Any]] = []
    for algorithm, evidence in history["algorithms"].items():
        oracle = evidence["oracle_selection"]
        bounded = evidence["bounded_selection"]
        oracle_record = {
            "algorithm": algorithm,
            "protocol": "oracle_upper_bound/v1",
            "agent_available": False,
            "include_in_normal_ranking": False,
            "truth_used": True,
            "selected_trial_index": None if oracle is None else oracle["trial_index"],
            "selected_parameters": None if oracle is None else oracle["parameter_overrides"],
            "private_mean_psnr": None if oracle is None else oracle["mean_psnr_private_oracle"],
        }
        oracle_records.append(oracle_record)
        algorithms[algorithm] = {
            "fixed_defaults": fixed_by_algorithm.get(algorithm),
            "metadata_recommendation": recommendations[algorithm],
            "bounded_tuning": {
                "protocol": "common_validation/v1",
                "truth_used": False,
                "selected_trial_index": None if bounded is None else bounded["trial_index"],
                "selected_parameters": None if bounded is None else bounded["parameter_overrides"],
                "mean_heldout_residual": None if bounded is None else bounded["mean_heldout_residual"],
            },
            "oracle_upper_bound": oracle_record,
        }
    return {
        "schema_version": SCHEMA,
        "modes": list(config["parameter_study"]["modes"]),
        "normal_selection_metric": "mean_heldout_projection_residual",
        "oracle_selection_metric": "private_mean_psnr",
        "oracle_excluded_from_agent_and_normal_ranking": True,
        "algorithms": algorithms,
        "oracle_records": oracle_records,
    }


def robustness_matrix(
    config: Mapping[str, Any], protocols: Mapping[str, ComparisonProtocol], monitor: BudgetMonitor,
    runtime_root: Path, data_root: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    protocol = protocols["fixed_defaults/v1"]
    declaration = config["robustness"]
    for seed in config["seeds"]:
        for noise in declaration["noise_levels"]:
            for views in declaration["view_counts"]:
                for coverage in declaration["angle_coverages_deg"]:
                    case_id = f"batch13_transmission/s{seed}_n{noise:g}_v{views}_a{coverage:g}_32"
                    axes = {"seed": seed, "noise_level": noise, "view_count": views, "angle_coverage_deg": coverage}
                    digest = case_hash(data_root, case_id)
                    for algorithm in declaration["transmission_algorithms"]:
                        result, _ = run_atomic(
                            monitor=monitor,
                            job_dir=runtime_root / "jobs" / "robustness" / "transmission" / case_id.replace("/", "__") / algorithm,
                            algorithm=algorithm, case_id=case_id, config_path=algorithm_config(algorithm),
                            data_root=data_root,
                        )
                        records.append(compact_record(
                            result, algorithm=algorithm, case_id=case_id, stratum="transmission",
                            protocol=protocol, case_hash=digest, robustness=axes,
                        ))
        for count in declaration["count_levels"]:
            for views in declaration["view_counts"]:
                for coverage in declaration["angle_coverages_deg"]:
                    case_id = f"batch13_count/s{seed}_c{count:g}_v{views}_a{coverage:g}_32"
                    axes = {"seed": seed, "count_level": count, "view_count": views, "angle_coverage_deg": coverage}
                    digest = case_hash(data_root, case_id)
                    for algorithm in declaration["count_algorithms"]:
                        result, _ = run_atomic(
                            monitor=monitor,
                            job_dir=runtime_root / "jobs" / "robustness" / "count" / case_id.replace("/", "__") / algorithm,
                            algorithm=algorithm, case_id=case_id, config_path=algorithm_config(algorithm),
                            data_root=data_root,
                        )
                        records.append(compact_record(
                            result, algorithm=algorithm, case_id=case_id, stratum="emission_count",
                            protocol=protocol, case_hash=digest, robustness=axes,
                        ))
    return records


def fairness_reports(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in records:
        key = (str(row["case_id"]), str(row["observation_stratum"]), str(row["tuning_protocol"]))
        groups.setdefault(key, []).append(row)
    reports = {
        "|".join(key): check_fairness(rows)
        for key, rows in sorted(groups.items())
    }
    if not all(report["fair"] for report in reports.values()):
        failures = {key: value for key, value in reports.items() if not value["fair"]}
        raise RuntimeError(f"fairness validation failed: {failures}")
    return reports


def validate_schema(records: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    for row in records:
        restored = BenchmarkResult.from_mapping(row)
        payload = restored.to_dict()
        if payload["schema_version"] != "ct.benchmark_record.v2":
            raise ValueError("benchmark record schema mismatch")
        if set(payload["axes"]) != {
            "reconstruction_quality", "data_consistency", "optimization_behavior",
            "computational_efficiency", "robustness",
        }:
            raise ValueError("benchmark axes are incomplete")
        count += 1
    return count


def environment_record(config_path: Path, protocols: Mapping[str, ComparisonProtocol]) -> dict[str, Any]:
    return {
        "schema_version": "ct.batch13_environment.v1",
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "yaml": yaml.__version__,
        "cuda_available": torch.cuda.is_available(),
        "astra_module_available": importlib.util.find_spec("astra") is not None,
        "device": "cpu",
        "dtype": "float32",
        "main_git_sha": git_revision(MAIN_ROOT),
        "ct_git_sha": git_revision(CT_ROOT),
        "config_sha256": sha256_file(config_path),
        "protocol_digests": {name: value.digest for name, value in protocols.items()},
    }


def estimate(config: Mapping[str, Any]) -> dict[str, Any]:
    fixed_jobs = 9 * 3 + 2
    robustness_jobs = 9 * 16 + 2 * 16
    tuning_jobs = len(config["protocol_study"]["algorithms"]) * 4 * 3
    total_jobs = fixed_jobs + robustness_jobs + tuning_jobs
    return {
        "schema_version": "ct.batch13_cost_estimate.v1",
        "fixed_jobs": fixed_jobs,
        "robustness_jobs": robustness_jobs,
        "unique_tuning_jobs": tuning_jobs,
        "total_numerical_jobs": total_jobs,
        "conservative_seconds_per_job": 10,
        "estimated_wall_seconds": total_jobs * 10,
        "conservative_bytes_per_job": 512 * 1024,
        "estimated_result_bytes": total_jobs * 512 * 1024,
        "estimated_peak_memory_bytes": 1024**3,
        "within_budget": (
            total_jobs * 10 < int(config["global_budget"]["wall_seconds"])
            and total_jobs * 512 * 1024 < int(config["global_budget"]["result_bytes"])
            and 1024**3 < int(config["global_budget"]["peak_memory_bytes"])
        ),
    }


def write_checksums(output_root: Path) -> dict[str, str]:
    files = sorted(
        path for path in output_root.rglob("*")
        if path.is_file() and path.name != "checksums.json"
    )
    checksums = {path.relative_to(output_root).as_posix(): sha256_file(path) for path in files}
    atomic_json(output_root / "checksums.json", checksums)
    return checksums


def validate_checksums(output_root: Path) -> int:
    checksums = json.loads((output_root / "checksums.json").read_text(encoding="utf-8"))
    for relative, expected in checksums.items():
        path = output_root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"checksum mismatch: {relative}")
    return len(checksums)


def validate_evidence(output_root: Path) -> dict[str, Any]:
    validate_checksums(output_root)
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    fixed = json.loads((output_root / "fixed_defaults.json").read_text(encoding="utf-8"))["records"]
    robustness = json.loads((output_root / "robustness.json").read_text(encoding="utf-8"))["records"]
    protocol = json.loads((output_root / "protocol_study.json").read_text(encoding="utf-8"))
    parameter = json.loads((output_root / "parameter_study.json").read_text(encoding="utf-8"))
    if len(fixed) != 29 or len(robustness) != 176:
        raise ValueError("planned fixed-default or robustness record count changed")
    if summary["fdk"]["status"] not in {"available", "unavailable"}:
        raise ValueError("FDK capability outcome is missing")
    if not all(status_is_accepted(row) for row in [*fixed, *robustness]):
        raise ValueError("a numerical benchmark record has an unaccepted outcome")
    schema_count = validate_schema([*fixed, *robustness])
    if not parameter["oracle_excluded_from_agent_and_normal_ranking"]:
        raise ValueError("oracle reference was not excluded")
    if "total_score" in canonical_bytes(summary).decode("utf-8"):
        raise ValueError("aggregate total score is forbidden")
    budgets = manifest["observed_budget"]
    limits = manifest["budget_limits"]
    if budgets["wall_seconds"] > limits["wall_seconds"]:
        raise ValueError("wall budget exceeded")
    if budgets["result_bytes"] > limits["result_bytes"]:
        raise ValueError("result budget exceeded")
    if budgets["peak_working_set_bytes"] is not None and budgets["peak_working_set_bytes"] > limits["peak_memory_bytes"]:
        raise ValueError("memory budget exceeded")
    for algorithm, evidence in protocol["algorithms"].items():
        if evidence["unique_completed_trials"] > limits["tuning_trials_per_algorithm"]:
            raise ValueError(f"{algorithm} tuning trial budget exceeded")
        if evidence["unique_runtime_seconds"] > limits["tuning_runtime_seconds_per_algorithm"]:
            raise ValueError(f"{algorithm} tuning time budget exceeded")
        if evidence["unique_forward_calls"] > limits["tuning_forward_calls_per_algorithm"]:
            raise ValueError(f"{algorithm} tuning forward-call budget exceeded")
        if evidence["unique_adjoint_calls"] > limits["tuning_adjoint_calls_per_algorithm"]:
            raise ValueError(f"{algorithm} tuning adjoint-call budget exceeded")
    return {
        "schema_records_validated": schema_count,
        "checksum_files_validated": validate_checksums(output_root),
        "fairness_groups": len(summary["fairness"]),
        "fixed_records": len(fixed),
        "robustness_records": len(robustness),
        "tuning_algorithms": len(protocol["algorithms"]),
        "oracle_records": len(parameter["oracle_records"]),
    }


def run(config_path: Path, output_root: Path) -> dict[str, Any]:
    config = load_config(config_path)
    protocols = load_protocols()
    cost = estimate(config)
    if not cost["within_budget"]:
        raise RuntimeError("conservative Batch 13 estimate exceeds the frozen budget")
    runtime_root = CT_ROOT / str(config["runtime_output"]).split("/")[0]
    runtime_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    elapsed_offset = 0.0
    previous_manifest = output_root / "manifest.json"
    if previous_manifest.is_file():
        previous = json.loads(previous_manifest.read_text(encoding="utf-8"))
        elapsed_offset = float(previous.get("observed_budget", {}).get("wall_seconds") or 0.0)
    monitor = BudgetMonitor(
        started=started, runtime_root=runtime_root, limits=config["global_budget"],
        elapsed_offset=elapsed_offset,
    )
    generated_data = ensure_runtime_cases(config, runtime_root)
    environment = environment_record(config_path, protocols)
    fixed = fixed_default_matrix(config, protocols, monitor, runtime_root, generated_data)
    fdk = fdk_capability_record(config)
    if fdk["status"] == "available":
        raise RuntimeError("FDK capability changed; this frozen CPU benchmark has no generated cone case")
    history, logical_protocol_records = protocol_history(config, protocols, monitor, runtime_root)
    parameters = parameter_study(
        config, fixed, history, environment["main_git_sha"], environment["ct_git_sha"]
    )
    robustness = robustness_matrix(config, protocols, monitor, runtime_root, generated_data)
    all_normal_records = [*fixed, *robustness, *logical_protocol_records]
    fairness = fairness_reports(all_normal_records)
    schema_count = validate_schema(all_normal_records)
    observed = monitor.snapshot()
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(output_root / "environment.json", environment)
    atomic_json(output_root / "fixed_defaults.json", {"schema_version": SCHEMA, "records": fixed})
    atomic_json(output_root / "protocol_study.json", history)
    atomic_json(output_root / "parameter_study.json", parameters)
    atomic_json(output_root / "robustness.json", {"schema_version": SCHEMA, "records": robustness})
    summary = {
        "schema_version": SCHEMA,
        "passed": True,
        "fixed_defaults": {
            "transmission_records": sum(row["observation_stratum"] == "transmission" for row in fixed),
            "count_records": sum(row["observation_stratum"] == "emission_count" for row in fixed),
            "accepted_records": sum(status_is_accepted(row) for row in fixed),
        },
        "fdk": fdk,
        "protocol_study": {
            "algorithms": len(history["algorithms"]),
            "unique_completed_trials": sum(row["unique_completed_trials"] for row in history["algorithms"].values()),
            "history_reused_across_protocol_views": True,
        },
        "parameter_study": {
            "modes": parameters["modes"],
            "oracle_excluded": True,
        },
        "robustness": {
            "transmission_records": sum(row["observation_stratum"] == "transmission" for row in robustness),
            "count_records": sum(row["observation_stratum"] == "emission_count" for row in robustness),
            "accepted_records": sum(status_is_accepted(row) for row in robustness),
        },
        "fairness": fairness,
        "schema_records_validated": schema_count,
        "axes": [
            "reconstruction_quality", "data_consistency", "optimization_behavior",
            "computational_efficiency", "robustness",
        ],
        "aggregate_score": None,
    }
    atomic_json(output_root / "summary.json", summary)
    manifest = {
        "schema_version": SCHEMA,
        "config": str(config_path.relative_to(CT_ROOT)).replace("\\", "/"),
        "config_sha256": sha256_file(config_path),
        "environment": "environment.json",
        "cost_estimate": cost,
        "budget_limits": config["global_budget"],
        "observed_budget": observed,
        "raw_runtime_root": str(runtime_root.relative_to(CT_ROOT)).replace("\\", "/"),
        "raw_runtime_ignored": True,
        "atomic_state_model": ["pending", "temporary", "finalized"],
        "resume_key_fields": ["case_id", "algorithm", "protocol", "seed", "config_sha256"],
        "fixed_record_count": len(fixed),
        "robustness_record_count": len(robustness),
        "logical_protocol_record_count": len(logical_protocol_records),
        "oracle_record_count": len(parameters["oracle_records"]),
        "fdk_status": fdk["status"],
        "passed": True,
    }
    atomic_json(output_root / "manifest.json", manifest)
    report = (
        "# Batch 13 budgeted ordinary-CT benchmark\n\n"
        f"- Fixed defaults: 27 transmission + 2 count-domain records; {sum(status_is_accepted(row) for row in fixed)}/29 accepted.\n"
        f"- FDK: `{fdk['status']}` ({fdk['reason']}).\n"
        f"- Protocol study: {len(history['algorithms'])} tunable algorithms, four unique candidates each; one 3-fold history reused by four accepted protocol views.\n"
        f"- Robustness: 144 transmission + 32 count-domain records; {sum(status_is_accepted(row) for row in robustness)}/176 accepted.\n"
        "- Parameter modes: fixed defaults, metadata recommendation, bounded held-out tuning, and separately tagged oracle upper bound.\n"
        "- Metrics remain on five independent axes; no aggregate score is produced.\n"
        f"- Observed wall: {observed['wall_seconds']:.3f}s; peak working set: {observed['peak_working_set_bytes']} bytes; ignored runtime size: {observed['result_bytes']} bytes.\n"
    )
    atomic_text(output_root / "REPORT.md", report)
    write_checksums(output_root)
    validation = validate_evidence(output_root)
    return {"output_root": str(output_root), "observed_budget": observed, "validation": validation}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=CT_ROOT / "configs" / "benchmarks" / "batch13_budgeted_32.yaml",
    )
    parser.add_argument(
        "--output", type=Path,
        default=CT_ROOT / "artifacts" / "ct_agent_trustworthy_v1",
    )
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    output_root = args.output.resolve()
    config = load_config(config_path)
    if args.estimate_only:
        print(json.dumps(estimate(config), indent=2, sort_keys=True))
        return
    if args.validate_only:
        print(json.dumps(validate_evidence(output_root), indent=2, sort_keys=True))
        return
    print(json.dumps(run(config_path, output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

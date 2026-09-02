"""Deterministic Batch 12 ordinary-CT status-matrix runner.

This module is deliberately test-only.  Fault trajectories used to exercise
stall/divergence classification never enter solver or runtime production code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch

# A direct ``python test/batch12_synthetic_matrix.py`` invocation otherwise
# resolves any globally installed package before this checkout.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from inv_framework.convergence import classify_trajectory
from inv_framework.operators.base import LinearOperator
from inv_framework.regularizers import TVRegularizer
from inv_framework.solvers import (
    CGLSSolver,
    LandweberSolver,
    LSQRSolver,
    MLEMSolver,
    OSEMSolver,
    OSSARTSolver,
    SARTSolver,
    SIRTSolver,
    SolveControl,
    TikhonovSolver,
    TVFISTASolver,
)
from inv_framework.solvers.detailed import solve_fbp_detailed, solve_fdk_detailed
from inv_framework.solvers.specs import CANONICAL_ALGORITHM_IDS, validate_parameter_values


SCHEMA_VERSION = "ct.batch12_synthetic_evidence.v1"
ITERATIVE_ALGORITHMS = tuple(
    name for name in CANONICAL_ALGORITHM_IDS if name not in {"fbp", "fdk"}
)
DIRECT_ALGORITHMS = ("fbp", "fdk")


class DiagonalAngleOperator(LinearOperator):
    """Small deterministic operator with an exact adjoint and subset support."""

    def __init__(self, size: int, *, dtype: torch.dtype = torch.float64):
        self.domain_shape = (1, int(size), int(size))
        self.range_shape = (1, int(size), int(size))
        self.diagonal = torch.linspace(0.75, 1.25, int(size), dtype=dtype)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        weights = self.diagonal.to(value).reshape(1, 1, -1, 1)
        return value * weights

    def adjoint(self, value: torch.Tensor) -> torch.Tensor:
        weights = self.diagonal.to(value).reshape(1, 1, -1, 1)
        return value * weights

    def subset(self, indices: torch.Tensor) -> LinearOperator:
        return DiagonalAngleSubset(self.diagonal, indices)


class DiagonalAngleSubset(LinearOperator):
    def __init__(self, diagonal: torch.Tensor, indices: torch.Tensor):
        self.diagonal = diagonal
        self.indices = indices.to(dtype=torch.long)
        size = int(diagonal.numel())
        self.domain_shape = (1, size, size)
        self.range_shape = (1, int(self.indices.numel()), size)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        selected = value.index_select(2, self.indices)
        weights = self.diagonal.index_select(0, self.indices).to(value).reshape(1, 1, -1, 1)
        return selected * weights

    def adjoint(self, value: torch.Tensor) -> torch.Tensor:
        result = torch.zeros(
            (value.shape[0], *self.domain_shape), dtype=value.dtype, device=value.device
        )
        weights = self.diagonal.index_select(0, self.indices).to(value).reshape(1, 1, -1, 1)
        return result.index_add(2, self.indices, value * weights)


class FDKTestOperator(DiagonalAngleOperator):
    def __init__(self, size: int, *, mode: str = "valid"):
        super().__init__(size)
        self.mode = mode

    def fdk(self, value: torch.Tensor, **_: Any) -> torch.Tensor:
        if self.mode == "nonfinite":
            return torch.full(
                (value.shape[0], *self.domain_shape),
                float("nan"),
                dtype=value.dtype,
                device=value.device,
            )
        return self.adjoint(value)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _tensor_sha256(value: torch.Tensor) -> str:
    payload = value.detach().cpu().contiguous().numpy().tobytes(order="C")
    return _sha256_bytes(payload)


def _solver(name: str, iterations: int):
    constructors = {
        "sirt": lambda: SIRTSolver(num_iterations=iterations),
        "landweber": lambda: LandweberSolver(num_iterations=iterations, step_size=0.25),
        "cgls": lambda: CGLSSolver(num_iterations=iterations, tol=0.0),
        "lsqr": lambda: LSQRSolver(num_iterations=iterations, atol=0.0, btol=0.0),
        "sart": lambda: SARTSolver(num_iterations=iterations, block_size=4),
        "os_sart": lambda: OSSARTSolver(num_iterations=iterations, block_size=4),
        "mlem": lambda: MLEMSolver(num_iterations=iterations),
        "osem": lambda: OSEMSolver(num_iterations=iterations, block_size=4),
        "tikhonov": lambda: TikhonovSolver(
            reg_strength=0.0, num_iterations=iterations, tolerance=0.0
        ),
        "tv_fista": lambda: TVFISTASolver(
            reg_strength=0.01,
            num_iterations=iterations,
            step_size=0.25,
            tolerance=0.0,
            regularizer=TVRegularizer(num_iterations=2, tolerance=0.0),
        ),
    }
    return constructors[name]()


def _control(*, iterations: int, convergent: bool) -> SolveControl:
    if not convergent:
        return SolveControl(max_iterations=iterations, tolerance=0.0, stop_on_convergence=False)
    return SolveControl(
        max_iterations=iterations,
        tolerance=1e-10,
        min_iterations=1,
        patience=2,
        discrepancy_target=1e-10,
        relative_iterate_tolerance=1e-10,
        normalized_normal_residual_tolerance=1e-10,
        relative_objective_tolerance=1e-10,
        prox_gradient_mapping_tolerance=1e-10,
    )


def _run_solver(
    name: str,
    operator: DiagonalAngleOperator,
    measurement: torch.Tensor,
    *,
    iterations: int,
    x_init: torch.Tensor | None,
    convergent: bool,
):
    control = _control(iterations=iterations, convergent=convergent)
    if convergent and name in {"mlem", "osem"}:
        control = SolveControl(
            max_iterations=iterations,
            min_iterations=1,
            patience=2,
            relative_iterate_tolerance=1e-10,
            relative_objective_tolerance=1e-10,
        )
    return _solver(name, iterations).solve_detailed(
        measurement,
        operator,
        x_init=x_init,
        control=control,
    )


def _record(
    scenario: str,
    expected: str,
    actual: str,
    *,
    mode: str,
    reason: str,
    finite: bool | None = None,
    trajectory_points: int | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if actual != expected:
        raise AssertionError(f"{scenario}: expected {expected}, got {actual} ({reason})")
    payload: dict[str, Any] = {
        "scenario": scenario,
        "expected_status": expected,
        "actual_status": actual,
        "passed": True,
        "mode": mode,
        "reason": reason,
    }
    if finite is not None:
        payload["finite_reconstruction"] = bool(finite)
    if trajectory_points is not None:
        payload["trajectory_points"] = int(trajectory_points)
    if details:
        payload["details"] = details
    return payload


def _classified_status(name: str, scenario: str) -> tuple[str, str, dict[str, Any]]:
    if scenario == "stalled":
        rows = [
            {"iteration": index, "residual": 1.0, "relative_iterate_change": 1e-12}
            for index in range(1, 7)
        ]
        report = classify_trajectory(
            rows, tolerance=1e-8, patience=5, max_iterations=10, algorithm=name
        )
    elif scenario == "diverged":
        rows = [
            {"iteration": index, "residual": float(2**index), "relative_iterate_change": 0.5}
            for index in range(1, 7)
        ]
        report = classify_trajectory(rows, tolerance=1e-8, patience=3, algorithm=name)
    else:
        raise ValueError(scenario)
    return report.status.value, report.stopping_reason, {
        "fault_fixture": scenario,
        "trajectory_points": len(rows),
        "algorithm_label": name,
    }


def _iterative_records(
    name: str, *, size: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    torch.manual_seed(seed)
    operator = DiagonalAngleOperator(size)
    # A unit-sensitivity emission operator makes the exact positive image an
    # EM fixed point.  Transmission solvers retain the non-uniform diagonal.
    if name in {"mlem", "osem"}:
        operator.diagonal = torch.ones_like(operator.diagonal)
    truth = torch.ones((1, *operator.domain_shape), dtype=torch.float64)
    measurement = operator.forward(truth)
    data = {
        "size": size,
        "seed": seed,
        "measurement_sha256": _tensor_sha256(measurement),
        "operator_diagonal_sha256": _tensor_sha256(operator.diagonal),
        "observation_domain": "nonnegative_counts" if name in {"mlem", "osem"} else "line_integral",
        "initialization": "exact_constant_truth_for_normal; zeros_for_budget",
        "iteration_budget": 8,
    }
    records: list[dict[str, Any]] = []

    normal = _run_solver(
        name, operator, measurement, iterations=8, x_init=truth.clone(), convergent=True
    )
    records.append(
        _record(
            "normal",
            "converged",
            normal.status,
            mode="solver_invocation",
            reason=normal.stopping_reason,
            finite=bool(torch.isfinite(normal.reconstruction).all()),
            trajectory_points=len(normal.trajectory),
        )
    )

    budget = _run_solver(
        name,
        operator,
        measurement,
        iterations=1,
        x_init=torch.zeros_like(truth),
        convergent=False,
    )
    records.append(
        _record(
            "max_iterations",
            "max_iterations",
            budget.status,
            mode="solver_invocation",
            reason=budget.stopping_reason,
            finite=bool(torch.isfinite(budget.reconstruction).all()),
            trajectory_points=len(budget.trajectory),
        )
    )

    for scenario in ("stalled", "diverged"):
        status, reason, details = _classified_status(name, scenario)
        records.append(
            _record(
                scenario,
                scenario,
                status,
                mode="test_only_fault_trajectory",
                reason=reason,
                trajectory_points=details["trajectory_points"],
                details=details,
            )
        )

    nonfinite_measurement = measurement.clone()
    nonfinite_measurement[..., 0, 0] = float("nan")
    nonfinite = _run_solver(
        name,
        operator,
        nonfinite_measurement,
        iterations=2,
        x_init=truth.clone(),
        convergent=False,
    )
    records.append(
        _record(
            "nonfinite",
            "numerical_error",
            nonfinite.status,
            mode="solver_invocation",
            reason=nonfinite.stopping_reason,
            finite=bool(torch.isfinite(nonfinite.reconstruction).all()),
            trajectory_points=len(nonfinite.trajectory),
        )
    )

    validation = validate_parameter_values(
        name,
        {"num_iterations": 0},
        views=size,
        dimension=2,
        image_shape=(size, size) if name == "fdk" else None,
        observation_domain=data["observation_domain"],
        observation_model="poisson_emission" if name in {"mlem", "osem"} else "xray_transmission",
    )
    invalid_status = "invalid_parameters" if not validation.valid else "unexpected_valid"
    records.append(
        _record(
            "invalid_parameters",
            "invalid_parameters",
            invalid_status,
            mode="registry_validation",
            reason=validation.reason_codes[0] if validation.reason_codes else "validation_failed",
            details={"reason_codes": list(validation.reason_codes)},
        )
    )

    tolerance_rows = [
        {"iteration": 1, "residual": 1.0, "relative_iterate_change": 0.2},
        {"iteration": 2, "residual": 1e-4, "relative_iterate_change": 1e-5},
    ]
    loose = classify_trajectory(
        tolerance_rows, tolerance=1e-3, max_iterations=3, algorithm=name
    )
    strict = classify_trajectory(
        tolerance_rows, tolerance=1e-8, max_iterations=2, algorithm=name
    )
    tolerance_actual = (
        "tolerance_sensitive"
        if loose.status.value == "converged" and strict.status.value == "max_iterations"
        else "inconsistent"
    )
    records.append(
        _record(
            "tolerance_sensitivity",
            "tolerance_sensitive",
            tolerance_actual,
            mode="status_classifier",
            reason="loose_converged_strict_budget_limited",
            details={"loose": loose.status.value, "strict": strict.status.value},
        )
    )

    trajectory_consistent = bool(
        normal.trajectory
        and normal.trajectory[-1].iteration == normal.actual_iterations
        and all(row.finite for row in normal.trajectory)
        and budget.trajectory
        and budget.trajectory[-1].iteration == budget.actual_iterations
        and budget.status != "converged"
    )
    records.append(
        _record(
            "trajectory_consistency",
            "consistent",
            "consistent" if trajectory_consistent else "inconsistent",
            mode="solver_evidence_assertion",
            reason="terminal_iteration_and_finite_flags_agree",
            details={
                "normal_iterations": normal.actual_iterations,
                "normal_terminal_iteration": normal.trajectory[-1].iteration if normal.trajectory else None,
                "budget_iterations": budget.actual_iterations,
                "budget_terminal_iteration": budget.trajectory[-1].iteration if budget.trajectory else None,
            },
        )
    )
    return records, data


def _direct_records(
    name: str, *, size: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    torch.manual_seed(seed)
    operator: DiagonalAngleOperator = (
        FDKTestOperator(size) if name == "fdk" else DiagonalAngleOperator(size)
    )
    truth = torch.ones((1, *operator.domain_shape), dtype=torch.float64)
    measurement = operator.forward(truth)
    solve = solve_fdk_detailed if name == "fdk" else solve_fbp_detailed
    valid = solve(operator, measurement)
    nonfinite = (
        solve(FDKTestOperator(size, mode="nonfinite"), measurement)
        if name == "fdk"
        else solve(operator, torch.full_like(measurement, float("nan")))
    )
    invalid = (
        solve(operator, measurement, voxel_supersampling=0)
        if name == "fdk"
        else solve(operator, measurement, scale=0.0)
    )
    if name == "fdk":
        unavailable = solve(DiagonalAngleOperator(size), measurement)
        unavailable_status = unavailable.status
        unavailable_reason = unavailable.stopping_reason
    else:
        unavailable_status = "not_applicable"
        unavailable_reason = "fbp_has_no_optional_external_backend"
    records = [
        _record(
            "valid", "completed_valid", valid.status,
            mode="solver_invocation", reason=valid.stopping_reason,
            finite=bool(torch.isfinite(valid.reconstruction).all()), trajectory_points=0,
        ),
        _record(
            "nonfinite", "numerical_error", nonfinite.status,
            mode="solver_invocation", reason=nonfinite.stopping_reason,
            finite=bool(torch.isfinite(nonfinite.reconstruction).all()), trajectory_points=0,
        ),
        _record(
            "invalid_parameters", "invalid_parameters", invalid.status,
            mode="solver_invocation", reason=invalid.stopping_reason, trajectory_points=0,
        ),
        _record(
            "unavailable",
            "unavailable" if name == "fdk" else "not_applicable",
            unavailable_status,
            mode="solver_invocation" if name == "fdk" else "capability_assertion",
            reason=unavailable_reason,
            trajectory_points=0,
        ),
    ]
    for scenario in (
        "max_iterations", "stalled", "diverged", "tolerance_sensitivity", "trajectory_consistency"
    ):
        records.append(
            _record(
                scenario,
                "not_applicable",
                "not_applicable",
                mode="direct_algorithm_applicability_assertion",
                reason="direct_algorithm_has_no_iteration_trajectory",
                trajectory_points=0,
            )
        )
    return records, {
        "size": size,
        "seed": seed,
        "measurement_sha256": _tensor_sha256(measurement),
        "operator_diagonal_sha256": _tensor_sha256(operator.diagonal),
        "observation_domain": "line_integral",
        "initialization": "not_applicable_direct",
        "iteration_budget": 0,
    }


def run_matrix(protocol_path: Path, output_dir: Path) -> dict[str, Any]:
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    if protocol.get("schema_version") != "ct.batch12_synthetic_protocol.v1":
        raise ValueError("unsupported Batch 12 synthetic protocol schema")
    seeds = tuple(int(value) for value in protocol["seeds"])
    sizes = tuple(int(value) for value in protocol["sizes"])
    if seeds != (17, 29, 43) or sizes != (16, 32):
        raise ValueError("Batch 12 protocol must retain seeds 17/29/43 and sizes 16/32")

    algorithms: dict[str, Any] = {}
    for index, name in enumerate(CANONICAL_ALGORITHM_IDS):
        size = sizes[index % len(sizes)]
        seed = seeds[index % len(seeds)]
        if name in DIRECT_ALGORITHMS:
            records, data = _direct_records(name, size=size, seed=seed)
            kind = "direct"
        else:
            records, data = _iterative_records(name, size=size, seed=seed)
            kind = "iterative"
        algorithms[name] = {"kind": kind, "data": data, "scenarios": records}

    scenario_count = sum(len(value["scenarios"]) for value in algorithms.values())
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "protocol_sha256": _sha256_bytes(protocol_bytes),
        "repository_base_sha": "2ab4f19b7fc74f44d1d414757eb83ff60e154efc",
        "ordinary_ct_only": True,
        "seeds": list(seeds),
        "sizes": list(sizes),
        "algorithm_order": list(CANONICAL_ALGORITHM_IDS),
        "algorithms": algorithms,
        "summary": {
            "algorithm_count": len(algorithms),
            "iterative_algorithm_count": len(ITERATIVE_ALGORITHMS),
            "direct_algorithm_count": len(DIRECT_ALGORITHMS),
            "scenario_count": scenario_count,
            "passed_scenario_count": sum(
                int(record["passed"])
                for value in algorithms.values()
                for record in value["scenarios"]
            ),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.json"
    result_bytes = _canonical_bytes(evidence)
    result_path.write_bytes(result_bytes)
    checksums = {
        "schema_version": "ct.batch12_checksums.v1",
        "files": {
            "protocol.json": _sha256_bytes(protocol_bytes),
            "results.json": _sha256_bytes(result_bytes),
        },
    }
    (output_dir / "checksums.json").write_bytes(_canonical_bytes(checksums))
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_protocol = Path(__file__).parent / "data" / "batch12_synthetic" / "protocol.json"
    parser.add_argument("--protocol", type=Path, default=default_protocol)
    parser.add_argument("--output", type=Path, default=default_protocol.parent)
    args = parser.parse_args(argv)
    evidence = run_matrix(args.protocol, args.output)
    print(json.dumps(evidence["summary"], sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

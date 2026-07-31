from __future__ import annotations

import torch
import pytest

from inv_framework.benchmarks import load_ct_case
from inv_framework.operators.ct import ParallelBeamRadon2D
from inv_framework.solvers.classical import FBPSolver

from .benchmark import QUALITY_CASE_IDS, write_quality_artifacts


def _operator(case):
    device = case.truth.device
    angles = torch.tensor(case.geometry["angles_rad"], device=device)
    return ParallelBeamRadon2D(
        image_size=case.truth.shape[-1], angles=angles, device=str(device)
    )


def _solver(_case):
    return FBPSolver()


@torch.no_grad()
def _load_cases(device: torch.device):
    return [load_ct_case(case_id, device=device) for case_id in QUALITY_CASE_IDS]


def test_quality_artifact_writer_accepts_injected_dependencies(tmp_path):
    case = load_ct_case(QUALITY_CASE_IDS[0])
    summary = write_quality_artifacts(
        [case],
        operator_factory=_operator,
        solver_factory=_solver,
        output_dir=tmp_path,
    )
    assert summary["benchmark_kind"] == "synthetic_quality"
    assert summary["cases"][0]["reference_kind"] == "backend_reference"
    assert (tmp_path / "synthetic_quality.json").exists()
    assert (tmp_path / "synthetic_quality.png").exists()
    assert (tmp_path / "synthetic_quality.pt").exists()


@torch.no_grad()
@pytest.mark.quality_benchmark
def test_write_all_quality_benchmark_artifacts(quality_artifact_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cases = _load_cases(device)
    summary = write_quality_artifacts(
        cases,
        operator_factory=_operator,
        solver_factory=_solver,
        output_dir=quality_artifact_dir,
    )
    assert len(summary["cases"]) == len(QUALITY_CASE_IDS)

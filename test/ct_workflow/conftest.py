from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .helpers import make_case_2d, solver_specs_2d


def pytest_addoption(parser):
    parser.addoption(
        "--run-ct-benchmark",
        action="store_true",
        default=False,
        help="Run CT workflow benchmark tests that write artifacts.",
    )
    parser.addoption(
        "--ct-artifact-dir",
        default=None,
        help="Directory for optional CT benchmark artifacts.",
    )


def pytest_configure(config):
    for marker, description in {
        "astra": "requires the ASTRA backend",
        "benchmark": "writes optional CT benchmark artifacts",
        "gpu": "requires a CUDA-capable GPU",
        "leap": "requires a configured LLNL LEAP cone model",
        "slow": "runs a larger or noisier reconstruction case",
        "smoke": "minimal Windows/fno environment smoke test",
    }.items():
        config.addinivalue_line("markers", f"{marker}: {description}")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-ct-benchmark"):
        return
    skip = pytest.mark.skip(reason="pass --run-ct-benchmark to write benchmark artifacts")
    for item in items:
        if "benchmark" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def ct_case_2d():
    return make_case_2d()


@pytest.fixture(scope="session")
def ct_case_2d_noisy_sparse():
    return make_case_2d(num_angles=16, noise_fraction=0.01, seed=1234)


@pytest.fixture
def ct_solvers_2d():
    return solver_specs_2d()


@pytest.fixture
def benchmark_artifact_dir(request, tmp_path):
    configured = request.config.getoption("--ct-artifact-dir")
    path = Path(configured).resolve() if configured else tmp_path / "ct_artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path

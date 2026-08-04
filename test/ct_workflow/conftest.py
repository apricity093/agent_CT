from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .helpers import make_case_2d, solver_specs_2d


def pytest_configure(config):
    for marker, description in {
        "astra": "requires the ASTRA backend",
        "gpu": "requires a CUDA-capable GPU",
        "leap": "requires a configured LLNL LEAP cone model",
        "slow": "runs a larger or noisier reconstruction case",
        "smoke": "minimal Windows/fno environment smoke test",
    }.items():
        config.addinivalue_line("markers", f"{marker}: {description}")


@pytest.fixture(scope="session")
def ct_case_2d():
    return make_case_2d()


@pytest.fixture(scope="session")
def ct_case_2d_noisy_sparse():
    return make_case_2d(num_angles=16, noise_fraction=0.01, seed=1234)


@pytest.fixture
def ct_solvers_2d():
    return solver_specs_2d()

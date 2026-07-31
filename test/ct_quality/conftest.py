from __future__ import annotations

from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-ct-quality",
        action="store_true",
        default=False,
        help="Run high-resolution CT quality tests that write artifacts.",
    )
    parser.addoption(
        "--ct-quality-artifact-dir",
        default=None,
        help="Directory for high-resolution CT quality artifacts.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "quality_benchmark: writes high-resolution CT quality artifacts"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-ct-quality"):
        return
    skip = pytest.mark.skip(reason="pass --run-ct-quality to write quality artifacts")
    for item in items:
        if "quality_benchmark" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def quality_artifact_dir(request, tmp_path):
    configured = request.config.getoption("--ct-quality-artifact-dir")
    path = Path(configured).resolve() if configured else tmp_path / "ct_quality"
    path.mkdir(parents=True, exist_ok=True)
    return path

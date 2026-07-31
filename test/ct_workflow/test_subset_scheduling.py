from __future__ import annotations

import pytest
import torch

from inv_framework.operators.base import LinearOperator
from inv_framework.operators.ct import ParallelBeamRadon2D
from inv_framework.solvers import OSSARTSolver, SARTSolver
from inv_framework.solvers import subset as subset_module

from .helpers import make_phantom_2d


class _RecordingSubsetOperator(LinearOperator):
    def __init__(self, delegate, record):
        self.delegate = delegate
        self.record = record
        self.domain_shape = delegate.domain_shape
        self.range_shape = delegate.range_shape

    def forward(self, x):
        self.record["forward_calls"] += 1
        return self.delegate.forward(x)

    def adjoint(self, y):
        self.record["adjoint_calls"] += 1
        return self.delegate.adjoint(y)


@pytest.mark.parametrize(
    ("solver", "block_size", "expected_calls"),
    [
        (SARTSolver(num_iterations=1, block_size=1, relaxation=0.25), 1, 8),
        (OSSARTSolver(num_iterations=1, block_size=2, relaxation=0.25), 2, 4),
    ],
    ids=["sart", "os_sart"],
)
def test_subset_solver_uses_every_requested_angle(
    monkeypatch,
    solver,
    block_size,
    expected_calls,
):
    operator = ParallelBeamRadon2D(image_size=16, num_angles=8, device="cpu")
    measurement = operator.forward(make_phantom_2d(16)).detach()
    original_factory = subset_module.make_subset_operator
    records = []

    def recording_factory(parent, indices):
        record = {
            "indices": tuple(int(value) for value in indices.detach().cpu().tolist()),
            "forward_calls": 0,
            "adjoint_calls": 0,
        }
        records.append(record)
        return _RecordingSubsetOperator(original_factory(parent, indices), record)

    monkeypatch.setattr(subset_module, "make_subset_operator", recording_factory)
    reconstruction = solver.solve(measurement, operator)

    assert reconstruction.shape == (1, *operator.domain_shape)
    assert torch.isfinite(reconstruction).all()
    assert len(records) == expected_calls
    assert all(len(record["indices"]) == block_size for record in records)
    assert all(record["forward_calls"] >= 2 for record in records)
    assert all(record["adjoint_calls"] >= 2 for record in records)
    flattened = [angle for record in records for angle in record["indices"]]
    assert sorted(flattened) == list(range(operator.num_angles))
    assert len(set(record["indices"] for record in records)) == expected_calls

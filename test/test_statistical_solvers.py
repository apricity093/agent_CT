import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inv_framework.operators.base import LinearOperator
from inv_framework.solvers.statistical import MLEMSolver, OSEMSolver


class PositiveAngleOperator(LinearOperator):
    def __init__(self, matrix: torch.Tensor, num_angles: int, detector_size: int, calls=None):
        self.matrix = matrix.clamp_min(0.05)
        self.num_angles = num_angles
        self.detector_size = detector_size
        self.domain_shape = (matrix.shape[1],)
        self.range_shape = (num_angles, detector_size)
        self.calls = [] if calls is None else calls

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        matrix = self.matrix.to(device=x.device, dtype=x.dtype)
        y = x.reshape(x.shape[0], -1).matmul(matrix.t())
        return y.reshape(x.shape[0], self.num_angles, self.detector_size)

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        matrix = self.matrix.to(device=y.device, dtype=y.dtype)
        return y.reshape(y.shape[0], -1).matmul(matrix)

    def subset(self, indices: torch.Tensor):
        index_tuple = tuple(int(v) for v in indices.detach().cpu().tolist())
        self.calls.append(index_tuple)
        rows = []
        for angle in index_tuple:
            start = angle * self.detector_size
            rows.extend(range(start, start + self.detector_size))
        return PositiveAngleOperator(
            self.matrix[rows],
            num_angles=len(index_tuple),
            detector_size=self.detector_size,
            calls=self.calls,
        )


def _operator():
    matrix = torch.tensor(
        [
            [1.0, 0.2, 0.1],
            [0.5, 1.0, 0.3],
            [0.2, 0.3, 1.0],
            [1.0, 0.4, 0.2],
            [0.3, 1.0, 0.5],
            [0.2, 0.6, 1.0],
        ]
    )
    return PositiveAngleOperator(matrix, num_angles=3, detector_size=2)


def test_mlem_outputs_nonnegative_shape():
    operator = _operator()
    x_true = torch.tensor([[0.5, 1.0, 0.25]])
    y = operator.forward(x_true)
    x = MLEMSolver(num_iterations=3).solve(y, operator)
    assert x.shape == (1, *operator.domain_shape)
    assert torch.all(x >= 0)


def test_osem_outputs_nonnegative_and_uses_subsets():
    operator = _operator()
    x_true = torch.tensor([[0.5, 1.0, 0.25]])
    y = operator.forward(x_true)
    x = OSEMSolver(num_iterations=2, block_size=1).solve(y, operator)
    assert x.shape == (1, *operator.domain_shape)
    assert torch.all(x >= 0)
    assert operator.calls[:3] == [(0,), (1,), (2,)]

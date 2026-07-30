import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inv_framework.operators.base import LinearOperator
from inv_framework.operators.ct.radon_torch import ParallelBeamRadon2D
from inv_framework.solvers._utils import make_subset_operator
from inv_framework.solvers.subset import OSSARTSolver, SARTSolver


class DenseAngleOperator(LinearOperator):
    def __init__(self, matrix: torch.Tensor, num_angles: int, detector_size: int, calls=None):
        self.matrix = matrix
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
        return DenseAngleOperator(
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
            [0.8, 0.1, 0.4],
            [0.1, 0.9, 0.6],
        ]
    )
    return DenseAngleOperator(matrix, num_angles=4, detector_size=2)


def test_sart_uses_subset_operator_and_returns_shape():
    operator = _operator()
    y = torch.ones(1, *operator.range_shape)
    x = SARTSolver(num_iterations=1, block_size=2, min_value=0.0).solve(y, operator)
    assert x.shape == (1, *operator.domain_shape)
    assert operator.calls == [(0, 1), (2, 3)]


def test_ossart_accepts_explicit_and_random_subsets():
    operator = _operator()
    y = torch.ones(1, *operator.range_shape)
    x = OSSARTSolver(
        num_iterations=1,
        subset_indices=[(0, 2), (1, 3)],
        min_value=0.0,
    ).solve(y, operator)
    assert x.shape == (1, *operator.domain_shape)
    assert operator.calls == [(0, 2), (1, 3)]

    operator_random = _operator()
    OSSARTSolver(num_iterations=1, block_size=2, order_strategy="random", seed=7).solve(y, operator_random)
    assert len(operator_random.calls) == 2


def test_parallel_beam_radon_subset_fallback_uses_angle_slice():
    operator = ParallelBeamRadon2D(image_size=8, num_angles=4, device="cpu")
    indices = torch.tensor([0, 2])
    sub_operator = make_subset_operator(operator, indices)
    assert isinstance(sub_operator, ParallelBeamRadon2D)
    assert sub_operator.range_shape[-2] == 2
    assert torch.allclose(sub_operator.angles, operator.angles[indices])

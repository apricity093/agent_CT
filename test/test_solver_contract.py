import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inv_framework.operators.base import ForwardOperator, LinearOperator
from inv_framework.solvers.classical import CGLSSolver, FDKSolver, LSQRSolver
from inv_framework.solvers.statistical import MLEMSolver
from inv_framework.solvers.subset import SARTSolver


class DenseLinearOperator(LinearOperator):
    def __init__(self, matrix: torch.Tensor):
        self.matrix = matrix
        self.domain_shape = (matrix.shape[1],)
        self.range_shape = (matrix.shape[0],)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        matrix = self.matrix.to(device=x.device, dtype=x.dtype)
        return x.reshape(x.shape[0], -1).matmul(matrix.t())

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        matrix = self.matrix.to(device=y.device, dtype=y.dtype)
        return y.reshape(y.shape[0], -1).matmul(matrix)


class DenseAngleOperator(DenseLinearOperator):
    def __init__(self, matrix: torch.Tensor, num_angles: int, detector_size: int):
        super().__init__(matrix)
        self.num_angles = num_angles
        self.detector_size = detector_size
        self.domain_shape = (matrix.shape[1],)
        self.range_shape = (num_angles, detector_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        return y.reshape(x.shape[0], self.num_angles, self.detector_size)

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        return super().adjoint(y.reshape(y.shape[0], -1))

    def subset(self, indices: torch.Tensor):
        rows = []
        for angle in [int(v) for v in indices.detach().cpu().tolist()]:
            start = angle * self.detector_size
            rows.extend(range(start, start + self.detector_size))
        return DenseAngleOperator(self.matrix[rows], len(rows) // self.detector_size, self.detector_size)


class FDKOperator(DenseLinearOperator):
    def fdk(self, y: torch.Tensor):
        return torch.zeros((y.shape[0], *self.domain_shape), device=y.device, dtype=y.dtype)


class NonlinearForward(ForwardOperator):
    def __init__(self):
        self.domain_shape = (3,)
        self.range_shape = (3,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x


def test_new_solver_shapes_and_x_init_dtype_migration():
    operator = DenseLinearOperator(torch.eye(3))
    y = torch.ones(2, 3, dtype=torch.float64)
    x_init = torch.zeros(2, 3, dtype=torch.float32)
    for solver in [CGLSSolver(num_iterations=1), LSQRSolver(num_iterations=1)]:
        x = solver.solve(y, operator, x_init=x_init)
        assert x.shape == (2, *operator.domain_shape)
        assert x.dtype == y.dtype

    fdk_x = FDKSolver().solve(y, FDKOperator(torch.eye(3)))
    assert fdk_x.shape == (2, 3)
    assert fdk_x.dtype == y.dtype


def test_subset_and_statistical_contract_shapes():
    matrix = torch.ones(4, 3)
    operator = DenseAngleOperator(matrix, num_angles=2, detector_size=2)
    y = torch.ones(2, *operator.range_shape)
    for solver in [SARTSolver(num_iterations=1, block_size=1), MLEMSolver(num_iterations=1)]:
        x = solver.solve(y, operator)
        assert x.shape == (2, *operator.domain_shape)


def test_x_init_shape_error():
    operator = DenseLinearOperator(torch.eye(3))
    y = torch.ones(1, 3)
    with pytest.raises(ValueError):
        CGLSSolver(num_iterations=1).solve(y, operator, x_init=torch.zeros(1, 4))


def test_adjoint_solvers_reject_forward_operator():
    operator = NonlinearForward()
    y = torch.ones(1, *operator.range_shape)
    solvers = [
        CGLSSolver(num_iterations=1),
        LSQRSolver(num_iterations=1),
        FDKSolver(),
        SARTSolver(num_iterations=1),
        MLEMSolver(num_iterations=1),
    ]
    for solver in solvers:
        with pytest.raises(TypeError, match="LinearOperator"):
            solver.solve(y, operator)

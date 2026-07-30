import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inv_framework.operators.base import LinearOperator
from inv_framework.solvers.classical import CGLSSolver, FDKSolver, LSQRSolver


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


class FDKDenseOperator(DenseLinearOperator):
    def __init__(self, matrix: torch.Tensor):
        super().__init__(matrix)
        self.called = False

    def fdk(self, y: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        self.called = True
        return torch.full((y.shape[0], *self.domain_shape), scale, device=y.device, dtype=y.dtype)


def test_cgls_residual_decreases():
    torch.manual_seed(0)
    matrix = torch.tensor(
        [[2.0, 0.0, 1.0], [0.0, 1.5, 0.5], [1.0, 0.5, 2.0], [0.5, 1.0, 0.0]]
    )
    operator = DenseLinearOperator(matrix)
    x_true = torch.tensor([[0.5, 1.0, -0.25]])
    y = operator.forward(x_true)
    x_rec = CGLSSolver(num_iterations=12, tol=0.0).solve(y, operator)
    initial_residual = (operator.forward(torch.zeros_like(x_rec)) - y).norm()
    final_residual = (operator.forward(x_rec) - y).norm()
    assert x_rec.shape == (1, *operator.domain_shape)
    assert final_residual < 0.05 * initial_residual


def test_lsqr_matches_dense_lstsq_residual():
    torch.manual_seed(1)
    matrix = torch.tensor(
        [[1.0, 0.0, 0.5], [0.0, 1.0, 0.25], [1.0, 1.0, 0.0], [0.5, -0.5, 1.0], [0.0, 0.5, 1.5]]
    )
    operator = DenseLinearOperator(matrix)
    y = torch.tensor([[1.0, 0.5, 1.25, 0.0, 1.0]])
    x_rec = LSQRSolver(num_iterations=20, atol=0.0, btol=0.0).solve(y, operator)
    ref = torch.linalg.lstsq(matrix, y.t()).solution.t()
    residual = (operator.forward(x_rec) - y).norm()
    ref_residual = (operator.forward(ref) - y).norm()
    assert x_rec.shape == (1, *operator.domain_shape)
    assert residual <= ref_residual + 1e-4


def test_fdk_requires_backend():
    operator = DenseLinearOperator(torch.eye(3))
    y = torch.ones(1, 3)
    with pytest.raises(NotImplementedError):
        FDKSolver().solve(y, operator)


def test_fdk_calls_operator_backend():
    operator = FDKDenseOperator(torch.eye(3))
    y = torch.ones(2, 3)
    out = FDKSolver(scale=2.5).solve(y, operator)
    assert operator.called
    assert out.shape == (2, *operator.domain_shape)
    assert torch.allclose(out, torch.full_like(out, 2.5))

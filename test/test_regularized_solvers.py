import pytest
import torch

from inv_framework.operators.base import ForwardOperator, LinearOperator
from inv_framework.regularizers import TikhonovRegularizer, TVRegularizer
from inv_framework.solvers import TikhonovSolver, TVFISTASolver, tikhonov, tv_fista


class DenseLinearOperator(LinearOperator):
    def __init__(self, matrix: torch.Tensor):
        self.matrix = matrix
        self.domain_shape = (matrix.shape[1],)
        self.range_shape = (matrix.shape[0],)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.matrix.T

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        return y @ self.matrix


class IdentityImageOperator(LinearOperator):
    def __init__(self, shape=(1, 8, 8)):
        self.domain_shape = tuple(shape)
        self.range_shape = tuple(shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        return y


class NonlinearImageOperator(ForwardOperator):
    domain_shape = (1, 8, 8)
    range_shape = (1, 8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.square()


def test_tikhonov_matches_dense_closed_form():
    matrix = torch.tensor(
        [[1.0, 2.0, 0.0], [0.0, 1.0, 1.0], [2.0, 0.0, 1.0], [1.0, -1.0, 0.5]],
        dtype=torch.float64,
    )
    operator = DenseLinearOperator(matrix)
    measurement = torch.tensor(
        [[1.0, 0.5, 2.0, -0.25], [0.0, 1.0, 0.5, 1.5]],
        dtype=torch.float64,
    )
    strength = 0.2

    reconstruction = tikhonov(
        operator,
        measurement,
        reg_strength=strength,
        num_iterations=30,
        tolerance=1e-12,
    )
    normal = matrix.T @ matrix + strength * torch.eye(3, dtype=matrix.dtype)
    expected = torch.linalg.solve(normal, matrix.T @ measurement.T).T

    assert torch.allclose(reconstruction, expected, atol=1e-9, rtol=1e-8)


def test_generalized_tikhonov_matches_closed_form():
    matrix = torch.tensor([[1.0, 0.0], [0.5, 1.0], [0.0, 2.0]], dtype=torch.float64)
    regularization_matrix = torch.tensor([[1.0, -1.0]], dtype=torch.float64)
    operator = DenseLinearOperator(matrix)
    regularization_operator = DenseLinearOperator(regularization_matrix)
    measurement = torch.tensor([[1.0, 0.5, 2.0]], dtype=torch.float64)
    strength = 0.4

    reconstruction = TikhonovSolver(
        reg_strength=strength,
        num_iterations=20,
        tolerance=1e-12,
        regularization_operator=regularization_operator,
    ).solve(measurement, operator)
    normal = matrix.T @ matrix + strength * regularization_matrix.T @ regularization_matrix
    expected = torch.linalg.solve(normal, matrix.T @ measurement.T).T

    assert torch.allclose(reconstruction, expected, atol=1e-9, rtol=1e-8)


def test_tikhonov_regularizer_value_and_gradient():
    x = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
    regularizer = TikhonovRegularizer()

    assert torch.allclose(regularizer.value(x), torch.tensor([2.5, 12.5]))
    assert torch.equal(regularizer.gradient(x), x)


@pytest.mark.parametrize("mode", ["isotropic", "anisotropic"])
def test_tv_proximal_preserves_constant_and_reduces_tv(mode):
    regularizer = TVRegularizer(mode=mode, num_iterations=80, tolerance=1e-7)
    constant = torch.ones(2, 1, 8, 8, dtype=torch.float64)
    noisy = constant.clone()
    noisy[:, :, 2:6, 2:6] += 1.0
    noisy[:, :, ::2, 1::2] -= 0.4

    constant_result = regularizer.proximal(constant, 0.2)
    denoised = regularizer.proximal(noisy, 0.2)

    assert torch.allclose(constant_result, constant, atol=1e-10)
    assert torch.all(regularizer.value(denoised) <= regularizer.value(noisy) + 1e-8)
    assert denoised.shape == noisy.shape
    assert denoised.dtype == noisy.dtype


@pytest.mark.parametrize("mode", ["isotropic", "anisotropic"])
def test_tv_proximal_matches_two_pixel_closed_form(mode):
    regularizer = TVRegularizer(mode=mode, num_iterations=200, tolerance=1e-10)
    image = torch.tensor([[[[0.0, 1.0]]]], dtype=torch.float64)

    result = regularizer.proximal(image, 0.2)

    assert torch.allclose(result, torch.tensor([[[[0.2, 0.8]]]], dtype=torch.float64), atol=1e-7)


def test_tv_fista_decreases_objective_and_estimates_step():
    operator = IdentityImageOperator()
    measurement = torch.zeros(2, *operator.range_shape, dtype=torch.float64)
    measurement[:, :, 2:6, 2:6] = 1.0
    measurement[:, :, 1::2, ::2] += 0.2
    x_init = torch.zeros_like(measurement)
    regularizer = TVRegularizer(num_iterations=60, tolerance=1e-7)
    strength = 0.08

    reconstruction = tv_fista(
        operator,
        measurement,
        reg_strength=strength,
        num_iterations=30,
        x_init=x_init,
        regularizer=regularizer,
        tolerance=1e-8,
    )

    initial_objective = 0.5 * (x_init - measurement).reshape(2, -1).square().sum(dim=1)
    final_objective = (
        0.5 * (reconstruction - measurement).reshape(2, -1).square().sum(dim=1)
        + strength * regularizer.value(reconstruction)
    )
    assert torch.all(final_objective < initial_objective)
    assert reconstruction.shape == measurement.shape


def test_regularized_solver_contract_and_nonlinear_rejection():
    linear = IdentityImageOperator()
    nonlinear = NonlinearImageOperator()
    measurement = torch.ones(1, *linear.range_shape)

    tikhonov_result = TikhonovSolver(num_iterations=5).solve(measurement, linear)
    tv_result = TVFISTASolver(num_iterations=3).solve(measurement, linear)

    assert tikhonov_result.shape == (1, *linear.domain_shape)
    assert tv_result.shape == (1, *linear.domain_shape)
    with pytest.raises(TypeError, match="LinearOperator"):
        TikhonovSolver().solve(measurement, nonlinear)
    with pytest.raises(TypeError, match="LinearOperator"):
        TVFISTASolver().solve(measurement, nonlinear)

"""Independent small-matrix and coefficient checks for P1."""

import pytest
import torch

from inv_framework.core.operator import LinearOperator
from inv_framework.core.operator import GradientOperator, gradient
from inv_framework.core.operator import HaarOperator
from inv_framework.core.regularizer import TikhonovRegularizer, HaarL1Regularizer, TVRegularizer
from inv_framework.solvers import FISTASolver, TikhonovSolver, SolveControl, fista
from inv_framework.ct.registry import validate_parameter_values


class Matrix(LinearOperator):
    def __init__(self, matrix, shape):
        self.matrix = matrix
        self.domain_shape = shape
        self.range_shape = (matrix.shape[0],)
        self.calls = [0, 0]

    def forward(self, x):
        self.calls[0] += 1
        return x.flatten(1) @ self.matrix.T

    def adjoint(self, y):
        self.calls[1] += 1
        return (y @ self.matrix).reshape(y.shape[0], *self.domain_shape)

    def pseudo_inverse(self, y, **kwargs):
        return (y @ torch.linalg.pinv(self.matrix).T).reshape(y.shape[0], *self.domain_shape)




def test_identity_strength_nonnegative_and_convergence():
    a = torch.eye(16, dtype=torch.float64)
    op = Matrix(a, (1, 4, 4))
    y = torch.linspace(-2., 2., 16, dtype=torch.float64).unsqueeze(0)
    solver = FISTASolver(strength=0.7, min_value=0, step_size=1., num_iterations=20)
    result = solver.solve_detailed(y, op)
    torch.testing.assert_close(result.reconstruction.flatten(1), (y/1.7).clamp_min(0))
    assert result.status == 'converged'
    assert result.metadata['normalized_prox_gradient_mapping'] < 1e-14
    assert result.actual_iterations < 20


def explicit_d(h, w):
    rows = []
    for i in range(h):
        for j in range(w):
            for di, dj in ((1, 0), (0, 1)):
                if i+di < h and j+dj < w:
                    row = torch.zeros(h*w, dtype=torch.float64)
                    row[(i+di)*w+j+dj], row[i*w+j] = 1, -1
                    rows.append(row)
    return torch.stack(rows)


def test_h1_value_gradient_direction_and_generalized_cg():
    torch.manual_seed(7)
    d = GradientOperator((1, 4, 4))
    matrix_d = explicit_d(4, 4)
    reg = TikhonovRegularizer(d)
    x, direction = torch.randn(2, 1, 4, 4, dtype=torch.float64), torch.randn(2, 1, 4, 4, dtype=torch.float64)
    torch.testing.assert_close(d.forward(x), torch.stack(gradient(x), dim=1))
    torch.testing.assert_close(reg.value(x), 0.5*(x.flatten(1)@matrix_d.T).square().sum(1))
    torch.testing.assert_close(reg.gradient(x).flatten(1), x.flatten(1)@matrix_d.T@matrix_d)
    numerical = (reg.value(x+1e-6*direction)-reg.value(x-1e-6*direction))/(2e-6)
    torch.testing.assert_close(numerical, (reg.gradient(x)*direction).flatten(1).sum(1), rtol=1e-8, atol=1e-8)
    a = torch.diag(torch.linspace(1., 2., 16, dtype=x.dtype))
    op = Matrix(a, (1, 4, 4))
    y = torch.randn(2, 16, dtype=x.dtype)
    solver = TikhonovSolver(reg_strength=0.4, regularization_operator=d, num_iterations=60, tolerance=1e-12)
    result = solver.solve_detailed(y, op)
    expected = torch.linalg.solve(a.T@a + 0.4*matrix_d.T@matrix_d, (y@a).T).T
    error = float((result.reconstruction.flatten(1)-expected).norm()/expected.norm())
    assert error < 1e-6
    smooth = FISTASolver(reg, strength=.4, num_iterations=300, tolerance=1e-9).solve_detailed(y, op)
    assert smooth.status != 'invalid_parameters'
    torch.testing.assert_close(smooth.reconstruction.flatten(1), expected, atol=1e-6, rtol=1e-6)
    with pytest.raises(ValueError, match='Generalized'):
        reg.proximal(x, .1)
    print(f'H1 CG relative error={error:.3e}')


@pytest.mark.parametrize('level', [1, 2, 3])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_haar_orthogonal_layout_and_independent_channels(level, device):
    torch.manual_seed(9)
    x = torch.randn(2, 3, 8, 16, dtype=torch.float64, device=device)
    c = torch.randn_like(x)
    w = HaarOperator((3, 8, 16), level)
    torch.testing.assert_close(w.adjoint(w.forward(x)), x, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(w.forward(w.adjoint(c)), c, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close((w.forward(x)*c).sum(), (x*w.adjoint(c)).sum(), rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(w.forward(x).square().sum(), x.square().sum(), rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(w.forward(x)[1, 2], w.forward(x[1:2, 2:3])[0, 0])
    constant = w.forward(torch.ones_like(x))
    h, width = 8//2**level, 16//2**level
    torch.testing.assert_close(constant[..., :h, :width], torch.full_like(constant[..., :h, :width], 2**level))
    constant[..., :h, :width] = 0
    assert constant.abs().max() < 1e-14
    # Isolated packed coefficients at every level identify LH/HL/HH phases.
    for depth in range(level):
        height, width = 8//2**depth, 16//2**depth
        for i, j in ((0, width//2), (height//2, 0), (height//2, width//2)):
            coefficient = torch.zeros_like(c)
            coefficient[..., i, j] = 1
            torch.testing.assert_close(w.forward(w.adjoint(coefficient)), coefficient, atol=1e-14, rtol=1e-13)
    one = HaarOperator((1, 2, 2), 1)
    torch.testing.assert_close(one.forward(torch.tensor([[[[1., 2.], [3., 4.]]]], dtype=x.dtype, device=device)),
                               torch.tensor([[[[5., -1.], [-2., 0.]]]], dtype=x.dtype, device=device), atol=1e-14, rtol=1e-13)




@pytest.mark.parametrize('level,shape', [(0,(8,8)),(-1,(8,8)),(1.5,(8,8)),(True,(8,8)),(2,(6,8)),(2,(8,7))])
def test_haar_rejects_level_or_shape(level, shape):
    with pytest.raises(ValueError):
        HaarOperator(shape, level)




@pytest.mark.parametrize('params', [dict(regularizer='unknown'),
    dict(strength=.2,reg_strength=.3),dict(strength=float('inf')),dict(level=1.5),dict(power_iterations=0)])
def test_composite_parameter_rejection(params):
    assert validate_parameter_values('fista', params).errors


def test_two_level_haar_layout_and_detail_weights():
    x = torch.arange(1.,17.,dtype=torch.float64).reshape(1,1,4,4)
    expected = torch.tensor([[34.,-4.,-1.,-1.],[-16.,0.,-1.,-1.],[-4.,-4.,0.,0.],[-4.,-4.,0.,0.]],dtype=x.dtype).reshape_as(x)
    torch.testing.assert_close(HaarOperator((1,4,4)).forward(x),expected,atol=2e-14,rtol=1e-13)
    reg = HaarL1Regularizer()
    assert reg.value(x).item() == pytest.approx(40.)
    assert reg.value(torch.ones_like(x)).item() == pytest.approx(0.,abs=1e-14)




@pytest.mark.parametrize('bad',[float('nan'),float('inf')])
def test_nonfinite_operator_output_and_proximal(bad):
    class Broken(Matrix):
        def adjoint(self,y):
            return torch.full((y.shape[0],*self.domain_shape),bad,dtype=y.dtype)
    y = torch.ones(1,16,dtype=torch.float64)
    result = FISTASolver(step_size=1.).solve_detailed(y,Broken(torch.eye(16,dtype=y.dtype),(1,4,4)))
    assert result.status == 'numerical_error' and result.actual_iterations == 0
    for reg in (HaarL1Regularizer(),TikhonovRegularizer()):
        with pytest.raises(ValueError):
            reg.proximal(y.reshape(1,1,4,4),bad)




@pytest.fixture(autouse=True, params=["cpu", "cuda"])
def numerical_device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    with torch.device(request.param):
        yield request.param

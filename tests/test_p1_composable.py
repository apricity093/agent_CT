"""Independent small-matrix and coefficient checks for P1."""

import pytest
import torch

from inv_framework.operators.base import LinearOperator
from inv_framework.operators.gradient import GradientOperator, gradient
from inv_framework.operators.wavelet import HaarOperator
from inv_framework.regularizers import TikhonovRegularizer, HaarL1Regularizer, TVRegularizer
from inv_framework.solvers import FISTASolver, TikhonovSolver, SolveControl, fista
from inv_framework.instrumentation import CountingLinearOperator
from inv_framework.solvers.specs import validate_parameter_values


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


def test_quadratic_fista_closed_form_backtracking_and_actual_cost():
    torch.manual_seed(18)
    q, _ = torch.linalg.qr(torch.randn(16, 16, dtype=torch.float64))
    a = torch.diag(torch.linspace(1., 3., 16, dtype=torch.float64)) @ q
    base = Matrix(a, (1, 4, 4))
    op = CountingLinearOperator(base)
    y = torch.randn(2, 16, dtype=torch.float64)
    reg = TikhonovRegularizer()
    v = torch.randn(2, 1, 4, 4, dtype=torch.float64)
    torch.testing.assert_close(reg.proximal(v, 0.4), v/1.4)
    strength = 0.3
    solver = FISTASolver(reg, strength=strength, num_iterations=250, step_size=10., tolerance=0)
    result = solver.solve_detailed(y, op, control=SolveControl(tolerance=0, stop_on_convergence=False))
    expected = torch.linalg.solve(a.T@a + strength*torch.eye(16, dtype=a.dtype), (y@a).T).T
    error = float((result.reconstruction.flatten(1)-expected).norm()/expected.norm())
    assert error < 1e-8
    assert result.resources['backtracking_rejections'] >= 6
    assert base.calls == [op.stats()['forward_calls'], op.stats()['adjoint_calls']]
    assert op.stats()['forward_calls'] == 2*250 + result.resources['backtracking_rejections']
    assert op.stats()['adjoint_calls'] == 2*250
    assert op.stats()['prox_calls'] == 2*250 + result.resources['backtracking_rejections']
    assert result.metadata['endpoint_mapping_complete']
    assert result.trajectory[-1].metadata['objective_state'] == 'returned'
    assert result.final_objective == pytest.approx(result.trajectory[-1].objective)
    assert result.status == 'max_iterations'
    torch.testing.assert_close(solver.solve(y, base), result.reconstruction)
    torch.testing.assert_close(fista(base, y, regularizer=reg, strength=strength, num_iterations=250,
                                     step_size=10., tolerance=0), result.reconstruction)
    print(f'quadratic relative error={error:.3e}, counted_forward={op.stats()['forward_calls']}, counted_adjoint={op.stats()['adjoint_calls']}, rejected={result.resources["backtracking_rejections"]}')


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
    before = list(op.calls)
    rejected = FISTASolver(reg).solve_detailed(y, op)
    assert rejected.status == 'invalid_parameters' and rejected.actual_iterations == 0
    assert op.calls == before
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


def test_haar_proximal_and_identity_solution_independent_matrix():
    torch.manual_seed(11)
    # Independent Kronecker Haar matrix for one level; detail shrink with LL retained.
    h = torch.tensor([[1., 1.], [1., -1.]], dtype=torch.float64)/2**0.5
    matrix_w = torch.kron(h, h)
    y = torch.randn(2, 4, dtype=torch.float64)
    reg = HaarL1Regularizer(level=1)
    coefficients = y@matrix_w.T
    shrunk = coefficients.sign()*(coefficients.abs()-.25).clamp_min(0)
    shrunk[:, 0] = coefficients[:, 0]
    expected = shrunk@matrix_w
    torch.testing.assert_close(reg.proximal(y.reshape(2,1,2,2), .25).flatten(1), expected)
    torch.testing.assert_close(reg.value(y.reshape(2,1,2,2)), coefficients[:, 1:].abs().sum(1))
    torch.testing.assert_close(reg.proximal(y.reshape(2,1,2,2), 0).flatten(1), y)
    op = CountingLinearOperator(Matrix(torch.eye(4, dtype=y.dtype), (1, 2, 2)))
    result = FISTASolver(reg, strength=.25, step_size=1., num_iterations=4, tolerance=0).solve_detailed(y, op)
    torch.testing.assert_close(result.reconstruction.flatten(1), expected)
    assert result.resources['haar_analysis_calls'] == 13  # initial value plus update, value, endpoint
    assert result.resources['haar_synthesis_calls'] == 8
    assert op.stats()['prox_calls'] == 8
    assert result.resources['haar_transform_seconds'] > 0


@pytest.mark.parametrize('level,shape', [(0,(8,8)),(-1,(8,8)),(1.5,(8,8)),(True,(8,8)),(2,(6,8)),(2,(8,7))])
def test_haar_rejects_level_or_shape(level, shape):
    with pytest.raises(ValueError):
        HaarOperator(shape, level)


@pytest.mark.parametrize('prior', [TikhonovRegularizer(), HaarL1Regularizer(), TVRegularizer(num_iterations=2)])
def test_fista_budget_and_nonfinite(prior):
    op = CountingLinearOperator(Matrix(torch.eye(16, dtype=torch.float64), (1,4,4)), max_adjoint_calls=1)
    y = torch.ones(1,16,dtype=torch.float64)
    result = FISTASolver(prior, step_size=1., num_iterations=5).solve_detailed(y, op)
    assert result.status == 'resource_exhausted'
    assert result.actual_iterations == 1 and torch.isfinite(result.reconstruction).all()
    assert not result.metadata['endpoint_mapping_complete']
    assert op.stats()['adjoint_calls'] == 1
    y[:,0] = float('nan')
    result = FISTASolver(prior).solve_detailed(y, op)
    assert result.status == 'numerical_error' and result.actual_iterations == 0


@pytest.mark.parametrize('params', [dict(regularizer='h1'), dict(regularizer='haar_l1',min_value=0),
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


def test_budget_during_norm_backtracking_and_endpoint_retains_last_point():
    y = torch.ones(1,16,dtype=torch.float64)
    for budget, options, accepted in [(1, {}, 0), (2, {'step_size':10.}, 0), (2, {'step_size':1.}, 1)]:
        base = Matrix(torch.eye(16,dtype=y.dtype),(1,4,4))
        op = CountingLinearOperator(base,max_forward_calls=budget,max_adjoint_calls=1)
        result = FISTASolver(num_iterations=5,**options).solve_detailed(y,op)
        assert result.status == 'resource_exhausted'
        assert result.actual_iterations == accepted
        assert op.stats()['forward_calls'] <= budget
        assert base.calls == [op.stats()['forward_calls'],op.stats()['adjoint_calls']]
        if accepted:
            torch.testing.assert_close(result.reconstruction.flatten(1),y/1.01)
        else:
            assert result.reconstruction.count_nonzero() == 0


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


def test_tensor_entry_preserves_budget_exception_and_alias_source():
    from inv_framework.instrumentation import OperatorBudgetExceeded
    y = torch.ones(1,16,dtype=torch.float64)
    operator = CountingLinearOperator(Matrix(torch.eye(16,dtype=y.dtype),(1,4,4)), max_forward_calls=1)
    with pytest.raises(OperatorBudgetExceeded):
        fista(operator,y,step_size=1.)
    for algorithm, alias, canonical in [('fista','reg_strength','strength'),('tikhonov','strength','reg_strength')]:
        result = validate_parameter_values(algorithm,{alias:.4},parameter_sources={alias:'repository_config@sha256'})
        assert result.valid and result.parameters[canonical] == .4
        assert result.sources[canonical] == 'repository_config@sha256'


@pytest.fixture(autouse=True, params=["cpu", "cuda"])
def numerical_device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    with torch.device(request.param):
        yield request.param

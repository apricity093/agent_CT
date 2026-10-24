"""P0 independent transpose, derivative and boundary regressions."""

import math

import pytest
import torch

from inv_framework.instrumentation import CountingLinearOperator
from inv_framework.operators.ct import radon_torch
from inv_framework.operators.ct.radon_torch import ParallelBeamRadon2D
from inv_framework.operators.gradient import gradient, gradient_adjoint
from inv_framework.regularizers import TVRegularizer
from p0_reference import explicit_matrix, frozen_forward, rotation_grid


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))]


def relative_error(actual, expected):
    return float((actual - expected).norm() / expected.norm().clamp_min(1e-12))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("size", [8, 16])
@pytest.mark.parametrize("geometry", ["uniform", "limited", "nonuniform"])
def test_ct_transpose_and_derivatives(device, size, geometry):
    torch.manual_seed(17)
    angles = {
        "uniform": torch.arange(7) * math.pi / 7,
        "limited": torch.arange(7) * (2 * math.pi / 3) / 7,
        "nonuniform": torch.tensor([0.0, 0.13, 0.47, 0.81, 1.52, 2.17, 2.91]),
    }[geometry].to(device)
    op = ParallelBeamRadon2D(size, angles=angles, device=device, in_channels=3)
    x = torch.randn(2, 3, size, size, device=device, requires_grad=True)
    y = torch.randn(2, 3, 7, size, device=device, requires_grad=True)
    matrix = explicit_matrix(size, angles)
    dual_basis = torch.eye(7 * size, device=device).reshape(-1, 1, 7, size)
    transpose_matrix = op.adjoint(dual_basis).reshape(7 * size, -1)
    assert relative_error(transpose_matrix, matrix) < (1e-5 if device == "cpu" else 1e-4)
    expected = (y.reshape(6, -1) @ matrix).reshape_as(x)
    actual = op.adjoint(y)
    threshold = 1e-5 if device == "cpu" else 1e-4
    transpose_error = relative_error(actual, expected)
    assert transpose_error < threshold
    # A frozen native forward both checks compatibility and bypasses custom
    # backward, so agreement cannot follow just from the paired Functions.
    native = frozen_forward(x, angles)
    assert torch.equal(op.forward(x), native)
    native_vjp, = torch.autograd.grad(native, x, y, retain_graph=True)
    assert relative_error(actual, native_vjp) < threshold
    forward_vjp, = torch.autograd.grad(op.forward(x), x, y, create_graph=True)
    assert relative_error(forward_vjp, expected) < threshold
    adjoint_vjp, = torch.autograd.grad(actual, y, x)
    assert relative_error(adjoint_vjp, native) < threshold
    # Verify the direct scatter computation's own derivative, too.
    raw_vjp, = torch.autograd.grad(
        radon_torch._radon_adjoint_impl(y, angles, size), y, x)
    assert relative_error(raw_vjp, native) < threshold
    lhs, rhs = (native * y).sum(), (x * actual).sum()
    inner_error = float((lhs - rhs).abs() / (native.norm() * y.norm()))
    assert inner_error < threshold
    # L2 data term: finite differences of the frozen forward vs autograd.
    loss = 0.5 * (op.forward(x) - y).square().sum()
    grad, = torch.autograd.grad(loss, x)
    direction = grad.detach() / grad.detach().norm()
    h = 1e-2  # float32, central difference (quadratic loss)
    plus = 0.5 * (frozen_forward(x.detach() + h * direction, angles) - y.detach()).square().sum()
    minus = 0.5 * (frozen_forward(x.detach() - h * direction, angles) - y.detach()).square().sum()
    fd = (plus - minus) / (2 * h)
    derivative_error = float((fd - (grad * direction).sum()).abs() / fd.abs())
    assert derivative_error < 2e-3
    # Double-backward Hessian-vector product uses the same transpose pair.
    hvp, = torch.autograd.grad(forward_vjp, y, direction)
    assert relative_error(hvp, frozen_forward(direction, angles)) < threshold
    print(f"{device} {size} {geometry}: transpose={transpose_error:.3e}, "
          f"inner={inner_error:.3e}, directional={derivative_error:.3e}")


def test_isolated_float64_transpose(monkeypatch):
    # Production CT grids/angles remain float32. Supply a double grid only
    # here to verify the scatter arithmetic against an independent matrix.
    monkeypatch.setattr(radon_torch, "_rotation_grid",
                        lambda theta, shape: rotation_grid(theta, shape, torch.float64))
    torch.manual_seed(17)
    angles = torch.tensor([0.0, 0.17, 0.73, 1.4, 2.6], dtype=torch.float64)
    matrix = explicit_matrix(8, angles, torch.float64)
    y = torch.randn(2, 2, 5, 8, dtype=torch.float64)
    expected = (y.reshape(4, -1) @ matrix).reshape(2, 2, 8, 8)
    actual = radon_torch._radon_adjoint_impl(y, angles, 8)
    error = relative_error(actual, expected)
    assert error < 1e-8
    x = torch.randn_like(expected, requires_grad=True)
    native_vjp, = torch.autograd.grad(frozen_forward(x, angles, torch.float64), x, y)
    assert relative_error(actual, native_vjp) < 1e-8
    print(f"isolated float64 transpose={error:.3e}")


def test_adjoint_has_no_hidden_forward_or_vjp(monkeypatch):
    op = CountingLinearOperator(ParallelBeamRadon2D(8, 7), max_forward_calls=0,
                                max_adjoint_calls=1)
    def forbidden(*args, **kwargs):
        raise AssertionError("adjoint must not execute forward or VJP")
    monkeypatch.setattr(radon_torch, "_radon_forward_impl", forbidden)
    monkeypatch.setattr(torch.autograd, "grad", forbidden)
    with torch.no_grad():
        result = op.adjoint(torch.ones(1, 1, 7, 8))
    assert torch.isfinite(result).all()
    assert op.stats()["forward_calls"] == 0
    assert op.stats()["adjoint_calls"] == 1


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(2, 3, 8, 16), (2, 1, 1, 8), (1, 2, 8, 1), (2, 1, 1, 1)])
def test_gradient_arbitrary_dual_and_boundaries(device, shape):
    torch.manual_seed(29)
    x = torch.randn(shape, dtype=torch.float64, device=device, requires_grad=True)
    qv, qh = torch.randn_like(x), torch.randn_like(x)
    v, h = gradient(x)
    lhs = (v * qv + h * qh).sum()
    adjoint = gradient_adjoint(qv, qh)
    assert torch.allclose(lhs, (x * adjoint).sum(), atol=1e-10, rtol=1e-10)
    vjp, = torch.autograd.grad(lhs, x)
    assert torch.allclose(vjp, adjoint, atol=1e-12, rtol=1e-12)
    assert not v[..., -1, :].any() and not h[..., :, -1].any()
    bv, bh = torch.zeros_like(x), torch.zeros_like(x)
    bv[..., -1, :], bh[..., :, -1] = 7, -11
    assert torch.equal(gradient_adjoint(bv, bh), torch.zeros_like(x))
    assert torch.equal(gradient_adjoint(qv + bv, qh + bh), adjoint)
    assert all(torch.equal(g, torch.zeros_like(x)) for g in gradient(torch.ones_like(x)))


@pytest.mark.parametrize("mode", ["isotropic", "anisotropic"])
def test_tv_value_and_proximal_regression(mode):
    corner = torch.tensor([[[[0., 1.], [2., 4.]]]], dtype=torch.float64)
    value = math.sqrt(5) + 5 if mode == "isotropic" else 8
    torch.testing.assert_close(TVRegularizer(mode=mode).value(corner),
                               torch.tensor([value], dtype=corner.dtype))
    # Exact 1x2 ROF solution, including independent batch/channel offsets.
    x = torch.tensor([[[[0., 1.]], [[2., 3.]]], [[[4., 5.]], [[-1., 0.]]]], dtype=torch.float64)
    tv = TVRegularizer(mode=mode, num_iterations=200, tolerance=1e-10)
    assert torch.equal(tv.value(x), torch.tensor([2., 2.], dtype=x.dtype))
    expected = x + torch.tensor([0.2, -0.2], dtype=x.dtype)
    torch.testing.assert_close(tv.proximal(x, 0.2), expected, atol=1e-7, rtol=0)
    assert torch.equal(tv.proximal(x, 0), x)
    assert tv._gradient is gradient and tv._gradient_adjoint is gradient_adjoint

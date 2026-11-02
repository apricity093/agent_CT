"""Acceptance uses the prescribed inner-product denominator on CPU and CUDA."""
import math
import pytest
import torch
from inv_framework.ct.backends.radon_torch import ParallelBeamRadon2D
from inv_framework.core.operator import GradientOperator, HaarOperator, WeightedLinearOperator

@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("kind", ["radon", "weighted", "gradient", "haar"])
def test_adjoint(kind, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(17)
    shape = (1, 16, 16)
    radon = ParallelBeamRadon2D(16, angles=torch.arange(13, device=device)*math.pi/13, device=device)
    if kind == "radon":
        operator = radon
    elif kind == "weighted":
        operator = WeightedLinearOperator(radon, torch.rand((1, *radon.range_shape), device=device)+0.2)
    elif kind == "gradient":
        operator = GradientOperator(shape)
    else:
        operator = HaarOperator(shape)
    x = torch.randn((1, *shape), device=device)
    y = torch.randn((1, *operator.range_shape), device=device)
    lhs = (operator.forward(x).double()*y.double()).sum()
    rhs = (x.double()*operator.adjoint(y).double()).sum()
    error = float((lhs-rhs).abs()/lhs.abs())
    print(f"{kind} {device}: {error:.8g}")
    assert error < 1e-5


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_astra_adjoint(device):
    pytest.importorskip("astra")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from inv_framework.ct.backends.astra_adapter import ASTRAOperator2D
    torch.manual_seed(17)
    operator = ASTRAOperator2D(16, torch.arange(13)*math.pi/13, device)
    try:
        x = torch.randn(1, 1, 16, 16, device=device)
        y = torch.randn(1, 1, 13, 16, device=device)
        lhs = (operator.forward(x).double()*y.double()).sum()
        rhs = (x.double()*operator.adjoint(y).double()).sum()
        x.requires_grad_()
        gradient, = torch.autograd.grad((operator.forward(x)*y).sum(), x)
        torch.testing.assert_close(gradient, operator.adjoint(y))
        error = float((lhs-rhs).abs()/lhs.abs())
        print(f"astra {device}: {error:.8g}")
        assert error < 1e-5
    finally:
        operator.close()

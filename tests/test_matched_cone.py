"""The cone projector and its transpose share the same physical ray weights."""
import pytest
import torch
import numpy as np
from inv_framework.ct.backends.astra_fdk_adapter import ASTRAFDKOperator3D
from inv_framework.ct.backends import astra_adapter


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_cone_transpose_against_explicit_matrix(device,dtype):
    if not astra_adapter._HAS_ASTRA:
        pytest.skip('ASTRA unavailable')
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    astra = astra_adapter.astra
    op = ASTRAFDKOperator3D(astra.create_vol_geom(4,4,4), astra.create_proj_geom(
        'cone',1.,1.,6,6,np.array([0.,.4,1.2]),20.,20.))
    torch.manual_seed(71)
    x = torch.randn(1,4,4,4,device=device,dtype=dtype,requires_grad=True)
    basis = torch.eye(64,device=device,dtype=dtype).reshape(64,4,4,4)
    matrix = op.forward(basis).flatten(1).T
    y = torch.randn_like(op.forward(x))
    actual = op.adjoint(y)
    expected = (y.flatten(1)@matrix).reshape_as(x)
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-5)
    lhs, rhs = (op.forward(x)*y).sum(), (x*actual).sum()
    assert abs(float(lhs))>1e-3
    assert float(abs(lhs-rhs)/abs(lhs))<1e-5
    gradient, = torch.autograd.grad(lhs,x)
    torch.testing.assert_close(gradient,actual)
    # A small central object must project into the central detector footprint.
    impulse = torch.zeros_like(x); impulse[:,1:3,1:3,1:3] = 1
    projection = op.forward(impulse)
    assert (projection>=0).all() and projection.sum()>0
    assert projection[:,2:4,:,2:4].sum()>projection[:,:1,:,:].sum()
    op.close()

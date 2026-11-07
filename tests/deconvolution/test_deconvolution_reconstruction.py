import numpy as np
import pytest
from inv_framework.deconvolution.operators import Convolution
from inv_framework.deconvolution.api import run_deconvolution_task


def direct_convolution(x, psf, origin, boundary):
    out = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        for j in np.ndindex(psf.shape):
            source = tuple(a-b+o for a,b,o in zip(i,j,origin))
            if boundary == 'periodic':
                source = tuple(a%n for a,n in zip(source,x.shape))
            elif any(a<0 or a>=n for a,n in zip(source,x.shape)):
                continue
            out[i] += psf[j]*x[source]
    return out


@pytest.mark.parametrize('shape,kshape,origin', [((5,6),(2,3),(0,2)), ((3,4,5),(2,2,3),(1,0,1))])
@pytest.mark.parametrize('boundary', ['zero','periodic'])
def test_convolution_and_adjoint(shape,kshape,origin,boundary):
    rng = np.random.default_rng(2)
    psf = rng.uniform(size=kshape)
    x,y = rng.normal(size=shape),rng.normal(size=shape)
    op=Convolution(shape,psf,origin,tuple(range(len(shape))),boundary)
    np.testing.assert_allclose(op.forward(x), direct_convolution(x,psf,origin,boundary), atol=1e-12)
    np.testing.assert_allclose(np.vdot(op.forward(x), y),np.vdot(x,op.adjoint(y)),atol=1e-12)


def test_wiener_independent_quadratic_solve_and_channels():
    rng=np.random.default_rng(4); shape=(5,6)
    psf=np.array([[.1,.3],[.2,.4]]); origin=(0,1)
    n=np.prod(shape); basis=np.eye(n).reshape((n,)+shape)
    h=np.stack([direct_convolution(v,psf,origin,'periodic').ravel() for v in basis],1)
    l=np.stack([(4*v-np.roll(v,1,0)-np.roll(v,-1,0)-np.roll(v,1,1)-np.roll(v,-1,1)).ravel() for v in basis],1)
    y=rng.normal(size=shape+(2,)); balance=.08
    request=dict(algorithm='wiener_hunt',inputs=dict(observation=y,psf=psf),
        model=dict(axes=['y','x','channel'],boundary='periodic',psf_origin=list(origin)),params=dict(balance=balance))
    result=run_deconvolution_task(request)
    assert result.status=='success',result.error
    for c in range(2):
        direct=np.linalg.solve(h.T@h+balance*l.T@l,h.T@y[...,c].ravel())
        np.testing.assert_allclose(result.outputs['reconstruction'][...,c].ravel(),direct,atol=1e-11)
    assert result.iterations == 0 and result.stopping_reason == 'completed_valid'
    request['model']['boundary']='zero'
    assert run_deconvolution_task(request).status=='failed'


@pytest.mark.parametrize('boundary', ['zero','periodic'])
def test_rl_update_gain_background_sensitivity_likelihood_recovery(boundary):
    rng=np.random.default_rng(5); shape=(9,11)
    truth=np.ones(shape); truth[2:6,3:8]=8
    psf=np.array([[.1,.4,.2],[.3,.1,.2]]); origin=(1,0)
    n=truth.size; basis=np.eye(n).reshape((n,)+shape)
    h=np.stack([direct_convolution(v,psf,origin,boundary).ravel() for v in basis],1)
    gain,background=2.,.7
    y=(gain*h@truth.ravel()+background).reshape(shape)
    initial=rng.uniform(.5,1.5,shape)
    req=dict(algorithm='richardson_lucy',inputs=dict(observation=y,psf=psf,initial_estimate=initial),
        model=dict(axes=['y','x'],boundary=boundary,psf_origin=list(origin),gain=gain,background=background),params=dict(max_iter=1,tolerance=0))
    result=run_deconvolution_task(req)
    assert result.status=='success',result.error
    expected=initial.ravel()*(h.T@(y.ravel()/(gain*h@initial.ravel()+background)))/(h.T@np.ones(n))
    np.testing.assert_allclose(result.outputs['reconstruction'].ravel(),expected,atol=1e-12)
    assert not result.converged and result.stopping_reason=='max_iterations'
    req['params']['max_iter']=80
    result=run_deconvolution_task(req)
    assert min(np.diff(result.diagnostics['poisson_nll_history'])) <= 0
    assert max(np.diff(result.diagnostics['poisson_nll_history'])) < 1e-9
    x=result.outputs['reconstruction']
    assert x.min()>=0
    assert np.linalg.norm(x-truth)<np.linalg.norm(initial-truth)*.4

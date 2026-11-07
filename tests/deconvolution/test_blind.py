import numpy as np
import pytest
from inv_framework.deconvolution.blind import dark_channel, scatter, resize_centered, latent_l0
from inv_framework.deconvolution.operators import Convolution
from inv_framework.experiments import resolve_parameters
from inv_framework.deconvolution.api import run_deconvolution_task


def test_compact_kernel_least_squares():
    from inv_framework.deconvolution.blind import estimate_kernel
    x=np.random.default_rng(5).uniform(size=(20,24))
    k=np.zeros((3,3));k[1]=[.2,.5,.3]
    y=sum(k[i,j]*np.roll(x,(i-1,j-1),(0,1)) for i,j in np.ndindex(k.shape))
    initial=np.ones((3,3))/9
    p=resolve_parameters('deconvolution','blind_dcp',dict(kernel_ridge=1e-8))
    fitted,_=estimate_kernel(x,y,initial,p,True)
    np.testing.assert_allclose(fitted,k,atol=1e-5)


def test_dark_positions_transpose_and_periodic_edges():
    x = np.random.default_rng(3).uniform(size=(5,7,3))
    d,idx = dark_channel(x,3)
    np.testing.assert_equal(d,x.ravel()[idx])
    for i,j in np.ndindex(d.shape):
        expected=min(x[(i+a)%5,(j+b)%7,c] for a in (-1,0,1) for b in (-1,0,1) for c in range(3))
        assert d[i,j]==expected
    v = np.random.default_rng(4).normal(size=d.shape)
    np.testing.assert_allclose(np.vdot(d,v),np.vdot(x,scatter(v,idx,x.shape)))


def test_scale_displacements_and_l0_subproblem():
    k = np.zeros((5,5));k[2,3]=1
    up = resize_centered(k,(9,9),(2,2))
    assert np.unravel_index(up.argmax(),up.shape)==(4,6)
    y=np.random.default_rng(2).normal(size=(12,14))
    delta=np.zeros((3,3));delta[1,1]=1
    op=Convolution(y.shape,delta,(1,1),(0,1),'periodic')
    p=resolve_parameters('deconvolution','blind_l0',dict(epsilon_levels=1,epsilon_initial=100.))
    x,_=latent_l0(op,y,y,p)
    from inv_framework.core.numpy_math import gradient,gradient_adjoint
    np.testing.assert_allclose(x+p['lam']/100**2*gradient_adjoint(gradient(x,(0,1)),(0,1)),y,atol=1e-12)


@pytest.mark.parametrize('algorithm',['blind_dcp','blind_l0'])
def test_observation_only_multiscale(algorithm):
    x=np.zeros((24,28));x[5:18,7:21]=.8;x[10:14,3:10]=.4
    k=np.zeros((7,7));k[3,2:5]=1/3
    # Independent synthesis; neither x nor k is passed to reconstruction.
    y=sum(k[i,j]*np.roll(x,(i-3,j-3),(0,1)) for i,j in np.ndindex(k.shape))
    if algorithm=='blind_dcp':
        y=y[...,None]
    p=dict(max_iter=2,coarse_kernel_size=3,final_max_iter=10)
    if algorithm=='blind_dcp': p.update(patch_size=3,latent_iter=4)
    r=run_deconvolution_task(dict(algorithm=algorithm,task_type='blind_deconvolution',inputs=dict(observation=y),
        model=dict(axes=['y','x','channel'] if y.ndim==3 else ['y','x'],kernel_shape=[7,7],boundary='periodic'),params=p))
    assert r.status=='success',r.error
    assert r.outputs['reconstruction'].shape==y.shape
    estimated=r.outputs['psf'];assert estimated.shape==(7,7)
    assert estimated.min()>=0 and abs(estimated.sum()-1)<1e-12
    assert r.diagnostics['scales']>1
    assert np.isfinite(r.diagnostics['data_residual'])

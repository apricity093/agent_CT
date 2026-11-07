import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from inv_framework.deconvolution.regularized import hyper_scalar, poisson_prox
from inv_framework.deconvolution.api import run_deconvolution_task


def test_hyper_scalar_zero_weight():
    v=np.array([-4.,-.3,0.,.3,4.])
    np.testing.assert_array_equal(hyper_scalar(v,0,.8),v)


def test_hyper_laplacian_zero_lam_api():
    from inv_framework.deconvolution.operators import Convolution
    truth=np.ones((20,24));truth[5:15,7:18]=6
    psf=np.array([[0,.1,0],[.1,.6,.1],[0,.1,0]])
    op=Convolution(truth.shape,psf,(1,1),(0,1),'periodic')
    y=op.forward(truth)
    r=run_deconvolution_task(dict(algorithm='hyper_laplacian',
        inputs=dict(observation=y,psf=psf),
        model=dict(axes=['y','x'],boundary='periodic',psf_origin=[1,1],noise_model='gaussian'),
        params=dict(lam=0,max_iter=100)))
    assert r.status=='success',r.error
    x=r.outputs['reconstruction']
    assert np.isfinite(x).all()
    assert np.linalg.norm(op.forward(x)-y)<np.linalg.norm(op.forward(y)-y)
    assert np.linalg.norm(x-truth)<np.linalg.norm(y-truth)
    assert r.diagnostics['regularizer_contribution']==0


def test_scalar_global_minimum_and_poisson_prox():
    for p in [.3,.5,.8]:
        for value in [.01,.3,1.,4.]:
            weight=.2
            observed=hyper_scalar(np.array([value]),weight,p)[0]
            # Independent exhaustive intervals, including the nonsmooth zero candidate.
            edges=np.linspace(0,value,100)
            candidates=[0]+[minimize_scalar(lambda t:.5*(t-value)**2+weight*t**p,bounds=(a,b),method='bounded').x for a,b in zip(edges[:-1],edges[1:])]
            objective=lambda t:.5*(t-value)**2+weight*abs(t)**p
            assert objective(observed)<=min(map(objective,candidates))+1e-10
    y=np.array([0.,2.,10.]);v=np.array([-2.,.5,3.]);rho=.7
    z=poisson_prox(v,y,rho)
    assert z[0]==0
    np.testing.assert_allclose(1-y[1:]/z[1:]+rho*(z[1:]-v[1:]),0,atol=1e-12)


@pytest.mark.parametrize('noise',['gaussian','poisson'])
def test_tv_identity_kkt(noise):
    yy,xx=np.mgrid[:8,:9];truth=2+np.sin(xx)+np.cos(yy)
    y=truth.copy();lam=.003
    r=run_deconvolution_task(dict(algorithm='tv_admm',inputs=dict(observation=y,psf=np.ones((1,1))),model=dict(axes=['y','x'],boundary='periodic',psf_origin=[0,0],noise_model=noise),params=dict(lam=lam,max_iter=3000,tolerance=1e-9)))
    assert r.status=='success',r.error
    assert r.converged
    x=r.outputs['reconstruction']
    # Independent forward difference and divergence with periodic edges.
    gy=np.roll(x,-1,0)-x;gx=np.roll(x,-1,1)-x
    norm=np.sqrt(gx*gx+gy*gy)
    qx,qy=gx/norm,gy/norm
    dt=np.roll(qx,1,1)-qx+np.roll(qy,1,0)-qy
    data=x-y if noise=='gaussian' else 1-y/x
    np.testing.assert_allclose(data+lam*dt,0,atol=2e-6)


def test_rl_tv_formula_and_undefined_denominator():
    rng=np.random.default_rng(3);x=rng.uniform(1,3,(7,8));y=rng.poisson(5*x).astype(float)
    model=dict(axes=['y','x'],boundary='periodic',psf_origin=[0,0],noise_model='poisson',background=.4)
    params=dict(lam=.03,max_iter=1,tv_epsilon=.01)
    req=dict(algorithm='rl_tv',inputs=dict(observation=y,psf=np.ones((1,1)),initial_estimate=x),model=model,params=params)
    r=run_deconvolution_task(req);assert r.status=='success',r.error
    g=np.stack([np.roll(x,-1,0)-x,np.roll(x,-1,1)-x]);q=g/np.sqrt(np.sum(g*g,axis=0)+.01**2)
    dt=sum(np.roll(v,1,a)-v for a,v in enumerate(q))
    np.testing.assert_allclose(r.outputs['reconstruction'],x*y/(x+.4)/(1+.03*dt),atol=1e-12)
    assert r.outputs['reconstruction'].min()>=0
    req['params']['lam']=100
    assert run_deconvolution_task(req).stopping_reason=='numerical_error'


@pytest.mark.parametrize('algorithm,noise',[('tv_admm','gaussian'),('tv_admm','poisson'),('rl_tv','poisson'),('hyper_laplacian','gaussian')])
def test_representative_restoration(algorithm,noise):
    from inv_framework.deconvolution.operators import Convolution
    truth=np.ones((20,24,2));truth[5:15,7:18]=6
    psf=np.array([[0,.1,0],[.1,.6,.1],[0,.1,0]])
    op=Convolution(truth.shape,psf,(1,1),(0,1),'periodic')
    y=op.forward(truth)
    r=run_deconvolution_task(dict(algorithm=algorithm,inputs=dict(observation=y,psf=psf),model=dict(axes=['y','x','channel'],boundary='periodic',psf_origin=[1,1],noise_model=noise),params=dict(lam=.005,max_iter=300)))
    assert r.status=='success',r.error
    assert np.linalg.norm(r.outputs['reconstruction']-truth)<np.linalg.norm(y-truth)
    assert np.isfinite(r.diagnostics['regularizer_contribution'])

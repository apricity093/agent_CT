import numpy as np
import pytest
from inv_framework.core.numpy_math import gradient, gradient_adjoint
from inv_framework.mri.solvers import Haar
from inv_framework.mri.api import run_mri_task
from inv_framework.mri.operators import fft
from inv_framework.mri.calibration import grappa, espirit


def test_complex_transforms():
    rng=np.random.default_rng(42)
    x=rng.normal(size=(2,8,12))+1j*rng.normal(size=(2,8,12))
    z=rng.normal(size=x.shape)+1j*rng.normal(size=x.shape)
    w=Haar(x.shape,2)
    np.testing.assert_allclose(w.adjoint(w.forward(x)),x,atol=2e-15)
    np.testing.assert_allclose(np.vdot(w.forward(x),z),np.vdot(x,w.adjoint(z)),atol=1e-13)
    g=np.stack([z,2j*z])
    np.testing.assert_allclose(np.vdot(gradient(x,(1,2)),g),np.vdot(x,gradient_adjoint(g,(1,2))),atol=1e-12)


@pytest.mark.parametrize('algorithm',['cs_tv','cs_wavelet'])
def test_cs_full_sampling_optimality(algorithm):
    rng=np.random.default_rng(12); shape=(1,8,8)
    truth=rng.normal(size=shape)+1j*rng.normal(size=shape)
    y=fft(truth[:,None],2);lam=.07
    result=run_mri_task(dict(algorithm=algorithm,inputs=dict(kspace=y,sampling_mask=np.ones_like(y.real),sensitivity_maps=np.ones_like(y)),model=dict(sampling='cartesian',spatial_shape=[8,8],axes=['batch','coil','y','x']),params=dict(lam=lam,max_iter=1500,tolerance=1e-8)))
    assert result.status=='success',result.error
    assert result.converged
    x=result.outputs['reconstruction']
    if algorithm=='cs_wavelet':
        w=Haar(shape,2);c=w.forward(truth)
        expected=w.adjoint(c*np.maximum(1-lam/abs(c),0))
        np.testing.assert_allclose(x,expected,atol=1e-7)
    else:
        g=gradient(x,(1,2));q=g/np.sqrt(np.sum(abs(g)**2,axis=0))
        np.testing.assert_allclose(x-truth+lam*gradient_adjoint(q,(1,2)),0,atol=2e-6)


def test_calibration_equations_and_phase_subspace():
    n=24; yy,xx=np.mgrid[:n,:n]
    object_=np.exp(-((yy-12)**2+(xx-12)**2)/30)*np.exp(.03j*xx)
    maps=np.stack([np.ones((n,n)),np.exp(2j*np.pi*yy/n)])/np.sqrt(2)
    full=fft((maps*object_)[None],2)
    mask=np.zeros_like(full.real);mask[:,:,::2,:]=1
    filled,diag=grappa(full*mask,mask,full,5,1e-8)
    np.testing.assert_array_equal(filled[mask==1],full[mask==1])
    assert np.linalg.norm(filled-full)<np.linalg.norm(full*(1-mask))*.12
    assert max(diag['calibration_relative_residuals'][:5])<1e-5
    assert max(diag['calibration_relative_residuals'])<.4  # one-sided edge patterns
    estimated,values,_=espirit(full,(n,n),5,.02,.9)
    projection=np.sum(estimated.conj()*maps[None],axis=1)
    support=values>.99
    assert support.sum()>100
    np.testing.assert_allclose(abs(projection[support]),1,atol=.03)
    phased=full*np.exp(.7j)
    other,_,_=espirit(phased,(n,n),5,.02,.9)
    # Compare rank-one projectors, independent of arbitrary map gauge.
    np.testing.assert_allclose(estimated[:,:,None]*estimated[:,None,:].conj(),other[:,:,None]*other[:,None,:].conj(),atol=1e-10)


@pytest.mark.parametrize('algorithm',['cs_tv','cs_wavelet'])
def test_weighted_noncartesian_3d_cs(algorithm):
    from inv_framework.mri.operators import SenseEncoding
    rng=np.random.default_rng(9);shape=(4,4,4)
    maps=np.ones((1,2)+shape,complex)/np.sqrt(2)
    trajectory=rng.uniform(-3,3,(100,3))
    op=SenseEncoding(maps,trajectory=trajectory)
    truth=np.ones((1,)+shape)*np.exp(.2j)
    y=op.forward(truth);weights=rng.uniform(.5,2,y.shape)
    result=run_mri_task(dict(algorithm=algorithm,inputs=dict(kspace=y,trajectory=trajectory,sensitivity_maps=maps,data_weights=weights),model=dict(sampling='noncartesian',axes=['batch','coil','z','y','x'],spatial_shape=list(shape),trajectory_units='radians/voxel'),params=dict(lam=.001,max_iter=20)))
    assert result.status=='success',result.error
    assert result.diagnostics['weighted_data_residual']<np.linalg.norm(np.sqrt(weights)*y)*.1

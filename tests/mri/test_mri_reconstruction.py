import numpy as np
import pytest
from inv_framework.mri.operators import SenseEncoding
from inv_framework.mri.api import run_mri_task


def complex_random(rng, shape):
    return rng.normal(size=shape)+1j*rng.normal(size=shape)


def dft(shape, trajectory):
    positions = np.stack(np.meshgrid(*(np.arange(n)-n//2 for n in shape), indexing='ij'), -1)
    return np.exp(-1j*trajectory@positions.reshape(-1, len(shape)).T)/np.sqrt(np.prod(shape))


@pytest.mark.parametrize('shape,coils', [((5,6),1), ((4,5),3), ((3,4,3),2)])
@pytest.mark.parametrize('noncart', [False, True])
def test_fourier_direct_adjoint_and_weighted_sense(shape, coils, noncart):
    rng = np.random.default_rng(23)
    maps = complex_random(rng, (1,coils)+shape)
    grid = np.stack(np.meshgrid(*(2*np.pi*(np.arange(n)-n//2)/n for n in shape), indexing='ij'), -1).reshape(-1,len(shape))
    trajectory = rng.uniform(-2.9,2.9,(2*np.prod(shape),len(shape))) if noncart else grid
    f = dft(shape, trajectory)
    mask = None if noncart else rng.integers(0,2,(1,coils)+shape)
    op = SenseEncoding(maps, mask, trajectory if noncart else None)
    x = complex_random(rng, (1,)+shape)
    expected = np.stack([f@(maps[0,c]*x[0]).ravel() for c in range(coils)])[None]
    if not noncart:
        expected = expected.reshape((1,coils)+shape)*mask
    np.testing.assert_allclose(op.forward(x), expected, atol=2e-3 if noncart else 1e-12, rtol=2e-3 if noncart else 1e-12)
    y = complex_random(rng, expected.shape)
    np.testing.assert_allclose(np.vdot(op.forward(x), y), np.vdot(x, op.adjoint(y)), atol=1e-10)
    # Matrix columns use independent direct DFT, not the tested forward routine.
    matrix = np.concatenate([f*maps[0,c].ravel()[None] for c in range(coils)])
    if not noncart:
        matrix *= mask.ravel()[:,None]
    weights = rng.uniform(.3, 1.4, expected.shape)
    lam = .7
    model = dict(sampling='noncartesian' if noncart else 'cartesian',
                 axes=['batch','coil']+(['y','x'] if len(shape)==2 else ['z','y','x']), spatial_shape=list(shape))
    inputs = dict(kspace=y, sensitivity_maps=maps, data_weights=weights)
    if noncart:
        inputs['trajectory'] = trajectory/(2*np.pi)
        model['trajectory_units'] = 'cycles/voxel'
    else:
        inputs['sampling_mask'] = mask
    result = run_mri_task(dict(algorithm='cg_sense', inputs=inputs, model=model,
                              params=dict(lam=lam, max_iter=200, tolerance=1e-10)))
    assert result.status == 'success', result.error
    assert result.converged
    observed = y.ravel() if noncart else (y*mask).ravel()
    direct = np.linalg.solve(matrix.conj().T@(weights.ravel()[:,None]*matrix)+lam*np.eye(np.prod(shape)),
                             matrix.conj().T@(weights.ravel()*observed))
    np.testing.assert_allclose(result.outputs['reconstruction'].ravel(), direct, atol=2e-3 if noncart else 1e-9, rtol=2e-3 if noncart else 1e-9)


def test_zero_filled_coil_combination_and_sampled_zeros():
    rng = np.random.default_rng(3)
    x = complex_random(rng, (2,5,7))
    maps = np.broadcast_to(np.array([1, 2j])[None,:,None,None], (2,2,5,7)).copy()
    k = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(maps*x[:,None], axes=(-2,-1)), axes=(-2,-1), norm='ortho'),axes=(-2,-1))
    request = dict(algorithm='zero_filled', inputs=dict(kspace=k, sensitivity_maps=maps,sampling_mask=np.ones(k.shape)),
                   model=dict(sampling='cartesian',spatial_shape=[5,7], axes=['batch','coil','y','x']))
    result = run_mri_task(request)
    np.testing.assert_allclose(result.outputs['reconstruction'], x, atol=1e-12)
    assert result.iterations == 0 and not result.converged
    request['params'] = dict(coil_combination='rss')
    result = run_mri_task(request)
    np.testing.assert_allclose(result.outputs['reconstruction'], np.sqrt(5)*abs(x), atol=1e-12)
    request['algorithm']='cg_sense'; request['params']={}
    request['inputs']['kspace']=np.zeros_like(k)
    result=run_mri_task(request)
    assert result.converged and result.iterations == 0
    np.testing.assert_array_equal(result.outputs['reconstruction'], 0)

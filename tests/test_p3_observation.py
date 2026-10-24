import numpy as np
import pytest
import torch
from scipy.optimize import brentq
from inv_framework.measurement import simulate_observation
from inv_framework.fidelity import QuadraticFidelity, TransmissionPoissonFidelity


CUDA_DEVICE = pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))

@pytest.mark.parametrize('device', ['cpu', CUDA_DEVICE])
@pytest.mark.parametrize('model', ['gaussian', 'poisson', 'poisson_gaussian'])
def test_noise(device, model):
    p = torch.full((1, 200, 200), .5, device=device, dtype=torch.float64)
    args = dict(model=model, seed=17, sigma=.2, I0=20., sigma_e=0. if model=='poisson' else 3.)
    y,a,m = simulate_observation(p, **args)
    yy,aa,mm = simulate_observation(p, **args)
    assert torch.equal(y, yy) and m==mm
    assert all(torch.equal(a[k],aa[k]) for k in a)
    assert y.device == p.device
    if model=='gaussian':
        assert abs(float((y-p).mean())) < .004
        assert abs(float((y-p).var())-.04) < .002
    else:
        raw = a['counts' if model=='poisson' else 'c_obs']
        mu=20*np.exp(-.5)
        assert abs(float(raw.mean())-mu) < .1
        assert abs(float(raw.var())-(mu+(9 if model=='poisson_gaussian' else 0))) < .5
        assert torch.allclose(y,-torch.log(raw.clamp_min(1)/20))
        assert 'counts' not in a if model=='poisson_gaussian' else bool((raw>=0).all())


@pytest.mark.parametrize('device', ['cpu', CUDA_DEVICE])
def test_poisson_reference(device):
    c=torch.tensor([0., 1., 8., 100.],dtype=torch.float64,device=device)
    v=torch.tensor([-3., 0., 2., 4.],dtype=c.dtype,device=device)
    f=TransmissionPoissonFidelity(c,20.,scale=.7)
    z=f.proximal(v,.3)
    ref=[brentq(lambda x: x-vi+.3*.7*(ci-20*np.exp(-.7*x)), -50,100,xtol=1e-13)
         for ci,vi in zip(c.cpu().numpy(),v.cpu().numpy())]
    assert np.max(np.abs(z.cpu().numpy()-ref)) < 1e-9
    assert float((z-v+.3*f.gradient(z)).abs().max()) < 1e-9
    assert torch.allclose(f.dual_prox(v,.3),v-.3*f.proximal(v/.3,1/.3),atol=1e-10)
    d=torch.tensor([.1,-.2,.3,-.4],dtype=c.dtype,device=device)
    fd=(f.value(v+1e-6*d)-f.value(v-1e-6*d))/2e-6
    assert torch.allclose(fd,(f.gradient(v)*d).sum(),rtol=1e-7)
    with pytest.raises(RuntimeError,match='inner_max'):
        TransmissionPoissonFidelity(c,20.,max_iterations=1).proximal(v,.3)


@pytest.mark.parametrize('device', ['cpu', CUDA_DEVICE])
def test_quadratic_whitening(device):
    y=torch.tensor([1.,2.,3.],device=device,dtype=torch.float64)
    w=torch.tensor([.2,1.,4.],device=device,dtype=y.dtype)
    x=y-.4
    f=QuadraticFidelity(y,w)
    assert torch.allclose(f.value(x),.5*((w.sqrt()*x-w.sqrt()*y)**2).sum())
    assert torch.allclose(f.gradient(x),w.sqrt()*(w.sqrt()*x-w.sqrt()*y))
    # Gaussian NLL regularization lambda_g corresponds to L2 lambda_g*sigma².
    A=torch.tensor([[1.,.2],[.3,2.],[1.,1.]],device=device,dtype=y.dtype)
    sigma=.3; lam=.7
    gaussian=torch.linalg.solve(A.T@A/sigma**2+lam*torch.eye(2,device=device,dtype=y.dtype),A.T@y/sigma**2)
    l2=torch.linalg.solve(A.T@A+lam*sigma**2*torch.eye(2,device=device,dtype=y.dtype),A.T@y)
    assert torch.allclose(gaussian,l2,atol=1e-12,rtol=0)


def test_zero_negative_and_invalid():
    p=torch.full((1,20,20),20.,dtype=torch.float64)
    y,a,m=simulate_observation(p,model='poisson',I0=1.,seed=3)
    assert bool((a['counts']==0).all()) and m['clipped_fraction']==1.
    y,a,m=simulate_observation(p,model='poisson_gaussian',I0=1.,sigma_e=10.,seed=3)
    assert bool((a['c_obs']<0).any()) and 'counts' not in a
    for kwargs in [dict(I0=0.),dict(I0=1.,floor=0.),dict(I0=1.,scale=0.)]:
        with pytest.raises(ValueError): simulate_observation(p,model='poisson',seed=3,**kwargs)


def test_public_case_roundtrip_split_staging(tmp_path):
    import json
    import h5py
    from pathlib import Path
    from dataclasses import replace
    from inv_framework.benchmarks.ct_cases import load_ct_case, write_ct_case, restrict_ct_case
    from inverse_agent.cases.staging import stage_ct_case
    root=Path(__file__).resolve().parents[1]/'test'/'data'
    case=load_ct_case('parallel_2d/disk_analytic_32',data_root=root)
    y,arrays,meta=simulate_observation(case.measurement_clean,model='poisson_gaussian',
                                     I0=20.,sigma_e=2.,seed=17)
    metadata=dict(case.metadata)
    metadata['measurement']={**metadata['measurement'],**meta}
    case=replace(case,measurement=y,public_measurement=arrays,metadata=metadata)
    record=write_ct_case(case,tmp_path/'case'); record['path']='case'
    catalog=tmp_path/'catalog.json'
    catalog.write_text(json.dumps(dict(schema_version='1.0',cases=[record])))
    loaded=load_ct_case(case.case_id,data_root=tmp_path)
    subset=restrict_ct_case(loaded,[3,1])
    for key,value in arrays.items():
        assert torch.equal(loaded.public_measurement[key],value)
        assert torch.equal(subset.public_measurement[key],value.index_select(-2,torch.tensor([3,1])))
    with h5py.File(tmp_path/'case'/'arrays.h5','a') as h:
        h.create_dataset('measurement/latent_counts',data=np.ones(y.shape))
        h.create_dataset('measurement/electronic_realization',data=np.ones(y.shape))
    staged=stage_ct_case(catalog,case.case_id,tmp_path/'staged')
    with h5py.File(staged.public_path) as h:
        assert set(h['measurement']) == {'y_observed',*arrays}
        assert not np.any(h['truth/x'][...])
        for key,value in arrays.items():
            assert np.array_equal(h['measurement/'+key][...],value.numpy())


@pytest.mark.parametrize('device', ['cpu', CUDA_DEVICE])
def test_observation_contract_repairs(device):
    f=TransmissionPoissonFidelity(torch.tensor([0,1,8],device=device),20.5)
    assert bool((f.I0 == 20.5).all())
    q=QuadraticFidelity(torch.zeros(2,3,device=device),0.)
    assert bool((q.dual_prox(torch.ones_like(q.observed),.3)==0).all())
    with pytest.raises(ValueError,match='shape'):
        q.gradient(torch.ones(2,1,device=device))
    c=torch.tensor([0.,1.,8.,100.],device=device)
    f=TransmissionPoissonFidelity(c,20.,scale=.7)
    v=torch.tensor([-3.,0.,2.,4.],device=device)
    z=f.proximal(v,.3)
    assert bool(((z-v+.3*f.gradient(z)).abs() <= f.tolerance*(1+v.abs())).all())
    with pytest.raises(ValueError,match='precision'):
        TransmissionPoissonFidelity(c,20.,tolerance=1e-10)

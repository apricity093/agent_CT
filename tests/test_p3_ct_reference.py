import json
import numpy as np
import pytest
import torch
from scipy.optimize import lsq_linear,minimize
from inv_framework.operators.ct.radon_torch import ParallelBeamRadon2D
from inv_framework.fidelity import QuadraticFidelity,TransmissionPoissonFidelity
from inv_framework.measurement import simulate_observation
from inv_framework.regularizers import TikhonovRegularizer
from inv_framework.solvers import FISTASolver,ADMMSolver,PDHGSolver

CUDA_DEVICE = pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))

@pytest.mark.parametrize('device',['cpu', CUDA_DEVICE])
@pytest.mark.parametrize('algorithm,model',[('fista','wls'),('admm','wls'),('pdhg','wls'),('fista','gaussian_nll'),('pdhg','transmission_poisson_nll')])
def test_ct_reference(tmp_path,device,algorithm,model):
    angles=torch.linspace(0,np.pi,13,device=device)[:-1]
    op=ParallelBeamRadon2D(8,angles=angles,device=device)
    eye=torch.eye(64,device=device).reshape(64,1,8,8)
    a=op.forward(eye).flatten(1).T.double().cpu().numpy()
    truth=torch.zeros(1,1,8,8,device=device);truth[...,2:6,2:6]=.3
    clean=op.forward(truth)
    lam=10.
    if model=='transmission_poisson_nll':
        y,public,meta=simulate_observation(clean,model='poisson',I0=20.,seed=17)
        fidelity=TransmissionPoissonFidelity(public['counts'],public['I0'])
        c=public['counts'].flatten().double().cpu().numpy()
        objective=lambda x:np.sum(20*np.exp(-a@x)+c*(a@x))+lam/2*(x@x)
        gradient=lambda x:a.T@(c-20*np.exp(-a@x))+lam*x
        reference=minimize(objective,np.zeros(64),jac=gradient,bounds=[(0,None)]*64,method='L-BFGS-B',options={'ftol':1e-14,'gtol':1e-10,'maxiter':10000})
        assert reference.success,reference.message
        ref=reference.x;denominator=max(1.,np.linalg.norm(a.T@c))
    else:
        y=clean+.05*torch.cos(torch.arange(clean.numel(),device=device).reshape(clean.shape))
        weights=torch.full_like(y,4.) if model=='gaussian_nll' else torch.linspace(.5,2.,y.numel(),device=device).reshape(y.shape)
        fidelity=QuadraticFidelity(y,weights)
        yy=y.flatten().double().cpu().numpy();w=weights.flatten().double().cpu().numpy()
        matrix=np.vstack([w[:,None]**.5*a,np.sqrt(lam)*np.eye(64)])
        target=np.r_[w**.5*yy,np.zeros(64)]
        reference=lsq_linear(matrix,target,bounds=(0,np.inf),tol=1e-12,max_iter=10000)
        assert reference.success
        ref=reference.x
        objective=lambda x:.5*np.sum(w*(a@x-yy)**2)+lam/2*(x@x)
        gradient=lambda x:a.T@(w*(a@x-yy))+lam*x
        denominator=max(1.,np.linalg.norm(a.T@(w*yy)))
    cls={'fista':FISTASolver,'admm':ADMMSolver,'pdhg':PDHGSolver}[algorithm]
    args={'cg_max_iterations':200} if algorithm=='admm' else {}
    result=cls(TikhonovRegularizer(),strength=lam,num_iterations=5000,tolerance=1e-6,min_value=0,**args).solve_detailed(y,op,fidelity=fidelity)
    x=result.reconstruction.flatten().double().cpu().numpy()
    relative_error=np.linalg.norm(x-ref)/max(np.linalg.norm(ref),1e-12)
    kkt=np.linalg.norm(x-np.maximum(x-gradient(x),0))/denominator
    objective_error=abs(result.final_objective-objective(x))/max(1,abs(objective(x)))
    record=dict(device=str(result.reconstruction.device),status=result.status,iterations=result.actual_iterations,relative_error=relative_error,kkt=kkt,objective_error=objective_error,resources=result.resources)
    (tmp_path/'evidence.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(algorithm,model,record)
    assert result.status=='converged',record
    assert str(result.reconstruction.device).startswith(device)
    assert relative_error<=1e-4,record
    assert kkt<=1e-5,record
    assert objective_error<=1e-5,record
    assert x.min()>=0

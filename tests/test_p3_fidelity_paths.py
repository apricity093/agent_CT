import numpy as np
import pytest
import torch
from test_p1_composable import Matrix
from inv_framework.fidelity import QuadraticFidelity, TransmissionPoissonFidelity
from inv_framework.solvers import FISTASolver, ADMMSolver, PDHGSolver
from inv_framework.regularizers import TikhonovRegularizer

CUDA_DEVICE = pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))

@pytest.mark.parametrize('device',['cpu', CUDA_DEVICE])
@pytest.mark.parametrize('solver',[FISTASolver,ADMMSolver,PDHGSolver])
def test_weighted_public_solver(device,solver):
    a=np.diag([1.1,1.2,1.3,1.4]); y=np.array([-1.,.4,-.3,1.2]); w=np.array([0.,.5,2.,3.]); lam=.17
    tensor=lambda x: torch.tensor(x,dtype=torch.float64,device=device)
    f=QuadraticFidelity(tensor(y)[None],tensor(w)[None])
    instance=solver(TikhonovRegularizer(),strength=lam,num_iterations=3000,tolerance=1e-8)
    r=instance.solve_detailed(tensor(y)[None],Matrix(tensor(a),(1,2,2)),fidelity=f)
    x=r.reconstruction.flatten().cpu().numpy()
    ref=np.linalg.solve(a.T@np.diag(w)@a+lam*np.eye(4),a.T@(w*y))
    assert r.status=='converged',(r.status,r.stopping_reason)
    assert np.linalg.norm(x-ref)<1e-6
    assert abs(r.final_objective-(.5*np.sum(w*(a@x-y)**2)+lam/2*np.sum(x*x)))<1e-9

@pytest.mark.parametrize('device',['cpu', CUDA_DEVICE])
def test_poisson_public_solver(device):
    from scipy.optimize import minimize
    a=np.diag([1.1,1.2,1.3,1.4]); c=np.array([0.,1.,8.,12.]); lam=.17
    tensor=lambda x: torch.tensor(x,dtype=torch.float64,device=device)
    f=TransmissionPoissonFidelity(tensor(c)[None],20.,tolerance=1e-12)
    r=PDHGSolver(TikhonovRegularizer(),strength=lam,num_iterations=5000,tolerance=1e-8).solve_detailed(tensor(c)[None],Matrix(tensor(a),(1,2,2)),fidelity=f)
    obj=lambda x: np.sum(20*np.exp(-a@x)+c*(a@x))+lam/2*(x@x)
    jac=lambda x: a.T@(c-20*np.exp(-a@x))+lam*x
    ref=minimize(obj,np.zeros(4),jac=jac,method='BFGS',options={'gtol':1e-10})
    x=r.reconstruction.flatten().cpu().numpy()
    assert r.status=='converged',(r.status,r.stopping_reason)
    assert np.linalg.norm(x-ref.x)<1e-6
    assert np.linalg.norm(jac(x))<1e-6
    assert abs(r.final_objective-obj(x))<1e-9

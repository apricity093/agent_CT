import numpy as np
import pytest
import torch
from scipy.optimize import minimize
from test_p1_composable import Matrix
from inv_framework.operators.gradient import GradientOperator
from inv_framework.regularizers import TikhonovRegularizer, TVRegularizer, HaarL1Regularizer, GradientHuberRegularizer
from inv_framework.solvers import ADMMSolver, PDHGSolver
from inv_framework.solvers.splitting import Penalty

CUDA_DEVICE = pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))

@pytest.mark.parametrize('device',['cpu', CUDA_DEVICE])
@pytest.mark.parametrize('model',['gaussian_nll','wls','transmission_poisson_nll'])
@pytest.mark.parametrize('solver',[ADMMSolver,PDHGSolver])
@pytest.mark.parametrize('prior',['identity','h1','huber','haar','tv','anisotropic'])
@pytest.mark.parametrize('nonnegative',[False,True])
def test_reference(solver,prior,nonnegative,model,device):
    from inv_framework.fidelity import QuadraticFidelity,TransmissionPoissonFidelity
    if model=='transmission_poisson_nll' and (solver!=PDHGSolver or not nonnegative):
        pytest.skip('NLL support matrix uses nonnegative PDHG')
    a=np.diag([1.1,1.2,1.3,1.4]); y=np.array([-1.,.4,-.3,1.2]); weight=.17
    d=np.zeros((2,4,4))
    for j in range(4):
        b=np.eye(4)[j].reshape(2,2)
        d[0,:2,j]=b[1]-b[0]; d[1,[0,2],j]=b[:,1]-b[:,0]
    d=d.reshape(8,4)
    w=np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]])/2
    regs={'identity':TikhonovRegularizer(),'h1':TikhonovRegularizer(GradientOperator((1,2,2))),
        'huber':GradientHuberRegularizer(.2),'haar':HaarL1Regularizer(1),
        'tv':TVRegularizer(),'anisotropic':TVRegularizer(mode='anisotropic')}
    t=np.eye(4) if prior=='identity' else w[1:] if prior=='haar' else d
    def pen(z):
        if prior in ('identity','h1'): return .5*(z@z)
        if prior=='huber': return np.where(abs(z)<=.2,z*z/.4,abs(z)-.1).sum()
        return np.linalg.norm(z.reshape(2,4),axis=0).sum() if prior=='tv' else abs(z).sum()
    weights=np.array([.5,1.,2.,3.]) if model=='wls' else np.full(4,4.)
    counts=np.array([1.,3.,8.,12.])
    def data(x):
        return np.sum(20*np.exp(-a@x)+counts*(a@x)) if model=='transmission_poisson_nll' else .5*np.sum(weights*(a@x-y)**2)
    def obj(x): return data(x)+weight*pen(t@x)
    bounds=[(0,None)]*4 if nonnegative else [(None,None)]*4
    if prior in ('tv','anisotropic','haar'):
        m=4 if prior=='tv' else t.shape[0]
        def cons(v):
            z=t@v[:4]
            return v[4:]-np.linalg.norm(z.reshape(2,4),axis=0) if prior=='tv' else np.r_[v[4:]-z,v[4:]+z]
        ref=minimize(lambda v:data(v[:4])+weight*v[4:].sum(),np.r_[np.zeros(4),np.ones(m)],
            method='SLSQP',bounds=bounds+[(0,None)]*m,constraints={'type':'ineq','fun':cons},options={'ftol':1e-12,'maxiter':2000})
    else:
        ref=minimize(obj,np.zeros(4),method='L-BFGS-B',bounds=bounds,options={'ftol':1e-15,'gtol':1e-10})
    if prior in ("identity","h1") and not nonnegative and model!="transmission_poisson_nll":
        ref.x=np.linalg.solve(a.T@np.diag(weights)@a+weight*t.T@t,a.T@(weights*y))
    assert ref.success
    instance=solver(regs[prior],strength=weight,num_iterations=2000,tolerance=1e-8,min_value=0 if nonnegative else None)
    f=TransmissionPoissonFidelity(torch.tensor(counts,device=device)[None],20.,tolerance=1e-12) if model=='transmission_poisson_nll' else QuadraticFidelity(torch.tensor(y,device=device)[None],torch.tensor(weights,device=device)[None])
    r=instance.solve_detailed(torch.tensor(y,device=device)[None],Matrix(torch.tensor(a,device=device),(1,2,2)),fidelity=f)
    x=r.reconstruction.flatten().cpu().numpy()
    assert r.status=='converged',(r.status,r.stopping_reason)
    assert r.reconstruction.device == torch.tensor(y,device=device).device
    dual=instance.endpoint_state['d'].flatten().cpu().numpy()
    if prior=='haar': dual=dual[1:]
    z=t@x
    g=a.T@(counts-20*np.exp(-a@x) if model=="transmission_poisson_nll" else weights*(a@x-y))+t.T@dual
    kkt=np.linalg.norm(x-np.maximum(x-g,0)) if nonnegative else np.linalg.norm(g)
    fenchel=abs(weight*pen(z)-dual@z) if prior in ('tv','anisotropic','haar') else np.linalg.norm(dual-weight*(z if prior in ('identity','h1') else np.clip(z/.2,-1,1)))
    print('device',r.reconstruction.device,'independent_kkt',kkt,'subgradient_error',fenchel)
    assert kkt<1e-6
    assert fenchel<1e-6
    print(solver.__name__,prior,nonnegative,"solution_error",np.linalg.norm(x-ref.x[:4]),"objective_error",abs(r.final_objective-obj(x)),"residual",r.metadata["split_residual"],"cost",r.resources)
    assert np.linalg.norm(x-ref.x[:4])<2e-5
    assert abs(r.final_objective-obj(x))<1e-10
    if nonnegative: assert x.min()>=0

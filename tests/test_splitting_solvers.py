import numpy as np
import pytest
import torch
from scipy.optimize import minimize
from test_composable_regularizers import Matrix
from inv_framework.core.operator import GradientOperator
from inv_framework.core.regularizer import TikhonovRegularizer, TVRegularizer, HaarL1Regularizer, GradientHuberRegularizer
from inv_framework.solvers import ADMMSolver, PDHGSolver
from inv_framework.solvers.splitting import Penalty

@pytest.mark.parametrize('solver',[ADMMSolver,PDHGSolver])
@pytest.mark.parametrize('prior',['identity','h1','huber','haar','tv','anisotropic'])
@pytest.mark.parametrize('nonnegative',[False,True])
def test_reference(solver,prior,nonnegative):
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
    def obj(x): return .5*np.sum((a@x-y)**2)+weight*pen(t@x)
    bounds=[(0,None)]*4 if nonnegative else [(None,None)]*4
    if prior in ('tv','anisotropic','haar'):
        m=4 if prior=='tv' else t.shape[0]
        def cons(v):
            z=t@v[:4]
            return v[4:]-np.linalg.norm(z.reshape(2,4),axis=0) if prior=='tv' else np.r_[v[4:]-z,v[4:]+z]
        ref=minimize(lambda v:.5*np.sum((a@v[:4]-y)**2)+weight*v[4:].sum(),np.r_[np.zeros(4),np.ones(m)],
            method='SLSQP',bounds=bounds+[(0,None)]*m,constraints={'type':'ineq','fun':cons},options={'ftol':1e-12,'maxiter':2000})
    else:
        ref=minimize(obj,np.zeros(4),method='L-BFGS-B',bounds=bounds,options={'ftol':1e-15,'gtol':1e-10})
    if prior in ("identity","h1") and not nonnegative:
        ref.x=np.linalg.solve(a.T@a+weight*t.T@t,a.T@y)
    assert ref.success
    instance=solver(regs[prior],strength=weight,num_iterations=2000,tolerance=1e-8,min_value=0 if nonnegative else None)
    r=instance.solve_detailed(torch.tensor(y)[None],Matrix(torch.tensor(a),(1,2,2)))
    x=r.reconstruction.flatten().cpu().numpy()
    assert r.status=='converged',(r.status,r.stopping_reason)
    assert r.reconstruction.device == torch.tensor(y).device
    dual=instance.endpoint_state['d'].flatten().cpu().numpy()
    if prior=='haar': dual=dual[1:]
    z=t@x
    g=a.T@(a@x-y)+t.T@dual
    kkt=np.linalg.norm(x-np.maximum(x-g,0)) if nonnegative else np.linalg.norm(g)
    fenchel=abs(weight*pen(z)-dual@z) if prior in ('tv','anisotropic','haar') else np.linalg.norm(dual-weight*(z if prior in ('identity','h1') else np.clip(z/.2,-1,1)))
    print('device',r.reconstruction.device,'independent_kkt',kkt,'subgradient_error',fenchel)
    assert kkt<1e-6
    assert fenchel<1e-6
    print(solver.__name__,prior,nonnegative,"solution_error",np.linalg.norm(x-ref.x[:4]),"objective_error",abs(r.final_objective-obj(x)),"residual",r.metadata["split_residual"],"cost",r.resources)
    assert np.linalg.norm(x-ref.x[:4])<2e-5
    assert abs(r.final_objective-obj(x))<1e-10
    if nonnegative: assert x.min()>=0

def test_huber():
    reg=GradientHuberRegularizer(.2); v=torch.tensor([-.6,-.2,0.,.2,.6],dtype=torch.float64); t=.3
    z=reg.coefficient_proximal(v,t)
    torch.testing.assert_close(z+t*(z/.2).clamp(-1,1),v)
    p=Penalty(reg,(1,2,2)); torch.testing.assert_close(p.dual_prox(v,2.,.7)+2*p.prox(v/2,.7/2),v)
    x=torch.randn(1,1,2,2,dtype=torch.float64,requires_grad=True)
    torch.testing.assert_close(torch.autograd.grad(reg.value(x).sum(),x)[0],reg.gradient(x))
    for delta in (0,-1,float('inf'),float('nan')):
        with pytest.raises(ValueError): GradientHuberRegularizer(delta)



def test_signed_matrix_bound():
    from inv_framework.solvers.splitting import norm_bound
    a=torch.tensor([[1.,-4.,2.,0.],[-2.,1.,3.,1.]],dtype=torch.float64)
    op=Matrix(a,(1,2,2))
    bound=norm_bound(op,torch.zeros(1,1,2,2,dtype=torch.float64))
    assert bound>=float(torch.linalg.matrix_norm(a,2)**2)

@pytest.mark.parametrize('delta',[.1,1.])
def test_huber_boundary_and_direction(delta):
    r=GradientHuberRegularizer(delta)
    v=torch.tensor([delta-1e-7,delta,delta+1e-7],dtype=torch.float64)
    val=r.coefficient_value(v)
    assert abs(float(val[1])-delta/2)<1e-14
    assert abs(float((val[2]-val[0])/2e-7)-1)<1e-6
    x=torch.randn(1,1,3,3,dtype=torch.float64); direction=torch.randn_like(x); eps=1e-6
    fd=float((r.value(x+eps*direction)-r.value(x-eps*direction))/(2*eps))
    assert abs(fd-float((r.gradient(x)*direction).sum()))<1e-7
    # The proximal solves the coefficient objective with lambda applied exactly once.
    weight=.37; step=.2; z=r.coefficient_proximal(v,step*weight)
    torch.testing.assert_close((z-v)/step+weight*(z/delta).clamp(-1,1),torch.zeros_like(z),atol=1e-14,rtol=0)

@pytest.mark.parametrize('solver',[ADMMSolver,PDHGSolver])
def test_closed_forms(solver):
    op=Matrix(torch.eye(4,dtype=torch.float64),(1,2,2)); y=torch.tensor([[-1.,.4,-.3,1.2]],dtype=torch.float64)
    for nonnegative in (False,True):
        r=solver(strength=.3,num_iterations=1500,tolerance=1e-10,min_value=0 if nonnegative else None).solve_detailed(y,op)
        expected=y/1.3
        if nonnegative: expected=expected.clamp_min(0)
        assert float((r.reconstruction.flatten(1)-expected).norm())<1e-8
    w=torch.tensor([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]],dtype=torch.float64)/2
    c=y@w.T; c[:,1:]=c[:,1:].sign()*(c[:,1:].abs()-.3).clamp_min(0)
    r=solver(HaarL1Regularizer(1),strength=.3,num_iterations=1500,tolerance=1e-10).solve_detailed(y,op)
    assert float((r.reconstruction.flatten(1)-c@w).norm())<1e-8

def test_haar_dual_zero_ll():
    p=Penalty(HaarL1Regularizer(1),(1,2,2))
    z=torch.randn(1,1,2,2,dtype=torch.float64)
    result=p.dual_prox(z,.37,.21)
    assert result[...,0,0].item()==0
    assert float(result.abs().max())<=.21+1e-15
    assert torch.count_nonzero(p.dual_prox(z,.37,0))==0

@pytest.mark.parametrize('solver',[ADMMSolver,PDHGSolver])
def test_outer_limit(solver):
    op=Matrix(torch.eye(4,dtype=torch.float64),(1,2,2))
    result=solver(num_iterations=1,tolerance=0).solve_detailed(torch.ones(1,4,dtype=torch.float64),op)
    assert result.status=='max_iterations' and result.actual_iterations==1
    assert result.metadata['endpoint_diagnostics_complete']


def test_cg_breakdown_and_fista_smooth_huber():
    class BrokenAdjoint(Matrix):
        def adjoint(self,y): return -super().adjoint(y)
    op=BrokenAdjoint(2*torch.eye(4,dtype=torch.float64),(1,2,2))
    y=torch.ones(1,4,dtype=torch.float64)
    r=ADMMSolver().solve_detailed(y,op)
    assert r.status=='numerical_error' and 'cg_nonpositive_curvature' in r.stopping_reason
    from inv_framework.solvers import FISTASolver
    assert FISTASolver(GradientHuberRegularizer(), num_iterations=2).solve_detailed(y,op).status != 'invalid_parameters'


@pytest.fixture(autouse=True, params=["cpu", "cuda"])
def numerical_device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    with torch.device(request.param):
        yield request.param

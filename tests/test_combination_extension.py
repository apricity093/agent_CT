"""Independent small-matrix references for the selected CT compositions."""
import numpy as np
import pytest
import torch
from scipy.optimize import minimize
from inv_framework.fidelity import QuadraticFidelity, TransmissionPoissonFidelity
from inv_framework.regularizers import (
    ZeroRegularizer, TikhonovRegularizer, TVRegularizer,
    GradientHuberRegularizer, HaarL1Regularizer,
)
from inv_framework.operators.gradient import GradientOperator
from inv_framework.solvers import FISTASolver, ADMMSolver, PDHGSolver
from inv_framework.solvers.classical import CGLSSolver, LSQRSolver
from test_p1_composable import Matrix


A = np.array([[1.2,.1,.2,.05],[.1,1.4,.05,.1],[.2,.1,1.1,.1],[.05,.2,.1,1.3]])
D = np.array([[-1,0,1,0],[0,-1,0,1],[0,0,0,0],[0,0,0,0],
              [-1,1,0,0],[0,0,0,0],[0,0,-1,1],[0,0,0,0]])
H = np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]])/2


def prior(name):
    return {
        'none':lambda:ZeroRegularizer(),
        'tikhonov':lambda:TikhonovRegularizer(),
        'h1':lambda:TikhonovRegularizer(GradientOperator((1,2,2))),
        'huber':lambda:GradientHuberRegularizer(.2),
        'tv_iso':lambda:TVRegularizer('isotropic', 3000, 1e-3),
        'tv_aniso':lambda:TVRegularizer('anisotropic', 3000, 1e-3),
        'haar_l1':lambda:HaarL1Regularizer(1),
    }[name]()


def penalty(name, x):
    d = D@x
    return {
        'none':lambda:0., 'tikhonov':lambda:.5*(x@x), 'h1':lambda:.5*(d@d),
        'huber':lambda:np.where(np.abs(d)<=.2,d*d/.4,np.abs(d)-.1).sum(),
        'tv_aniso':lambda:np.abs(d).sum(),
        'tv_iso':lambda:np.sqrt(d[:4]**2+d[4:]**2).sum(),
        'haar_l1':lambda:np.abs((H@x)[1:]).sum(),
    }[name]()


@pytest.mark.parametrize('solver',[FISTASolver,ADMMSolver,PDHGSolver])
@pytest.mark.parametrize('name',['none','tikhonov','h1','huber','tv_iso','tv_aniso','haar_l1'])
@pytest.mark.parametrize('model',['quadratic','poisson'])
@pytest.mark.parametrize('box',[False,True])
@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable'))])
def test_independent_composite_optimum(solver, name, model, box, device):
    tensor=lambda x:torch.tensor(x,dtype=torch.float64,device=device)
    observed=np.array([-.4,.4,.3,1.2]) if model=='quadratic' else np.array([0.,1.,8.,12.])
    weights=np.array([.4,.8,2.,3.])
    fidelity=QuadraticFidelity(tensor(observed)[None],tensor(weights)[None]) if model=='quadratic' else TransmissionPoissonFidelity(tensor(observed)[None],20.)
    bounds=(0.,1.5) if box else ((0.,None) if model=='poisson' else (None,None))
    strength=0. if name=='none' else .17
    data=lambda x:.5*np.sum(weights*(A@x-observed)**2) if model=='quadratic' else np.sum(20*np.exp(-A@x)+observed*(A@x))
    objective=lambda x:data(x)+strength*penalty(name,x)
    reference=minimize(objective,np.full(4,.4),method='Powell',bounds=[bounds]*4,
                       options={'xtol':1e-11,'ftol':1e-12,'maxiter':3000})
    reg=prior(name)
    reg.num_iterations,reg.tolerance=3000,1e-3
    kwargs={'cg_tolerance':1e-8} if solver is ADMMSolver else {}
    result=solver(reg,strength=strength,num_iterations=1800,tolerance=2e-6,
                  min_value=bounds[0],max_value=bounds[1],**kwargs).solve_detailed(
                      tensor(observed)[None],Matrix(tensor(A),(1,2,2)),fidelity=fidelity)
    x=result.reconstruction.flatten().cpu().numpy()
    assert result.status not in {'invalid_parameters','numerical_error'},result.stopping_reason
    assert objective(x)-reference.fun < 3e-5,(result.status,result.stopping_reason,objective(x),reference.fun)
    assert result.final_objective == pytest.approx(objective(x),abs=1e-8)
    if bounds[0] is not None: assert x.min()>=bounds[0]-1e-12
    if bounds[1] is not None: assert x.max()<=bounds[1]+1e-12


@pytest.mark.parametrize('solver',[CGLSSolver,LSQRSolver])
@pytest.mark.parametrize('name',['none','tikhonov','h1'])
def test_augmented_krylov_closed_form(solver,name):
    t=lambda x:torch.tensor(x,dtype=torch.float64)
    y=np.array([-.3,.4,.7,1.2]); w=np.array([0.,.7,2.,3.])
    strength=0. if name=='none' else .17
    L=D if name=='h1' else np.eye(4)
    B=np.concatenate((np.sqrt(w)[:,None]*A,np.sqrt(strength)*L))
    rhs=np.concatenate((np.sqrt(w)*y,np.zeros(L.shape[0])))
    ref=np.linalg.lstsq(B,rhs,rcond=None)[0]
    r=solver(regularizer=name,strength=strength,num_iterations=40,
             fidelity=QuadraticFidelity(t(y)[None],t(w)[None])).solve_detailed(t(y)[None],Matrix(t(A),(1,2,2)))
    x=r.reconstruction.flatten().numpy()
    np.testing.assert_allclose(B@x,B@ref,atol=1e-7)
    assert r.final_objective==pytest.approx(.5*np.sum(w*(A@x-y)**2)+strength/2*np.sum((L@x)**2),abs=1e-8)


def test_joint_prox_box_certificate_and_budget():
    from inv_framework.regularizers.joint_prox import joint_prox, ProximalAccuracyError
    reg=HaarL1Regularizer(1)
    v=torch.tensor([[[[-.9,.4],[1.2,-.3]]]],dtype=torch.float64)
    x=joint_prox(reg,v,.2,min_value=0.,max_value=.7,tolerance=1e-7,max_iterations=5000)
    assert x.min()>=0 and x.max()<=.7
    assert reg.last_prox_error_bound<=reg.last_prox_target
    with pytest.raises(ProximalAccuracyError):
        joint_prox(reg,v,.2,min_value=0.,max_value=.7,tolerance=1e-12,max_iterations=1)


def test_poisson_returned_float32_optimality_is_checked():
    f=TransmissionPoissonFidelity(torch.tensor([[0.,1.,1000.]],dtype=torch.float32),2010.)
    result=f.proximal(torch.tensor([[-3.,0.,2.]],dtype=torch.float32),.3)
    report=f.precision_report()
    assert result.dtype==torch.float32 and report['returned_optimality_satisfied']
    assert report['last_returned_residual']<=report['returned_residual_threshold']


@pytest.mark.skipif(not torch.cuda.is_available(),reason='ASTRA CUDA required')
def test_fdk_link_preserves_original_observations():
    import astra
    from inv_framework.operators.ct import ASTRAFDKOperator3D
    size=16
    operator=ASTRAFDKOperator3D(astra.create_vol_geom(size,size,size),
        astra.create_proj_geom('cone',1.,1.,size,size,np.linspace(0,2*np.pi,24,endpoint=False),48.,16.))
    image=torch.zeros((1,size,size,size),device='cuda')
    image[:,4:12,4:12,4:12]=.02
    observed=operator.forward(image)
    before=observed.clone()
    result=operator.fdk(observed)
    assert torch.equal(observed,before)
    assert torch.isfinite(result).all()

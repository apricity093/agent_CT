"""P4a independent references and native termination contracts."""
import json
import numpy as np
import pytest
import torch
from scipy.optimize import minimize
from test_composable_regularizers import Matrix
from inv_framework.solvers.base import SolveControl
from inv_framework.solvers.splitting import ConstrainedTVSolver

DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))]

@pytest.mark.parametrize('algorithm', ['pdhg','asd_pocs_tv'])
@pytest.mark.parametrize('device', DEVICES)
def test_stopping_control(algorithm, device):
    from inv_framework.ct.backends.radon_torch import ParallelBeamRadon2D
    op=ParallelBeamRadon2D(4,angles=torch.arange(3,device=device)*np.pi/3,device=device)
    y=op.forward(torch.zeros(1,1,4,4,device=device))
    solver=ConstrainedTVSolver(epsilon=.1,epsilon_source='explicit control fixture',algorithm=algorithm,num_iterations=40)
    r=solver.solve_detailed(y,op,control=SolveControl(min_iterations=10,check_every=2,patience=7))
    assert r.actual_iterations==22
    assert r.status==('converged' if algorithm=='pdhg' else 'stalled')
    assert [v.iteration for v in r.trajectory if v.metadata['checked']]==list(range(10,23,2))
    assert r.trajectory[-1].consecutive_criteria_count==7
    r=solver.solve_detailed(y,op,control=SolveControl(max_iterations=25,stop_on_convergence=False))
    assert r.status=='max_iterations' and r.actual_iterations==25
    r=solver.solve_detailed(y,op,callback=lambda row: row.iteration<4)
    assert r.status=='cancelled' and r.actual_iterations==4

@pytest.mark.parametrize('device', DEVICES)
def test_tolerance_override(device):
    a=torch.diag(torch.tensor([1.,1.2,1.4,1.6],dtype=torch.float64,device=device))
    op=Matrix(a,(1,2,2)); y=torch.tensor([[.1,.4,.2,.9]],dtype=a.dtype,device=device)
    solver=ConstrainedTVSolver(epsilon=10,epsilon_source='inactive ball fixture',num_iterations=20,tolerance=1e-12)
    r=solver.solve_detailed(y,op,x_init=torch.tensor([[[[.1,.2],[.3,.4]]]],device=device,dtype=a.dtype),control=SolveControl(tolerance=100,patience=1))
    assert r.actual_iterations==1 and r.status=='converged'
    assert r.trajectory[-1].native_criterion_threshold==100
    r=solver.solve_detailed(y,op,x_init=torch.tensor([[[[.1,.2],[.3,.4]]]],device=device,dtype=a.dtype),control=SolveControl(tolerance=0,patience=1,max_iterations=2))
    assert r.status=='max_iterations'


def independent_reference(a,y,epsilon,mode):
    # Explicit unit-grid 2x2 differences, independent of production D/proximal.
    d=np.array([[-1,0,1,0],[0,-1,0,1],[0,0,0,0],[0,0,0,0],[-1,1,0,0],[0,0,0,0],[0,0,-1,1],[0,0,0,0]],dtype=float)
    def tv(x):
        z=d@x
        return np.linalg.norm(z.reshape(2,4),axis=0).sum() if mode=='isotropic' else abs(z).sum()
    m=4 if mode=='isotropic' else 8
    def cones(v):
        z=d@v[:4]
        return v[4:]-np.linalg.norm(z.reshape(2,4),axis=0) if mode=='isotropic' else np.r_[v[4:]-z,v[4:]+z]
    ref=minimize(lambda v:v[4:].sum(),np.r_[np.linalg.solve(a,y),np.ones(m)],method='SLSQP',bounds=[(0,None)]*(4+m),constraints=[{'type':'ineq','fun':cones},{'type':'ineq','fun':lambda v:epsilon**2-np.sum((a@v[:4]-y)**2)}],options={'ftol':1e-13,'maxiter':4000})
    assert ref.success,ref.message
    assert np.linalg.norm(a@ref.x[:4]-y)<=epsilon+1e-9
    return ref.x[:4],tv




@pytest.mark.parametrize('device', DEVICES)
def test_asd_tolerance_override(device):
    from inv_framework.ct.backends.radon_torch import ParallelBeamRadon2D
    op=ParallelBeamRadon2D(4,angles=torch.arange(3,device=device)*np.pi/3,device=device)
    x=torch.zeros(1,1,4,4,device=device);x[...,1:3,1:3]=.3
    y=op.forward(x)
    solver=ConstrainedTVSolver(epsilon=10,epsilon_source='inactive radius control fixture',algorithm='asd_pocs_tv',num_iterations=2,tv_updates=2,tolerance=1e-12)
    loose=solver.solve_detailed(y,op,control=SolveControl(tolerance=100,patience=1))
    tight=solver.solve_detailed(y,op,control=SolveControl(tolerance=0,patience=1))
    assert loose.status=='stalled' and loose.actual_iterations==1
    assert tight.status=='max_iterations' and tight.actual_iterations==2

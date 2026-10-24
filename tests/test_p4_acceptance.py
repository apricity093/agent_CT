"""P4a independent references and native termination contracts."""
import json
import numpy as np
import pytest
import torch
from scipy.optimize import minimize
from test_p1_composable import Matrix
from inv_framework.solvers.base import SolveControl
from inv_framework.solvers.constrained_tv import ConstrainedTVSolver
from inv_framework.instrumentation import CountingLinearOperator

DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))]

@pytest.mark.parametrize('algorithm', ['pdhg','asd_pocs_tv'])
@pytest.mark.parametrize('device', DEVICES)
def test_stopping_control(algorithm, device):
    from inv_framework.operators.ct.radon_torch import ParallelBeamRadon2D
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
@pytest.mark.parametrize('mode',['isotropic','anisotropic'])
def test_nonconstant_matrix_reference(device,mode):
    a=np.diag([1.,1.2,1.4,1.6]); y=np.array([.1,.6,.35,1.3]); epsilon=.08
    reference,tv=independent_reference(a,y,epsilon,mode)
    base=Matrix(torch.tensor(a,device=device),(1,2,2)); op=CountingLinearOperator(base)
    r=ConstrainedTVSolver(epsilon=epsilon,epsilon_source='explicit nonzero-TV reference radius',tv_mode=mode,num_iterations=20000,tolerance=1e-8,feasibility_tolerance=1e-8).solve_detailed(torch.tensor(y,device=device)[None],op)
    x=r.reconstruction.cpu().numpy().ravel()
    error=abs(tv(x)-tv(reference))/max(1.,tv(reference))
    print('matrix',device,mode,'reference',reference.tolist(),'reference_tv',tv(reference),'tv_error',error,'residual',np.linalg.norm(a@x-y),'iterations',r.actual_iterations,'optimization_residual',r.metadata['optimization_residual'],'cost',r.resources,flush=True)
    assert tv(reference)>.1
    assert r.status=='converged'
    assert error<1e-5
    assert np.linalg.norm(a@x-y)<=epsilon+1e-8
    assert abs(tv(x)-r.final_objective)<1e-8
    # Independent smooth active-set KKT at this strictly positive fixture.
    d=np.array([[-1,0,1,0],[0,-1,0,1],[0,0,0,0],[0,0,0,0],[-1,1,0,0],[0,0,0,0],[0,0,-1,1],[0,0,0,0]],dtype=float)
    z=(d@x).reshape(2,4)
    lengths=np.linalg.norm(z,axis=0)
    dual=np.divide(z,lengths,out=np.zeros_like(z),where=lengths>0) if mode=='isotropic' else np.sign(z)
    gradient=d.T@dual.flatten(); normal=a.T@(a@x-y)
    multiplier=-gradient@normal/(normal@normal)
    kkt=np.linalg.norm(gradient+multiplier*normal)/max(1.,np.linalg.norm(gradient))
    assert x.min()>0 and multiplier>0
    assert kkt<1e-5
    print('independent_KKT',device,mode,kkt,'multiplier',multiplier,flush=True)
    assert r.metadata['optimization_residual']<1e-8
    assert base.calls==[op.counters.forward_calls,op.counters.adjoint_calls]

@pytest.mark.parametrize('device', DEVICES)
def test_budget_return_points_and_cost(device):
    a=torch.diag(torch.tensor([1.,1.2,1.4,1.6],device=device,dtype=torch.float64))
    y=torch.tensor([[.1,.6,.35,1.3]],device=device,dtype=a.dtype)
    solver=ConstrainedTVSolver(epsilon=.08,epsilon_source='interrupt fixture',num_iterations=2)
    full=solver.solve_detailed(y,Matrix(a,(1,2,2)),control=SolveControl(stop_on_convergence=False))
    first=solver.solve_detailed(y,Matrix(a,(1,2,2)),control=SolveControl(max_iterations=1))
    assert first.metadata['feasibility_status']=='violated'
    # Four basis calls estimate A's norm; each iteration: update forward,
    # update adjoint, endpoint forward, endpoint adjoint. Cut each phase.
    for limit in [0,4,5,6,7,8]:
        base=Matrix(a,(1,2,2));op=CountingLinearOperator(base,max_forward_calls=limit)
        r=solver.solve_detailed(y,op)
        assert base.calls==[op.counters.forward_calls,op.counters.adjoint_calls]
        if limit==8:
            torch.testing.assert_close(r.reconstruction,full.reconstruction)
            assert r.status=='max_iterations'
        else:
            assert r.status=='resource_exhausted'
            expected=torch.zeros_like(r.reconstruction) if limit<=4 else first.reconstruction if limit<=6 else full.reconstruction
            torch.testing.assert_close(r.reconstruction,expected)
            assert r.actual_iterations==(0 if limit<=4 else 1 if limit<=6 else 2)
            assert not r.metadata['endpoint_diagnostics_complete']
            assert r.final_objective is None and r.final_residual is None
            assert r.metadata['optimization_residual'] is None
            assert r.relative_iterate_change is None
            assert r.metadata['terminal_record'] is None
        print('budget',device,limit,r.actual_iterations,r.status,op.stats(),flush=True)

@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('algorithm',['pdhg','asd_pocs_tv'])
def test_ct_interrupts_and_nonfinite(device,algorithm):
    from inv_framework.operators.ct.radon_torch import ParallelBeamRadon2D
    base=ParallelBeamRadon2D(4,angles=torch.arange(3,device=device)*np.pi/3,device=device)
    phantom=torch.zeros(1,1,4,4,device=device);phantom[...,1:3,1:3]=.3
    y=base.forward(phantom)
    solver=ConstrainedTVSolver(epsilon=.1,epsilon_source='interrupt fixture',algorithm=algorithm,num_iterations=2,tv_updates=2)
    # Spy records admitted calls independently, including subset calls.
    from unittest.mock import patch
    events=[]
    original_forward=ParallelBeamRadon2D.forward
    original_adjoint=ParallelBeamRadon2D.adjoint
    def forward(self,x):
        events.append(('forward',x.clone()))
        return original_forward(self,x)
    def adjoint(self,x):
        events.append(('adjoint',x.clone()))
        return original_adjoint(self,x)
    with patch.object(ParallelBeamRadon2D,'forward',forward),patch.object(ParallelBeamRadon2D,'adjoint',adjoint):
        op=CountingLinearOperator(base)
        full=solver.solve_detailed(y,op,control=SolveControl(stop_on_convergence=False))
    assert len(events)==op.counters.total_calls
    assert sum(k=='forward' for k,x in events)==op.counters.forward_calls
    if algorithm=='pdhg':
        assert op.counters.forward_calls==5 and op.counters.adjoint_calls==5
        assert full.resources['transform_forward_calls']==5
        assert full.resources['transform_adjoint_calls']==4
        assert full.resources['coefficient_prox_calls']==4
        phases={'norm_estimation': [1,1], 'data_update':[2,2], 'endpoint':[2,2]}
    else:
        history=full.metadata['adaptation_history']
        trials=sum(v['accepted_tv_updates']+v['reductions'] for v in history)
        assert op.counters.forward_calls==13+trials
        assert op.counters.adjoint_calls==9
        phases={'normalization':[3,3], 'data_update':[6,6], 'data_residual':[2,0], 'tv_backtracking':[trials,0], 'endpoint':[2,0]}
    print('independent_call_phases',algorithm,device,phases,'auxiliary',full.resources,flush=True)
    # ASD: 3 normalization forwards, 3 data updates, data residual, then TV trials.
    limits=[0,3,4,5,7,8] if algorithm=='asd_pocs_tv' else [0,1,2]
    for limit in limits:
        op=CountingLinearOperator(base,max_forward_calls=limit)
        r=solver.solve_detailed(y,op)
        assert r.status=='resource_exhausted'
        assert r.actual_iterations==(1 if algorithm=='pdhg' and limit==2 else 0)
        if r.actual_iterations==0:
            torch.testing.assert_close(r.reconstruction,torch.zeros_like(phantom))
        assert r.final_objective is None
    endpoint_op=CountingLinearOperator(base)
    first=solver.solve_detailed(y,endpoint_op,control=SolveControl(max_iterations=1))
    cut=CountingLinearOperator(base,max_forward_calls=endpoint_op.counters.forward_calls-1)
    interrupted=solver.solve_detailed(y,cut,control=SolveControl(max_iterations=1))
    assert interrupted.status=='resource_exhausted'
    assert interrupted.actual_iterations==1
    torch.testing.assert_close(interrupted.reconstruction,first.reconstruction)
    assert interrupted.final_objective is None and interrupted.predicted_measurement is None
    assert interrupted.metadata['constraint_diagnostics'] is None
    assert interrupted.metadata['terminal_record'] is None
    assert interrupted.metadata['feasibility_status']=='unconfirmed'
    # An operator-generated NaN must be a numerical failure, not a failed
    # line-search trial silently discarded by ASD.
    target=5 if algorithm=='asd_pocs_tv' else 2
    counter=[0]
    def corrupt(self,x):
        counter[0]+=1
        value=original_forward(self,x)
        return value*float('nan') if counter[0]==target else value
    with patch.object(ParallelBeamRadon2D,'forward',corrupt):
        r=solver.solve_detailed(y,base)
    assert r.status=='numerical_error',r.to_dict()
    assert not r.metadata['endpoint_diagnostics_complete']
    assert r.final_objective is None
    assert torch.isfinite(r.reconstruction).all()

@pytest.mark.parametrize('device', DEVICES)
def test_asd_tolerance_override(device):
    from inv_framework.operators.ct.radon_torch import ParallelBeamRadon2D
    op=ParallelBeamRadon2D(4,angles=torch.arange(3,device=device)*np.pi/3,device=device)
    x=torch.zeros(1,1,4,4,device=device);x[...,1:3,1:3]=.3
    y=op.forward(x)
    solver=ConstrainedTVSolver(epsilon=10,epsilon_source='inactive radius control fixture',algorithm='asd_pocs_tv',num_iterations=2,tv_updates=2,tolerance=1e-12)
    loose=solver.solve_detailed(y,op,control=SolveControl(tolerance=100,patience=1))
    tight=solver.solve_detailed(y,op,control=SolveControl(tolerance=0,patience=1))
    assert loose.status=='stalled' and loose.actual_iterations==1
    assert tight.status=='max_iterations' and tight.actual_iterations==2

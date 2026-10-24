"""Nonconstant matched sparse/limited-angle P4 numerical acceptance."""
import json
import time
from pathlib import Path
import numpy as np
import pytest
import torch
from scipy.optimize import minimize
from inv_framework.operators.ct.radon_torch import ParallelBeamRadon2D
from inv_framework.solvers.constrained_tv import ConstrainedTVSolver
from inv_framework.solvers.base import SolveControl
from inv_framework.instrumentation import CountingLinearOperator
from test_p4_acceptance import DEVICES

def explicit_reference(op,y,mask,epsilon,initial,label,output):
    n=8; a=[]
    for j in range(n*n):
        e=torch.zeros_like(initial);e.flatten()[j]=1
        a.append(op.forward(e).flatten().cpu().numpy())
    a=np.stack(a,axis=1)[mask.flatten().cpu().numpy()]; b=y.flatten().cpu().numpy()[mask.flatten().cpu().numpy()]
    d=np.zeros((2,n*n,n*n))
    for i in range(n):
        for j in range(n):
            k=i*n+j
            if i<n-1:d[0,k,k]=-1;d[0,k,k+n]=1
            if j<n-1:d[1,k,k]=-1;d[1,k,k+1]=1
    def tv(x):return np.linalg.norm(d@x,axis=0).sum()
    def cones(v):return v[64:]-np.linalg.norm(d@v[:64],axis=0)
    def cone_jac(v):
        z=d@v[:64]; lengths=np.linalg.norm(z,axis=0)
        unit=np.divide(z,lengths,out=np.zeros_like(z),where=lengths>0)
        return np.c_[-np.einsum('ik,ikj->kj',unit,d),np.eye(64)]
    ref=minimize(lambda v:v[64:].sum(),np.r_[initial.flatten().cpu().numpy(),np.ones(64)],method='SLSQP',jac=lambda v:np.r_[np.zeros(64),np.ones(64)],bounds=[(0,None)]*128,constraints=[{'type':'ineq','fun':cones,'jac':cone_jac},{'type':'ineq','fun':lambda v:epsilon**2-np.sum((a@v[:64]-b)**2),'jac':lambda v:np.r_[-2*a.T@(a@v[:64]-b),np.zeros(64)]}],options={'ftol':1e-11,'maxiter':3000})
    # Independent reference precision must exceed solver acceptance.
    (output/(label+'-reference-status.json')).write_text(json.dumps(dict(success=bool(ref.success),message=str(ref.message),iterations=int(ref.nit),tv=float(tv(ref.x[:64])),epigraph_objective=float(ref.fun),cone_min=float(cones(ref.x).min()),residual=float(np.linalg.norm(a@ref.x[:64]-b)),epsilon=epsilon),indent=2),encoding='utf-8')
    np.savez(output/(label+'-reference.npz'),a=a,b=b,d=d,reference=ref.x,phantom=initial.cpu().numpy())
    print('reference',op.domain_shape,ref.success,ref.nit,ref.fun,flush=True)
    assert ref.success,ref.message
    assert np.linalg.norm(a@ref.x[:64]-b)<=epsilon+1e-8
    return a,b,d,ref,tv

@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('limited',[False,True])
def test_nonconstant_ct(device,limited,tmp_path):
    angles=torch.arange(7,device=device)*(2. if limited else np.pi)/7
    base=ParallelBeamRadon2D(8,angles=angles,device=device)
    phantom=torch.zeros(1,1,8,8,device=device);phantom[...,2:6,2:6]=.3;phantom[...,3:5,3:5]=.5
    y=base.forward(phantom);mask=torch.ones_like(y,dtype=torch.bool);mask[...,0,0]=False
    epsilon=.2 # Explicit numerical feasibility radius, not noise estimation/tuning.
    name=f"{'limited' if limited else 'sparse'}-{device}"
    a,b,d,ref,tv=explicit_reference(base,y,mask,epsilon,phantom,name,tmp_path)
    np.savez(tmp_path/(name+'-reference.npz'),a=a,b=b,d=d,reference=ref.x,phantom=phantom.cpu().numpy())
    results={}
    for algorithm in ['pdhg','asd_pocs_tv']:
        op=CountingLinearOperator(base)
        if device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter()
        r=ConstrainedTVSolver(epsilon=epsilon,epsilon_source='explicit numerical feasibility radius .2; noiseless known feasible construction',algorithm=algorithm,num_iterations=20000 if algorithm=='pdhg' else 500,tv_updates=3,tolerance=1e-5).solve_detailed(y,op,mask=mask)
        if device=='cuda':torch.cuda.synchronize()
        elapsed=time.perf_counter()-started
        x=r.reconstruction.cpu().numpy().ravel(); residual=float(np.linalg.norm(a@x-b));objective=float(tv(x));gap=abs(objective-tv(ref.x[:64]))/max(1.,tv(ref.x[:64]))
        info=dict(result=r.to_dict(),independent_residual=residual,independent_tv=objective,reference_tv=tv(ref.x[:64]),relative_tv_gap=gap,synchronized_seconds=elapsed,cuda_peak_allocated=torch.cuda.max_memory_allocated() if device=='cuda' else None)
        (tmp_path/(name+'-'+algorithm+'.json')).write_text(json.dumps(info,indent=2),encoding='utf-8')
        np.save(tmp_path/(name+'-'+algorithm+'.npy'),r.reconstruction.cpu().numpy())
        results[algorithm]=info
        print(name,algorithm,r.status,r.actual_iterations,residual,objective,gap,r.metadata['optimization_residual'],flush=True)
    # Both use identical observed data, mask, zero initialization and radius.
    for info in results.values():
        assert info['independent_residual']<=epsilon+1e-5
        assert abs(info['independent_tv']-info['result']['final_objective'])<1e-5
    assert results['pdhg']['result']['status']=='converged'
    assert results['pdhg']['relative_tv_gap']<1e-5

@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('limited',[False,True])
def test_feasible_nonconstant_descent(device,limited,tmp_path):
    angles=torch.arange(7,device=device)*(2. if limited else np.pi)/7
    base=ParallelBeamRadon2D(8,angles=angles,device=device)
    phantom=torch.zeros(1,1,8,8,device=device);phantom[...,2:6,2:6]=.3;phantom[...,3:5,3:5]=.5
    y=base.forward(phantom);mask=torch.ones_like(y,dtype=torch.bool);mask[...,0,0]=False
    epsilon=.2
    name=f"{'limited' if limited else 'sparse'}-{device}"
    image=phantom.cpu().numpy()[0,0]
    dy=np.zeros_like(image);dx=np.zeros_like(image)
    dy[:-1]=image[1:]-image[:-1];dx[:,:-1]=image[:,1:]-image[:,:-1]
    initial_tv=np.sqrt(dx*dx+dy*dy).sum()
    r=ConstrainedTVSolver(epsilon=epsilon,epsilon_source='same explicit radius',algorithm='asd_pocs_tv',num_iterations=20,tv_updates=3).solve_detailed(y,base,x_init=phantom,mask=mask,control=SolveControl(stop_on_convergence=False))
    assert r.final_residual<=epsilon+1e-5
    assert r.final_objective<initial_tv
    print('feasible descent',name,initial_tv,r.final_objective,r.final_residual,r.status,flush=True)
    (tmp_path/(name+'-feasible-descent.json')).write_text(json.dumps(r.to_dict(),indent=2),encoding='utf-8')

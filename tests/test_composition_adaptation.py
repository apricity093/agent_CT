"""Public three-domain adaptation against independent small convex references."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from scipy.optimize import minimize
from inv_framework.runner import run_experiment

ROOT=Path(__file__).resolve().parents[1]/'outputs/composition_adaptation'
ROOT.mkdir(exist_ok=True,parents=True)


def fixture(domain):
    truth=np.array([[.3,1.2],[.8,-.2]])
    if domain=='ct':
        from inv_framework.ct.backends import ParallelBeamRadon2D
        angles=np.array([0.,.45,1.,1.6],dtype=np.float32)
        op=ParallelBeamRadon2D(2,angles=torch.tensor(angles))
        matrix=np.column_stack([op.forward(torch.eye(4)[j].reshape(1,1,2,2)).numpy().ravel() for j in range(4)])
        y=(matrix@truth.ravel()).reshape(1,*op.range_shape).astype(np.float32)
        weights=np.linspace(.7,1.3,y.size).reshape(y.shape).astype(np.float32)
        config=dict(modality='ct',model={},backend='torch',case_id='external',fidelity='wls',inputs=dict(observation=y,
            geometry=dict(type='parallel_2d',domain_shape=[1,2,2],range_shape=list(op.range_shape),angles_rad=angles.tolist()),
            measurement=dict(domain='sinogram',observation_model='line_integral',noise_model='gaussian'),public_measurement=dict(weights=weights)))
    elif domain=='mri':
        from inv_framework.mri.operators import SenseEncoding
        truth=truth*np.exp(1j*np.array([[.3,1.2],[-.7,2.1]]))
        maps=np.stack([np.ones((2,2))*.8,np.exp(1j*np.array([[.4,.9],[1.3,-.2]]))*.6])[None]
        mask=np.ones(maps.shape);op=SenseEncoding(maps,mask)
        matrix=np.column_stack([op.forward(np.eye(4)[j].reshape(1,2,2)).ravel() for j in range(4)])
        y=op.forward(truth[None]);weights=np.linspace(.7,1.3,y.size).reshape(y.shape)
        config=dict(modality=domain,fidelity='wls',model=dict(axes=['batch','coil','y','x'],sampling='cartesian',spatial_shape=[2,2]),inputs=dict(kspace=y,sensitivity_maps=maps,sampling_mask=mask,data_weights=weights))
    else:
        from inv_framework.deconvolution.operators import Convolution
        psf=np.array([[.8,.2]]);op=Convolution((2,2),psf,[0,0],(0,1),'periodic')
        matrix=np.column_stack([op.forward(np.eye(4)[j].reshape(2,2)).ravel() for j in range(4)])
        y=op.forward(truth);weights=np.linspace(.7,1.3,4).reshape(2,2)
        config=dict(modality=domain,fidelity='wls',model=dict(axes=['y','x'],boundary='periodic',psf_origin=[0,0]),inputs=dict(observation=y,psf=psf,data_weights=weights))
    return config,matrix,y.ravel(),weights.ravel()


def difference(boundary):
    rows=[]
    for axis in (0,1):
        for i,j in np.ndindex(2,2):
            row=np.zeros(4);index=[i,j];other=index.copy();other[axis]=(other[axis]+1)%2
            if boundary=='periodic' or index[axis]<1:
                row[np.ravel_multi_index(other,(2,2))]+=1;row[2*i+j]-=1
            rows.append(row)
    return np.array(rows)


def reference(a,y,w,prior,domain,boundary='neumann',center=None,box=None):
    complex_=domain=='mri'; n=8 if complex_ else 4
    unpack=lambda v:v[:4]+1j*v[4:8] if complex_ else v[:4]
    d=difference(boundary)
    h=np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]])/2
    if domain!='mri':h=h[1:]
    groups=[]
    if prior in ('tv','tv_haar'):groups += [('tv',d[[j,4+j]],.08) for j in range(4)]
    if prior in ('haar_l1','tv_haar'):groups += [('haar',r[None],.06 if prior=='tv_haar' else .08) for r in h]
    def smooth(v):
        x=unpack(v)
        if center is not None:return .5*np.linalg.norm(x-center)**2
        value=.5*np.sum(w*abs(a@x-y)**2)
        if prior=='tikhonov':value+=.04*np.linalg.norm(x)**2
        if prior=='h1':value+=.04*np.linalg.norm(d@x)**2
        if prior=='gradient_huber':
            r=np.sqrt(np.sum(abs((d@x).reshape(2,4))**2,axis=0));value+=.08*np.sum(np.where(r<=.2,r*r/.4,r-.1))
        return value
    def fun(v):return smooth(v)+sum(weight*v[n+j] for j,(_,_,weight) in enumerate(groups))
    def cons(v):
        values=[]
        for j,(name,k,_) in enumerate(groups):
            z=k@unpack(v)
            if name=='haar' and not complex_:values.extend([v[n+j]-float(z[0]),v[n+j]+float(z[0])])
            else:values.append(v[n+j]-np.linalg.norm(z))
        return np.array(values)
    if groups and (domain=='mri' or (domain=='deconvolution' and prior=='haar_l1')):
        realify=lambda z:np.r_[z.real,z.imag] if complex_ else z.real
        if center is None:
            hc=a.conj().T@(w[:,None]*a);bc=a.conj().T@(w*y)
            hm=np.block([[hc.real,-hc.imag],[hc.imag,hc.real]]) if complex_ else hc.real
            b=realify(bc)
        else:hm=np.eye(n);b=realify(center)
        matrices=[];slices=[];bounds=[];offset=0
        for _,k,weight in groups:
            km=np.block([[k,np.zeros_like(k)],[np.zeros_like(k),k]]) if complex_ else k
            matrices.append(km);slices.append((slice(offset,offset+km.shape[0]),weight));offset+=km.shape[0]
            bounds.extend([(-weight,weight)]*km.shape[0])
        km=np.concatenate(matrices);hi=np.linalg.inv(hm);base=hi@b
        hh=km@hi@km.T;bb=km@base
        constraints=[]
        for sl,weight in slices:
            if sl.stop-sl.start>1:
                def cf(v,sl=sl,weight=weight):return weight**2-np.dot(v[sl],v[sl])
                def cj(v,sl=sl):
                    out=np.zeros_like(v);out[sl]=-2*v[sl];return out
                constraints.append(dict(type='ineq',fun=cf,jac=cj))
        dual=minimize(lambda v:.5*v@hh@v-bb@v,np.zeros(offset),jac=lambda v:hh@v-bb,
            bounds=bounds,constraints=constraints,method='SLSQP',options=dict(ftol=1e-14,maxiter=1500))
        assert dual.success,dual.message
        v=base-hi@km.T@dual.x;x=unpack(v)
        return x,smooth(v)+sum(weight*np.linalg.norm(k@x) for _,k,weight in groups),d,h
    initial=np.r_[np.zeros(n),np.ones(len(groups))]
    bounds=([(box[0],box[1])]*n if box else [(None,None)]*n)+[(0,None)]*len(groups)
    def jac(v):
        x=unpack(v)
        g=x-center if center is not None else a.conj().T@(w*(a@x-y))
        if center is None:
            if prior=='tikhonov':g=g+.08*x
            if prior=='h1':g=g+.08*d.T@(d@x)
            if prior=='gradient_huber':
                z=(d@x).reshape(2,4);g=g+.08*d.T@(z/np.maximum(np.linalg.norm(z,axis=0),.2)).ravel()
        return np.r_[g.real,g.imag,[weight for _,_,weight in groups]] if complex_ else np.r_[g.real,[weight for _,_,weight in groups]]
    def cjac(v):
        rows=[]
        for j,(name,k,_) in enumerate(groups):
            z=k@unpack(v)
            if name=='haar' and not complex_:
                for sign in (-1,1):
                    row=np.zeros_like(v);row[:n]=sign*k[0];row[n+j]=1;rows.append(row)
            else:
                r=np.linalg.norm(z);g=k.conj().T@(z/r) if r else np.zeros(4)
                row=np.zeros_like(v);row[:n]=-np.r_[g.real,g.imag] if complex_ else -g.real;row[n+j]=1;rows.append(row)
        return np.array(rows)
    ref=minimize(fun,initial,jac=jac,method='SLSQP',bounds=bounds,
        constraints=[dict(type='ineq',fun=cons,jac=cjac)] if groups else [],options=dict(ftol=1e-13,maxiter=1500))
    assert ref.success,ref.message
    return unpack(ref.x),fun(ref.x),d,h


CONFIGS=[(domain,method,prior) for domain in ('ct','mri','deconvolution') for method,prior in
    [('admm','tikhonov'),('pdhg','tikhonov'),('fista','tikhonov'),('admm','tv'),('admm','haar_l1'),('admm','h1'),('admm','gradient_huber'),('fista','tv'),('pdhg','haar_l1'),('admm','tv_haar')]]


@pytest.mark.parametrize('domain,method,prior',CONFIGS)
def test_public_adaptation(domain,method,prior):
    config,a,y,w=fixture(domain)
    params=dict(min_value=None,max_value=None,regularizer=prior,strength=.08,level=1,delta=.2,num_iterations=2000,tolerance=2e-8 if domain!='ct' else 1e-5)
    if domain=='ct':params['weight_source']='provided'
    if method=='admm':params.update(rho=.8,rho_data=1.1,rho_regularizer=.7,rho_constraint=.9,cg_tolerance=1e-5 if domain!='ct' else .2,cg_max_iterations=50)
    if prior=='tv_haar':params.update(haar_strength=.06,rho_haar=1.3)
    if method=='fista' and prior=='tv':params.update(tv_num_iterations=5000,tv_tolerance=1e-5 if domain!='ct' else .01)
    config.update(algorithm=method,params=params)
    result=run_experiment(config,ROOT/f'{domain}_{method}_{prior}')
    assert result.status=='success',result.error
    x=result.outputs['reconstruction'].ravel()
    xr,obj,d,h=reference(a,y,w,prior,domain)
    data=.5*np.sum(w*abs(a@x-y)**2)
    penalty=0.
    if prior in ('tv','tv_haar'):penalty+=.08*np.linalg.norm((d@x).reshape(2,4),axis=0).sum()
    if prior in ('haar_l1','tv_haar'):penalty+=(.06 if prior=='tv_haar' else .08)*abs(h@x).sum()
    if prior=='tikhonov':penalty=.04*np.linalg.norm(x)**2
    if prior=='h1':penalty=.04*np.linalg.norm(d@x)**2
    if prior=='gradient_huber':
        r=np.linalg.norm((d@x).reshape(2,4),axis=0);penalty=.08*np.where(r<=.2,r*r/.4,r-.1).sum()
    objective=data+penalty
    grad=a.conj().T@(w*(a@x-y))
    if prior=='tikhonov':grad+=.08*x
    if prior=='h1':grad+=.08*d.T@(d@x)
    if prior=='gradient_huber':
        z=(d@x).reshape(2,4);r=np.linalg.norm(z,axis=0);grad+=.08*d.T@(z/np.maximum(r,.2)).ravel()
    if prior in ('tv','haar_l1','tv_haar'):
        prox,_,_,_=reference(a,y,w,prior,domain,center=x-grad);optimality=np.linalg.norm(x-prox)/max(1.,np.linalg.norm(x))
    else:optimality=np.linalg.norm(grad)/max(1.,np.linalg.norm(x))
    gap=abs(objective-obj)/max(1.,abs(obj));error=np.linalg.norm(x-xr)/max(1.,np.linalg.norm(xr))
    evidence=dict(objective=objective,reference_objective=obj,objective_gap=gap,optimality=optimality,solution_error=error,stopping_reason=result.stopping_reason,iterations=result.iterations)
    (ROOT/f'{domain}_{method}_{prior}'/'reference.json').write_text(json.dumps(evidence,indent=2,default=float))
    print(domain,method,prior,evidence)
    threshold=1e-3 if domain=='ct' else 1e-5
    assert gap<threshold and optimality<threshold,(gap,optimality,result.stopping_reason)
    reported=result.diagnostics['objective']
    assert abs(reported-objective)<threshold
    if method=='admm':
        md=result.diagnostics['convergence']['metadata'] if domain=='ct' else result.diagnostics
        assert md['block_rho']['data']==1.1 and md['block_rho']['constraint']==.9
        assert abs(md['data_value']+sum(md['regularizer_contributions'].values())-objective)<threshold
    if domain=='mri':assert np.iscomplexobj(x) and np.linalg.norm(x.imag)>.1


def test_joint_degeneracy_bounds_budget_and_fft():
    config,_,_,_=fixture('deconvolution')
    config.update(algorithm='admm',fidelity='l2');config['inputs'].pop('data_weights')
    common=dict(strength=.08,level=1,boundary='periodic',num_iterations=600,tolerance=1e-8,cg_tolerance=1e-5,min_value=0.,max_value=.6,rho_data=1.1,rho_regularizer=.7,rho_constraint=.9)
    config['params']=dict(common,regularizer='tv_haar',haar_strength=0.,rho_haar=1.3)
    zero=run_experiment(config,ROOT/'zero_joint')
    config['params']=dict(common,regularizer='tv')
    single=run_experiment(config,ROOT/'single_tv_fft')
    assert zero.status==single.status=='success'
    np.testing.assert_array_equal(zero.outputs['reconstruction'],single.outputs['reconstruction'])
    assert 'haar_l1' not in zero.diagnostics['block_rho']
    assert single.diagnostics['subproblem_path']=='periodic_fft'
    assert single.outputs['reconstruction'].min()>=0 and single.outputs['reconstruction'].max()<=.6
    config['params']['subproblem_solver']='cg'
    general=run_experiment(config,ROOT/'single_tv_cg')
    assert general.status=='success' and general.diagnostics['subproblem_path']=='hermitian_cg'
    np.testing.assert_allclose(general.outputs['reconstruction'],single.outputs['reconstruction'],atol=1e-6)
    config['params'].update(cg_max_iterations=1,cg_tolerance=1e-14)
    limited=run_experiment(config,ROOT/'inner_budget')
    assert not limited.converged and limited.stopping_reason=='inner_budget_exhausted'
    assert 'reconstruction' in limited.outputs


@pytest.mark.parametrize('prior',['tv','haar_l1'])
def test_mri_legacy_equivalence(prior):
    config,_,_,_=fixture('mri')
    config.update(algorithm='cs_tv' if prior=='tv' else 'cs_wavelet',params=dict(lam=.08,rho=.7,max_iter=500,tolerance=1e-8,cg_tolerance=1e-5,cg_max_iter=50))
    if prior=='haar_l1':config['params']['levels']=1
    old=run_experiment(config,ROOT/f'legacy_mri_{prior}')
    config.update(algorithm='admm',params=dict(regularizer=prior,strength=.08,rho=.7,level=1,boundary='periodic',num_iterations=1000,tolerance=1e-8,cg_tolerance=1e-5))
    new=run_experiment(config,ROOT/f'equivalent_mri_{prior}')
    assert old.status==new.status=='success',(old.error,new.error)
    np.testing.assert_allclose(new.outputs['reconstruction'],old.outputs['reconstruction'],atol=1e-5)
    assert abs(new.diagnostics['objective']-old.diagnostics['data_fit']-old.diagnostics['regularizer_contribution'])<1e-5


def test_deconvolution_legacy_equivalence():
    config,_,_,_=fixture('deconvolution');config['inputs'].pop('data_weights')
    config.update(fidelity='l2',algorithm='tv_admm',params=dict(lam=.08,rho=.7,max_iter=600,tolerance=1e-8))
    old=run_experiment(config,ROOT/'legacy_deconvolution_tv')
    config.update(algorithm='admm',params=dict(regularizer='tv',strength=.08,rho=.7,boundary='periodic',num_iterations=600,tolerance=1e-8))
    new=run_experiment(config,ROOT/'equivalent_deconvolution_tv')
    assert old.status==new.status=='success',(old.error,new.error)
    np.testing.assert_allclose(new.outputs['reconstruction'],old.outputs['reconstruction'],atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_changed_admm_resident_arrays():
    from inv_framework.solvers.composition import CompositeSolver
    from inv_framework.ct.backends import ParallelBeamRadon2D
    op=ParallelBeamRadon2D(2,angles=torch.tensor([0.,.5,1.,1.5],device='cuda'),device='cuda')
    x=torch.rand(1,1,2,2,device='cuda');y=op.forward(x)
    from torch.utils._python_dispatch import TorchDispatchMode
    transfers=[]
    class Transfers(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            out=func(*args,**(kwargs or {}))
            if func==torch.ops.aten._to_copy.default:
                transfers.append(dict(shape=list(args[0].shape),source=str(args[0].device),destination=str(out.device)))
            return out
    with Transfers(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
        result=CompositeSolver('admm',regularizer='tv_haar',level=1,num_iterations=3,cg_tolerance=.1).solve_detailed(y,op)
        torch.cuda.synchronize()
    assert result.reconstruction.device.type=='cuda' and not result.reconstruction.requires_grad
    copies=[e.input_shapes for e in prof.events() if e.name=='aten::_to_copy']
    assert all(not t['shape'] or np.prod(t['shape'])==1 or t['source']==t['destination'] for t in transfers)
    (ROOT/'cuda_residency.json').write_text(json.dumps(dict(device=str(result.reconstruction.device),iterations=result.actual_iterations,same_device_array_conversions=int(sum(bool(t['shape']) and np.prod(t['shape'])>1 for t in transfers)),host_array_transfers=0)))


@pytest.mark.parametrize('prior',['tikhonov','tv','haar_l1','h1','huber'])
def test_ct_single_legacy_equivalence(prior):
    from inv_framework.solvers.splitting import solve_split_detailed
    from inv_framework.solvers.composition import make_regularizer
    from inv_framework.ct.backends import ParallelBeamRadon2D
    from inv_framework.core.fidelity import QuadraticFidelity
    config,_,_,_=fixture('ct')
    p=dict(regularizer=prior,strength=.08,level=1,num_iterations=500,tolerance=1e-5,cg_tolerance=.2,weight_source='provided')
    config.update(algorithm='admm',params=p)
    new=run_experiment(config,ROOT/f'equivalent_ct_{prior}')
    op=ParallelBeamRadon2D(2,angles=torch.tensor(config['inputs']['geometry']['angles_rad']))
    y=torch.tensor(config['inputs']['observation']);weights=torch.tensor(config['inputs']['public_measurement']['weights'])
    old=solve_split_detailed(op,y,algorithm='admm',regularizer=make_regularizer(prior,op.domain_shape,p),strength=.08,num_iterations=500,tolerance=1e-5,cg_tolerance=.2,cg_fixed_scale=True,fidelity=QuadraticFidelity(y,weights))
    assert new.status=='success'
    x=new.outputs['reconstruction'];gap=abs(new.diagnostics['objective']-old.final_objective)
    assert gap<1e-3
    assert np.linalg.norm(x-old.reconstruction.numpy())/max(1.,np.linalg.norm(x))<1e-3
    (ROOT/f'equivalent_ct_{prior}'/'legacy_comparison.json').write_text(json.dumps(dict(objective_gap=gap,legacy_status=old.status,legacy_reason=old.stopping_reason)))

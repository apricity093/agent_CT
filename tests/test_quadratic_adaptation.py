"""Public quadratic adaptation against independent dense reference systems."""
import json
from pathlib import Path
import numpy as np
import pytest
from inv_framework.runner import run_experiment
from test_composition_adaptation import fixture

ROOT=Path(__file__).resolve().parents[1]/'outputs/quadratic_adaptation'
ROOT.mkdir(parents=True,exist_ok=True)

@pytest.mark.parametrize('domain',['ct','mri','deconvolution'])
@pytest.mark.parametrize('method',['landweber','cgls','lsqr','cg'])
def test_public_quadratic(domain,method):
    config,a,y,q=fixture(domain)
    config.update(algorithm=method,params=dict(regularizer='tikhonov',strength=.08,num_iterations=2000,tolerance=1e-5 if domain=='ct' else 1e-9))
    if domain=='ct':config['params']['weight_source']='provided'
    result=run_experiment(config,ROOT/f'{domain}_{method}')
    assert result.status=='success',result.error
    x=result.outputs['reconstruction'].ravel()
    h=a.conj().T@(q[:,None]*a)+.08*np.eye(4);rhs=a.conj().T@(q*y)
    reference=np.linalg.solve(h,rhs)
    objective=lambda v:.5*np.sum(q*abs(a@v-y)**2)+.04*np.linalg.norm(v)**2
    gap=abs(objective(x)-objective(reference))/max(1.,abs(objective(reference)))
    nr=np.linalg.norm(h@x-rhs)/max(1.,np.linalg.norm(rhs))
    evidence=dict(domain=domain,algorithm=method,objective=float(objective(x)),reference_objective=float(objective(reference)),objective_gap=float(gap),normalized_normal_residual=float(nr),converged=result.converged,stopping_reason=result.stopping_reason,iterations=result.iterations)
    (ROOT/f'{domain}_{method}'/'reference.json').write_text(json.dumps(evidence,indent=2))
    print(evidence)
    threshold=1e-3 if domain=='ct' else 1e-5
    assert gap<threshold and nr<threshold
    assert result.converged
    if domain=='mri':assert np.iscomplexobj(x) and np.linalg.norm(x.imag)>.1


def test_h1_augmentation_and_landweber_step():
    import torch
    from inv_framework.core.imaging_operators import CartesianSense
    from inv_framework.solvers.quadratic import AugmentedQuadratic,QuadraticSolver
    from inv_framework.core.fidelity import QuadraticFidelity
    config,_,_,_=fixture('mri');inputs=config['inputs']
    op=CartesianSense(torch.as_tensor(inputs['sensitivity_maps']),torch.as_tensor(inputs['sampling_mask']))
    y=torch.as_tensor(inputs['kspace']);q=torch.as_tensor(inputs['data_weights'])
    b=AugmentedQuadratic(op,y,q,'h1',.7,'periodic')
    x=torch.tensor([[[.2+.3j,.7-.4j],[.8+.1j,-.1+.9j]]],dtype=torch.complex128)
    z=torch.arange(b.forward(x).numel(),dtype=torch.float64).to(torch.complex128)*(1+.4j)
    left=torch.vdot(b.forward(x),z);right=torch.vdot(x.flatten(),b.adjoint(z).flatten())
    error=float(abs(left-right)/abs(left));assert error<1e-5
    bound=b.bound(x)
    solver=QuadraticSolver('landweber',regularizer='h1',strength=.7,boundary='periodic',num_iterations=1,tolerance=0.)
    solver.fidelity=QuadraticFidelity(y,q)
    result=solver.solve_detailed(y,op,x_init=x)
    torch.testing.assert_close(result.reconstruction,x-b.adjoint(b.forward(x)-b.rhs())/bound)
    assert result.status!='converged'
    with pytest.raises(ValueError,match='not clipped'):solver.solve_detailed(y,op,x_init=x,step_size=2.1/bound)
    (ROOT/'augmentation.json').write_text(json.dumps(dict(hermitian_relative_error=error,augmented_squared_bound=bound,step=1/bound)))


def test_parameters_budget_damping_and_public_rejections():
    config,_,_,_=fixture('deconvolution')
    config.update(algorithm='lsqr',params=dict(regularizer='none',strength=0.,damping=.3,num_iterations=50,tolerance=1e-9))
    damped=run_experiment(config,ROOT/'damping')
    config['params'].update(regularizer='tikhonov',strength=.09,damping=0.)
    augmented=run_experiment(config,ROOT/'damping_augmented')
    assert damped.status==augmented.status=='success'
    np.testing.assert_allclose(damped.outputs['reconstruction'],augmented.outputs['reconstruction'],atol=1e-12)
    config['params']['damping']=.3
    assert run_experiment(config).status=='failed'
    config.update(algorithm='cg',params=dict(regularizer='h1',strength=.08))
    assert run_experiment(config).status=='failed'
    config.update(algorithm='landweber',params=dict(regularizer='tikhonov',strength=.08,num_iterations=80,tolerance=1e-9))
    limited=run_experiment(config,ROOT/'budget',params=dict(num_iterations=1))
    assert limited.status=='success' and limited.iterations==1 and not limited.converged
    first=limited.outputs['reconstruction'].copy()
    changed=run_experiment(config,params=dict(num_iterations=1,step_size=.01))
    assert changed.effective_params['num_iterations']==1
    assert np.linalg.norm(changed.outputs['reconstruction']-first)>.01
    config.update(fidelity='poisson_nll')
    assert run_experiment(config).status=='failed'
    config.update(fidelity='l2',params=dict(regularizer='tv'));config['inputs'].pop('data_weights')
    assert run_experiment(config).status=='failed'


def test_legacy_cg_sense_and_ct_baselines():
    config,_,_,_=fixture('mri')
    config.update(algorithm='cg_sense',params=dict(lam=.08,max_iter=50,tolerance=1e-9))
    old=run_experiment(config,ROOT/'legacy_cg_sense')
    config.update(algorithm='cg',params=dict(regularizer='tikhonov',strength=.08,num_iterations=50,tolerance=1e-9))
    new=run_experiment(config,ROOT/'cg_sense_equivalence')
    assert old.status==new.status=='success',(old.error,new.error)
    np.testing.assert_allclose(old.outputs['reconstruction'],new.outputs['reconstruction'],atol=1e-9)
    import torch
    from inv_framework.ct.backends import ParallelBeamRadon2D
    from inv_framework.solvers.algebraic import LandweberSolver,CGLSSolver,LSQRSolver
    from inv_framework.solvers.quadratic import QuadraticSolver
    config,_,_,_=fixture('ct');y=torch.as_tensor(config['inputs']['observation'])
    op=ParallelBeamRadon2D(2,angles=torch.tensor(config['inputs']['geometry']['angles_rad']))
    for method,cls in [('landweber',LandweberSolver),('cgls',CGLSSolver),('lsqr',LSQRSolver)]:
        params=dict(num_iterations=2,min_value=None,max_value=None)
        if method=='landweber':params['step_size']=.05
        old=cls(**params).solve(y,op)
        new=QuadraticSolver(method,**params,tolerance=0.).solve(y,op)
        torch.testing.assert_close(old,new,atol=2e-5,rtol=2e-5)


def test_discovery_and_cumulative_evidence():
    from inv_framework.ct.api import list_capabilities as ct
    from inv_framework.mri.api import list_capabilities as mri
    from inv_framework.deconvolution.api import list_capabilities as deconv
    for discover in (ct,mri,deconv):
        assert {'landweber','cgls','lsqr','cg'}<={r['name'] for r in discover()['algorithms']}
    rows=[]
    for domain in ('ct','mri','deconvolution'):
        _,a,y,q=fixture(domain)
        h=a.conj().T@(q[:,None]*a)+.08*np.eye(4);rhs=a.conj().T@(q*y)
        for method in ('admm','pdhg','fista'):
            path=ROOT.parent/'composition_adaptation'/f'{domain}_{method}_tikhonov'/'reference.json'
            evidence=json.loads(path.read_text());threshold=1e-3 if domain=='ct' else 1e-5
            assert evidence['objective_gap']<threshold and evidence['optimality']<threshold
            x=np.load(path.parent/'reconstruction.npy').ravel()
            evidence['normalized_normal_residual']=float(np.linalg.norm(h@x-rhs)/max(1.,np.linalg.norm(rhs)))
            assert evidence['normalized_normal_residual']<threshold
            rows.append(dict(domain=domain,algorithm=method,source=str(path),**evidence))
        for method in ('landweber','cgls','lsqr','cg'):
            path=ROOT/f'{domain}_{method}'/'reference.json'
            rows.append(dict(source=str(path),**json.loads(path.read_text())))
    (ROOT/'adaptation_21.json').write_text(json.dumps(rows,indent=2))
    assert len(rows)==21


def test_changed_quadratic_residency():
    import torch
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from inv_framework.ct.backends import ParallelBeamRadon2D
    from inv_framework.solvers.quadratic import QuadraticSolver
    from torch.utils._python_dispatch import TorchDispatchMode
    op=ParallelBeamRadon2D(2,angles=torch.tensor([0.,.5,1.,1.5],device='cuda'),device='cuda')
    y=op.forward(torch.ones(1,1,2,2,device='cuda'));transfers=[]
    class Transfers(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            out=func(*args,**(kwargs or {}))
            if func==torch.ops.aten._to_copy.default and args[0].numel()>1:
                transfers.append((str(args[0].device),str(out.device)))
            return out
    with Transfers():
        for method in ('landweber','cgls','lsqr','cg'):
            result=QuadraticSolver(method,regularizer='tikhonov',strength=.08,num_iterations=3,tolerance=1e-5).solve_detailed(y,op)
            assert result.reconstruction.device.type=='cuda' and not result.reconstruction.requires_grad
    assert all(a==b for a,b in transfers)
    (ROOT/'residency.json').write_text(json.dumps(dict(device='cuda',whole_array_host_transfers=0,methods=4)))


def test_undersampled_cg_and_public_h1():
    from test_composition_adaptation import difference
    config,a,y,q=fixture('mri')
    mask=config['inputs']['sampling_mask'].copy();mask[...,0,:]=0
    config['inputs']['sampling_mask']=mask
    a=mask.ravel()[:,None]*a;y=mask.ravel()*y
    config.update(algorithm='cg',params=dict(regularizer='tikhonov',strength=.08,num_iterations=50,tolerance=1e-9))
    result=run_experiment(config,ROOT/'undersampled_cg')
    assert result.converged,result.error
    h=a.conj().T@(q[:,None]*a)+.08*np.eye(4)
    np.testing.assert_allclose(result.outputs['reconstruction'].ravel(),np.linalg.solve(h,a.conj().T@(q*y)),atol=1e-8)
    # One public H1 path checks the common branch without a method/prior product.
    config,a,y,q=fixture('deconvolution');d=difference('periodic')
    config.update(algorithm='lsqr',params=dict(regularizer='h1',boundary='periodic',strength=.2,num_iterations=50,tolerance=1e-9))
    result=run_experiment(config,ROOT/'public_h1')
    assert result.converged,result.error
    reference=np.linalg.solve(a.T@(q[:,None]*a)+.2*d.T@d,a.T@(q*y))
    np.testing.assert_allclose(result.outputs['reconstruction'].ravel(),reference,atol=1e-8)

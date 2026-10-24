import numpy as np
import pytest
import torch
from scipy.optimize import brentq
from inv_framework.fidelity import TransmissionPoissonFidelity

CUDA_DEVICE = pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))

@pytest.mark.parametrize('device',['cpu', CUDA_DEVICE])
def test_masked_nll_value_derivative_and_hessian(device):
    counts=np.array([0.,1.,8.,100.]); values=np.array([-3.,0.,2.,4.])
    mask=np.array([1.,0.,1.,1.]); scale=.7; incident=20.
    tensor=lambda a:torch.tensor(a,device=device,dtype=torch.float64)
    fidelity=TransmissionPoissonFidelity(tensor(counts),incident,scale=scale,mask=tensor(mask))
    expected_value=float(np.sum(mask*(incident*np.exp(-scale*values)+counts*scale*values)))
    assert abs(float(fidelity.value(tensor(values)))-expected_value)<1e-9
    direction=tensor([.1,-.2,.3,-.4]); epsilon=1e-6
    derivative=float((fidelity.value(tensor(values)+epsilon*direction)-fidelity.value(tensor(values)-epsilon*direction))/(2*epsilon))
    expected=float(np.sum(mask*scale*(counts-incident*np.exp(-scale*values))*direction.cpu().numpy()))
    assert abs(derivative-expected)/max(abs(expected),1e-12)<1e-7
    hessian=torch.autograd.functional.hessian(fidelity.value,tensor(values)).cpu().numpy()
    reference=np.diag(mask*scale*scale*incident*np.exp(-scale*values))
    assert np.linalg.norm(hessian-reference)/np.linalg.norm(reference)<1e-12
    assert np.linalg.eigvalsh(hessian).min()>=0

@pytest.mark.parametrize('device',['cpu', CUDA_DEVICE])
def test_independent_conjugate(device):
    c=np.array([0.,1.,8.,100.]);v=np.array([-3.,0.,2.,4.]);s=.7;step=.3
    f=TransmissionPoissonFidelity(torch.tensor(c,device=device),20.,scale=s,tolerance=1e-12)
    before=f.inner_iterations
    z=f.dual_prox(torch.tensor(v,device=device),step).cpu().numpy()
    ref=np.array([brentq(lambda u:u-vi-step/s*np.log((ci-u/s)/20),-10000,s*ci-1e-12,xtol=1e-13) for ci,vi in zip(c,v)])
    assert np.max(abs(z-ref))<1e-9
    assert f.inner_iterations>before and f.prox_seconds>0
    f.max_iterations=1
    with pytest.raises(RuntimeError): f.proximal(torch.tensor(v,device=device),step)
    assert f.inner_failures==1
    assert f.last_residual is not None

def test_model_rejects_latent_and_negative_fields():
    from dataclasses import replace
    from pathlib import Path
    from inv_framework.benchmarks.ct_cases import load_ct_case
    case=load_ct_case('parallel_2d/disk_analytic_32',data_root=Path(__file__).resolve().parents[1]/'test/data')
    with pytest.raises(ValueError,match='latent'):
        replace(case,metadata={**case.metadata,'measurement':{'noise_model':'poisson_gaussian'}},public_measurement={'counts':torch.zeros_like(case.measurement)})


@pytest.mark.parametrize('model',['gaussian','poisson','poisson_gaussian'])
@pytest.mark.parametrize('device',['cpu', CUDA_DEVICE])
def test_heterogeneous_data_chain(tmp_path,model,device):
    import json,h5py
    from pathlib import Path
    from dataclasses import replace
    from inv_framework.measurement import simulate_observation
    from inv_framework.benchmarks.ct_cases import load_ct_case,write_ct_case,restrict_ct_case
    from inverse_agent.cases.staging import stage_ct_case
    case=load_ct_case('parallel_2d/disk_analytic_32',data_root=Path(__file__).resolve().parents[1]/'test/data')
    shape=case.measurement.shape
    known=torch.arange(shape[-2],dtype=case.measurement.dtype).reshape(1,1,-1,1).expand(shape)+1
    y,public,meta=simulate_observation(case.measurement_clean,model=model,seed=17,I0=known*10,sigma=known*.01,sigma_e=known if model=='poisson_gaussian' else 0.,scale=.7)
    mask=torch.ones_like(y,dtype=torch.bool);mask[...,::3,:]=False
    case=replace(case,measurement=y,public_measurement=public,valid_measurement_mask=mask,metadata={**case.metadata,'measurement':meta})
    record=write_ct_case(case,tmp_path/'case');record['path']='case'
    catalog=tmp_path/'catalog.json';catalog.write_text(json.dumps({'schema_version':'1.0','cases':[record]}))
    loaded=load_ct_case(case.case_id,data_root=tmp_path).to(device=device,dtype=torch.float64)
    subset=restrict_ct_case(loaded,[3,1])
    for key,value in public.items():
        assert torch.equal(subset.public_measurement[key].cpu(),value.double().index_select(-2,torch.tensor([3,1])))
    assert torch.equal(subset.valid_measurement_mask.cpu(),mask.index_select(-2,torch.tensor([3,1])))
    raw=subset.public_measurement.get('counts',subset.public_measurement.get('c_obs'))
    if raw is not None:assert subset.metadata['measurement']['clipped_fraction']==float((raw<meta['floor']).double().mean())
    staged=stage_ct_case(catalog,case.case_id,tmp_path/'staged')
    with h5py.File(staged.public_path) as h:
        assert set(h['measurement'])=={'y_observed',*public}
        assert not np.any(h['truth/x'][...])

@pytest.mark.parametrize('device',['cpu', CUDA_DEVICE])
def test_masked_nll(device):
    c=torch.tensor([0.,2.],device=device,dtype=torch.float64)
    f=TransmissionPoissonFidelity(c,20.,mask=torch.tensor([0.,1.],device=device),tolerance=1e-12)
    v=torch.tensor([3.,-2.],device=device,dtype=torch.float64)
    assert f.gradient(v)[0]==0
    assert abs(float(f.proximal(v,.3)[0])-3)<1e-10
    assert abs(float(f.dual_prox(v,.3)[0]))<1e-10


@pytest.mark.parametrize('device',['cpu', CUDA_DEVICE])
@pytest.mark.parametrize('prior',['identity','h1'])
def test_weighted_cg(device,prior):
    from test_p1_composable import Matrix
    from inv_framework.operators.gradient import GradientOperator
    from inv_framework.fidelity import QuadraticFidelity
    from inv_framework.solvers import TikhonovSolver
    a=np.diag([1.1,1.2,1.3,1.4]);y=np.array([-1.,.4,-.3,1.2]);w=np.array([0.,.5,2.,3.]);lam=.17
    tensor=lambda x:torch.tensor(x,device=device,dtype=torch.float64)
    d=np.zeros((8,4))
    d[0,:]=[-1,0,1,0];d[1,:]=[0,-1,0,1];d[4,:]=[-1,1,0,0];d[6,:]=[0,0,-1,1]
    t=np.eye(4) if prior=='identity' else d
    f=QuadraticFidelity(tensor(y)[None],tensor(w)[None])
    result=TikhonovSolver(reg_strength=lam,num_iterations=100,tolerance=1e-10,regularization_operator=None if prior=='identity' else GradientOperator((1,2,2))).solve_detailed(tensor(y)[None],Matrix(tensor(a),(1,2,2)),fidelity=f)
    reference=np.linalg.solve(a.T@np.diag(w)@a+lam*t.T@t,a.T@(w*y))
    x=result.reconstruction.flatten().cpu().numpy()
    assert np.linalg.norm(x-reference)<1e-6
    assert abs(result.final_objective-(.5*np.sum(w*(a@x-y)**2)+lam/2*np.sum((t@x)**2)))<1e-9


@pytest.mark.parametrize('device',['cpu', CUDA_DEVICE])
def test_explicit_whitened_operator(device):
    from test_p1_composable import Matrix
    from inv_framework.operators.weighted import WeightedLinearOperator
    a=torch.tensor([[1.,-.3,.2,0.],[0.,1.,.4,-.2],[.1,0.,1.,.3],[.3,.2,0.,1.]],device=device,dtype=torch.float64)
    weights=torch.tensor([[0.,.5,2.,3.]],device=device,dtype=a.dtype)
    b=WeightedLinearOperator(Matrix(a,(1,2,2)),weights)
    x=torch.arange(4,device=device,dtype=a.dtype).reshape(1,1,2,2);y=torch.tensor([[.2,-.4,.7,.1]],device=device,dtype=a.dtype)
    matrix=weights.sqrt().flatten()[:,None]*a
    assert torch.allclose(b.forward(x).flatten(),matrix@x.flatten(),atol=1e-12,rtol=0)
    assert torch.allclose(b.adjoint(y).flatten(),matrix.T@y.flatten(),atol=1e-12,rtol=0)
    assert abs(float((b.forward(x)*y).sum()-(x*b.adjoint(y)).sum()))<1e-12
    assert b.squared_norm_upper_bound(x)>=float(torch.linalg.matrix_norm(matrix,2)**2)


@pytest.mark.parametrize('device',['cpu', CUDA_DEVICE])
@pytest.mark.parametrize('failure_call',[2,3])
def test_nll_inner_interruption_complete_state(device,failure_call):
    from test_p1_composable import Matrix
    from inv_framework.solvers import PDHGSolver
    from inv_framework.regularizers import TikhonovRegularizer
    tensor=lambda x:torch.tensor(x,device=device,dtype=torch.float64)
    y=tensor([[1.,3.,8.,12.]])
    def operator():return Matrix(torch.eye(4,device=device,dtype=y.dtype),(1,2,2))
    def solve(f,n):return PDHGSolver(TikhonovRegularizer(),num_iterations=n,tolerance=0,min_value=0).solve_detailed(y,operator(),fidelity=f)
    reference=solve(TransmissionPoissonFidelity(y,20.,tolerance=1e-12),1)
    class Interrupt(TransmissionPoissonFidelity):
        calls=0
        def dual_prox(self,value,step):
            self.calls+=1
            if self.calls==failure_call:self.max_iterations=1
            return super().dual_prox(value,step)
    f=Interrupt(y,20.,tolerance=1e-12)
    result=solve(f,3)
    assert result.actual_iterations==1
    assert result.stopping_reason=='poisson_prox_inner_max_iterations'
    assert torch.equal(result.reconstruction,reference.reconstruction)
    assert result.final_objective is None and result.final_residual is None
    assert not result.metadata['endpoint_diagnostics_complete']
    assert f.inner_failures==1

@pytest.mark.parametrize('device',['cpu', CUDA_DEVICE])
@pytest.mark.parametrize('limit',[5,6])
def test_nll_operator_interrupt(device,limit):
    from test_p1_composable import Matrix
    from inv_framework.solvers import PDHGSolver
    from inv_framework.regularizers import TikhonovRegularizer
    from inv_framework.instrumentation import CountingLinearOperator
    y=torch.tensor([[1.,3.,8.,12.]],device=device,dtype=torch.float64)
    def operator():return Matrix(torch.eye(4,device=device,dtype=y.dtype),(1,2,2))
    f=lambda:TransmissionPoissonFidelity(y,20.,tolerance=1e-12)
    ref=PDHGSolver(TikhonovRegularizer(),num_iterations=1,tolerance=0,min_value=0).solve_detailed(y,operator(),fidelity=f())
    op=CountingLinearOperator(operator(),max_forward_calls=limit)
    result=PDHGSolver(TikhonovRegularizer(),num_iterations=3,tolerance=0,min_value=0).solve_detailed(y,op,fidelity=f())
    assert result.status=='resource_exhausted'
    assert result.actual_iterations==1
    assert torch.equal(result.reconstruction,ref.reconstruction)
    assert result.final_objective is None and result.final_residual is None


@pytest.mark.parametrize('device',['cpu', CUDA_DEVICE])
@pytest.mark.parametrize('model',['gaussian','poisson','poisson_gaussian'])
def test_noise_ensemble_and_rng(device,model):
    from inv_framework.measurement import simulate_observation
    p=torch.full((1,1,200,200),.5,device=device,dtype=torch.float64)
    torch.manual_seed(81)
    if device=='cuda':torch.cuda.manual_seed_all(81)
    before=torch.cuda.get_rng_state() if device=='cuda' else torch.random.get_rng_state()
    samples=[]
    for seed in (3,7,17,29):
        y,public,meta=simulate_observation(p,model=model,seed=seed,sigma=.2,I0=20.,sigma_e=3. if model=='poisson_gaussian' else 0.,scale=.7)
        if model=='gaussian':samples.append(((y-p)/.2).flatten())
        else:
            raw=public['counts' if model=='poisson' else 'c_obs'];mu=20*np.exp(-.7*.5)
            samples.append(((raw-mu)/np.sqrt(mu+(9 if model=='poisson_gaussian' else 0))).flatten())
            assert torch.allclose(y,-torch.log(raw.clamp_min(1)/20)/.7,rtol=0,atol=1e-12)
    after=torch.cuda.get_rng_state() if device=='cuda' else torch.random.get_rng_state()
    assert torch.equal(before,after)
    values=torch.cat(samples)
    assert abs(float(values.mean()))<.02
    assert abs(float(values.var())-1)<.02
    assert not torch.equal(samples[0],samples[1])


@pytest.mark.parametrize('device',['cpu', CUDA_DEVICE])
def test_nll_reuse_cost_and_failed_residual(device):
    from test_p1_composable import Matrix
    from inv_framework.solvers import PDHGSolver
    from inv_framework.regularizers import TikhonovRegularizer
    y=torch.tensor([[1.,3.,8.,12.]],device=device,dtype=torch.float64)
    f=TransmissionPoissonFidelity(y,20.,tolerance=1e-12)
    costs=[]
    for _ in range(2):
        result=PDHGSolver(TikhonovRegularizer(),num_iterations=2,tolerance=0,min_value=0).solve_detailed(y,Matrix(torch.eye(4,device=device,dtype=y.dtype),(1,2,2)),fidelity=f)
        costs.append((result.resources['fidelity_prox_calls'],result.resources['fidelity_inner_iterations']))
    assert costs[0]==costs[1]
    assert f.prox_calls==2*costs[0][0]
    assert f.last_residual is not None
    with pytest.raises(FloatingPointError):f.proximal(y*float('nan'),.3)
    assert f.last_residual is None

@pytest.mark.parametrize('device',['cpu', CUDA_DEVICE])
def test_full_prox_module_replacement(device):
    from inv_framework.regularizers.base import Regularizer,ProximalOperator
    from inv_framework.solvers import FISTASolver
    from test_p1_composable import Matrix
    class FullQuadratic(Regularizer,ProximalOperator):
        has_full_proximal=True
        supports_box_proximal=True
        def value(self,x):return .5*x.square().flatten(1).sum(1)
        def proximal(self,x,step_size):return x/(1+step_size)
    y=torch.tensor([[-1.,.4,-.3,1.2]],device=device,dtype=torch.float64)
    result=FISTASolver(FullQuadratic(),strength=.17,num_iterations=1000,tolerance=1e-8,min_value=0).solve_detailed(y,Matrix(torch.eye(4,device=device,dtype=y.dtype),(1,2,2)))
    assert result.status=='converged'
    assert torch.allclose(result.reconstruction.flatten(),y.clamp_min(0).flatten()/1.17,atol=1e-7,rtol=0)


def test_nested_acquisition_settings():
    from inv_framework.fidelity import build_fidelity
    from inv_framework.measurement import measurement_setting
    y=torch.zeros(1,2,dtype=torch.float64)
    public={'counts':torch.tensor([[1.,3.]],dtype=y.dtype),'I0':torch.full_like(y,20.)}
    f=build_fidelity('transmission_poisson_nll',y,public,{'noise_model':'poisson','parameters':{'scale':.7}})
    assert f.scale==.7
    with pytest.raises(ValueError,match='conflicting'):
        measurement_setting({'floor':1.,'parameters':{'floor':2.}},'floor')


def test_negative_objective_divergence_monitor():
    from inv_framework.solvers.base import SolveControl,ConsecutiveStoppingMonitor
    control=SolveControl(max_iterations=10,divergence_enabled=True,divergence_patience=3,divergence_relative_increase_tolerance=.1)
    monitor=ConsecutiveStoppingMonitor(control)
    for i,value in enumerate((-100.,-100.01,-100.02,-100.03),1):
        decision=monitor.observe(i,criteria={'fixed_point':False},relative_change=1.,monitor_value=value)
        assert not decision.diverged


def test_missing_usable_spectral_bound_rejected():
    from inv_framework.operators.base import LinearOperator
    from inv_framework.solvers import PDHGSolver
    class Large(LinearOperator):
        domain_shape=(1,65,65)
        range_shape=(1,)
        def forward(self,x):raise AssertionError('unexpected forward')
        def adjoint(self,y):raise AssertionError('unexpected adjoint')
    result=PDHGSolver().solve_detailed(torch.zeros(1,1),Large())
    assert result.status=='invalid_parameters'


@pytest.mark.parametrize('device',['cpu', CUDA_DEVICE])
def test_masked_common_kernel_rejected(device):
    from test_p1_composable import Matrix
    from inv_framework.fidelity import QuadraticFidelity
    from inv_framework.solvers import ADMMSolver
    from inv_framework.regularizers import TikhonovRegularizer
    from inv_framework.operators.gradient import GradientOperator
    y=torch.ones(1,4,device=device,dtype=torch.float64)
    f=QuadraticFidelity(y,0.)
    result=ADMMSolver(TikhonovRegularizer(GradientOperator((1,2,2))),num_iterations=2).solve_detailed(y,Matrix(torch.eye(4,device=device,dtype=y.dtype),(1,2,2)),fidelity=f)
    assert result.status=='invalid_parameters'


@pytest.mark.parametrize('device',['cpu', CUDA_DEVICE])
@pytest.mark.parametrize('nonnegative',[False,True])
def test_counts_above_incident_and_negative_objective(device,nonnegative):
    from test_p1_composable import Matrix
    from inv_framework.solvers import PDHGSolver
    from inv_framework.regularizers import TikhonovRegularizer
    y=torch.full((1,4),100.,device=device,dtype=torch.float64)
    f=TransmissionPoissonFidelity(y,20.,tolerance=1e-12)
    result=PDHGSolver(TikhonovRegularizer(),strength=.17,num_iterations=3000,tolerance=1e-8,min_value=0 if nonnegative else None).solve_detailed(y,Matrix(torch.eye(4,device=device,dtype=y.dtype),(1,2,2)),fidelity=f)
    root=brentq(lambda x:.17*x+100-20*np.exp(-x),-20,20,xtol=1e-13)
    expected=max(root,0.) if nonnegative else root
    assert result.status=='converged'
    assert torch.allclose(result.reconstruction,torch.full_like(result.reconstruction,expected),atol=1e-6,rtol=0)
    if not nonnegative:assert result.final_objective<0


def test_masked_high_counts_no_artificial_inner_failure():
    c=torch.tensor([100000.,20.],dtype=torch.float64)
    f=TransmissionPoissonFidelity(c,100000.,mask=torch.tensor([0.,1.]),tolerance=1e-12)
    value=torch.tensor([-1000.,2.],dtype=c.dtype)
    assert torch.isfinite(f.value(value))
    assert f.proximal(value,.3)[0]==value[0]
    assert f.dual_prox(value,.3)[0]==0


def test_staging_rejects_mixed_latent_counts(tmp_path):
    import json,h5py
    from pathlib import Path
    from dataclasses import replace
    from inv_framework.measurement import simulate_observation
    from inv_framework.benchmarks.ct_cases import load_ct_case,write_ct_case
    from inverse_agent.cases.staging import stage_ct_case
    case=load_ct_case('parallel_2d/disk_analytic_32',data_root=Path(__file__).resolve().parents[1]/'test/data')
    y,public,meta=simulate_observation(case.measurement_clean,model='poisson_gaussian',I0=20.,sigma_e=2.,seed=17)
    case=replace(case,measurement=y,public_measurement=public,metadata={**case.metadata,'measurement':meta})
    record=write_ct_case(case,tmp_path/'case');record['path']='case'
    catalog=tmp_path/'catalog.json';catalog.write_text(json.dumps({'schema_version':'1.0','cases':[record]}))
    with h5py.File(tmp_path/'case/arrays.h5','a') as h:h.create_dataset('measurement/counts',data=np.zeros(y.shape))
    with pytest.raises(ValueError,match='latent counts'):
        stage_ct_case(catalog,case.case_id,tmp_path/'staged')


def test_case_rejects_conflicting_acquisition_settings():
    from dataclasses import replace
    from pathlib import Path
    from inv_framework.benchmarks.ct_cases import load_ct_case
    case=load_ct_case('parallel_2d/disk_analytic_32',data_root=Path(__file__).resolve().parents[1]/'test/data')
    with pytest.raises(ValueError,match='conflicting'):
        replace(case,metadata={**case.metadata,'measurement':{'noise_model':'poisson','scale':1.,'parameters':{'scale':2.}}})


def test_l2_runtime_uses_measurement_mask():
    from dataclasses import replace
    from pathlib import Path
    from inv_framework.benchmarks.ct_cases import load_ct_case
    from inv_framework.ct_runtime import build_solver
    case=load_ct_case('parallel_2d/disk_analytic_32',data_root=Path(__file__).resolve().parents[1]/'test/data')
    mask=torch.ones_like(case.measurement,dtype=torch.bool);mask[...,0,:]=False
    case=replace(case,valid_measurement_mask=mask)
    solver=build_solver('fista',{'data_fidelity':'l2'},case)
    prediction=case.measurement+(~mask).to(case.measurement)
    assert solver.fidelity.value(prediction)==0
    assert bool((solver.fidelity.gradient(prediction)==0).all())
    full=replace(case,valid_measurement_mask=torch.ones_like(mask))
    assert build_solver('fista',{'data_fidelity':'l2'},full).fidelity is None


def test_exact_nll_rejects_electronic_noise_and_background():
    from inv_framework.fidelity import build_fidelity
    y=torch.zeros(1,2,dtype=torch.float64)
    public={'counts':torch.ones_like(y),'I0':torch.full_like(y,20.),'sigma_e':torch.ones_like(y)}
    with pytest.raises(ValueError,match='sigma_e'):
        build_fidelity('transmission_poisson_nll',y,public,{'noise_model':'poisson'})
    public.pop('sigma_e')
    with pytest.raises(ValueError,match='background'):
        build_fidelity('transmission_poisson_nll',y,public,{'noise_model':'poisson','parameters':{'background':1.}})


@pytest.mark.parametrize('device',['cpu', CUDA_DEVICE])
def test_h1_real_operator_budget(device):
    from test_p1_composable import Matrix
    from inv_framework.operators.gradient import GradientOperator
    from inv_framework.instrumentation import CountingLinearOperator
    from inv_framework.solvers import TikhonovSolver
    class Spy(Matrix):
        forwards=0
        adjoints=0
        def forward(self,x):
            self.forwards+=1
            return super().forward(x)
        def adjoint(self,x):
            self.adjoints+=1
            return super().adjoint(x)
    y=torch.tensor([[1.,2.,3.,4.]],device=device,dtype=torch.float64)
    def operator():return Spy(torch.diag(torch.tensor([1.1,1.2,1.3,1.4],device=device,dtype=y.dtype)),(1,2,2))
    raw=operator();counted=CountingLinearOperator(raw)
    result=TikhonovSolver(regularization_operator=GradientOperator((1,2,2)),num_iterations=2,tolerance=0).solve_detailed(y,counted)
    assert counted.stats()['forward_calls']==raw.forwards
    assert counted.stats()['adjoint_calls']==raw.adjoints
    assert result.resources['auxiliary_D_forward_calls']>0
    limit=raw.forwards
    constrained=CountingLinearOperator(operator(),max_forward_calls=limit)
    budget_result=TikhonovSolver(regularization_operator=GradientOperator((1,2,2)),num_iterations=2,tolerance=0).solve_detailed(y,constrained)
    assert budget_result.actual_iterations==2
    assert budget_result.status!='resource_exhausted'


def test_weighted_cg_rejects_constraints():
    from test_p1_composable import Matrix
    from inv_framework.solvers import TikhonovSolver
    from inv_framework.fidelity import QuadraticFidelity
    y=torch.ones(1,4,dtype=torch.float64)
    result=TikhonovSolver(min_value=0).solve_detailed(y,Matrix(torch.eye(4,dtype=y.dtype),(1,2,2)),fidelity=QuadraticFidelity(y))
    assert result.status=='invalid_parameters'

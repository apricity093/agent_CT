"""Directed numerical evidence for shared resident-array mathematics."""
import json
import os
from pathlib import Path
import torch
import numpy as np
import pytest
from scipy.optimize import minimize
from inv_framework.core.array_math import Differences, shrink, inner, norm
from inv_framework.core.operator import GradientOperator, HaarOperator, LinearOperator
from inv_framework.core.regularizer import TVRegularizer, HaarL1Regularizer, TikhonovRegularizer, GradientHuberRegularizer
from inv_framework.core.krylov import solve_cg
from inv_framework.core.norms import power_norm
from inv_framework.solvers.composite import quadratic_composite

EVIDENCE = {}


@pytest.fixture(scope='module', autouse=True)
def evidence():
    yield
    output = os.environ.get('MATH_OUTPUT')
    if output:
        Path(output).mkdir(parents=True, exist_ok=True)
        Path(output, 'numerics.json').write_text(json.dumps(EVIDENCE, indent=2), encoding='utf-8')


class Matrix(LinearOperator):
    def __init__(self, matrix, shape):
        self.matrix = matrix; self.domain_shape = shape; self.range_shape = (matrix.shape[0],)
    def forward(self, x):
        return x.flatten(1)@self.matrix.T
    def adjoint(self, y):
        return (y@self.matrix.conj()).reshape(y.shape[0], *self.domain_shape)


def explicit(op, shape, dtype=torch.float64):
    eye = torch.eye(int(np.prod(shape)), dtype=dtype).reshape(-1, *shape)
    return op.forward(eye).flatten(1).T


def adjoint_error(op, x, y):
    lhs = inner(op.forward(x), y); rhs = inner(x, op.adjoint(y))
    return float(abs(lhs-rhs)/abs(lhs))


@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128, torch.float32, torch.complex64])
def test_gradient_haar_norms(dtype):
    torch.manual_seed(91)
    x = torch.randn(1, 2, 4, 4, 4, dtype=dtype)
    d = GradientOperator(x.shape[1:], axes=(-3,-2,-1), spacing=(.7,1.3,2.))
    y = torch.randn_like(d.forward(x)); h = HaarOperator(x.shape[1:], 2)
    errors = [adjoint_error(d,x,y), adjoint_error(h,x,torch.randn_like(x))]
    assert max(errors) < 1e-5
    matrix = explicit(d,x.shape[1:],dtype)
    exact = float(torch.linalg.matrix_norm(matrix,2)**2)
    assert exact <= d.squared_norm_upper_bound(x)*(1+1e-6)
    assert abs(float(h.forward(x).norm()/x.norm())-1) < 1e-6
    estimate = power_norm(d,x,80,1e-8)
    assert abs(estimate['value']/exact-1) < .01
    EVIDENCE[str(dtype)] = dict(adjoint_errors=errors, exact_squared_norm=exact, upper_bound=d.differences.bound, power_estimate=estimate)


def test_periodic_equivalence_and_complex_shrink():
    rng = np.random.default_rng(8); x = rng.normal(size=(1,4,4))+1j*rng.normal(size=(1,4,4))
    d = Differences((-2,-1),(.5,2.), 'periodic'); y = rng.normal(size=d.forward(x).shape)+1j*rng.normal(size=d.forward(x).shape)
    error = adjoint_error(d,x,y); assert error < 1e-5
    from inv_framework.core.numpy_math import gradient, gradient_adjoint, laplacian_spectrum
    old = np.stack([np.roll(x,-1,a)-x for a in (-2,-1)])
    np.testing.assert_allclose(gradient(x,(-2,-1)),old)
    unit = Differences((-2,-1),boundary='periodic')
    rhs = x; rho = .7
    normal = lambda v: v+rho*unit.adjoint(unit.forward(v))
    fft_solution = np.fft.ifftn(np.fft.fftn(rhs)/(1+rho*laplacian_spectrum(x.shape,(-2,-1))))
    cg = solve_cg(normal,rhs,np.zeros_like(x),100,atol=1e-11,rtol=0.)
    assert cg.converged and norm(cg.reconstruction-fft_solution) < 1e-10
    z = shrink(x,.2); residual = z-x+.2*z/np.maximum(abs(z),1e-30)
    assert np.linalg.norm(residual[abs(z)>0]) < 1e-12
    EVIDENCE['periodic'] = dict(adjoint_error=error, fft_cg_error=norm(cg.reconstruction-fft_solution), cg=cg.diagnostics())


def test_physical_adjoint_and_direction():
    from inv_framework.ct.backends.radon_torch import ParallelBeamRadon2D
    from inv_framework.mri.operators import SenseEncoding
    from inv_framework.deconvolution.operators import Convolution
    torch.manual_seed(5)
    op = ParallelBeamRadon2D(8, angles=torch.arange(7)*torch.pi/7, device='cpu', in_channels=1)
    x = torch.randn(1,1,8,8); y = torch.randn_like(op.forward(x))
    ct = adjoint_error(op,x,y); assert ct < 1e-5
    rng = np.random.default_rng(54)
    maps = rng.normal(size=(1,2,4,4))+1j*rng.normal(size=(1,2,4,4))
    sense = SenseEncoding(maps, mask=(rng.random((1,1,4,4))>.3))
    x = rng.normal(size=(1,4,4))+1j*rng.normal(size=(1,4,4)); y = rng.normal(size=(1,2,4,4))+1j*rng.normal(size=(1,2,4,4))
    mri = adjoint_error(sense,x,y); assert mri < 1e-5
    direction = rng.normal(size=x.shape)+1j*rng.normal(size=x.shape)
    weights = rng.uniform(.3,2.,size=y.shape)
    value = lambda a: .5*np.sum(weights*abs(sense.forward(a)-y)**2)
    analytical = inner(direction,sense.adjoint(weights*(sense.forward(x)-y))).real
    finite = (value(x+1e-6*direction)-value(x-1e-6*direction))/2e-6
    derivative = abs(finite-analytical)/max(1,abs(analytical)); assert derivative < 1e-7
    conv = Convolution((1,4,4),np.array([[.1,.2],[.3,.4]]),(0,1),(-2,-1),'periodic')
    cx = rng.normal(size=(1,4,4)); cy = rng.normal(size=cx.shape)
    convolution = adjoint_error(conv,cx,cy); assert convolution < 1e-5
    EVIDENCE['physical'] = dict(ct=ct,sense=mri,convolution=convolution,direction_derivative=derivative)


def test_quadratic_cg_and_budget():
    rng = np.random.default_rng(4); a=rng.normal(size=(16,16))+1j*rng.normal(size=(16,16))
    h=a.conj().T@a+np.eye(16); b=rng.normal(size=16)+1j*rng.normal(size=16)
    direct=np.linalg.solve(h,b)
    cold=solve_cg(lambda x:h@x,b,np.zeros_like(b),100,atol=1e-10,rtol=0.)
    warm=solve_cg(lambda x:h@x,b,cold.reconstruction,100,atol=1e-10,rtol=0.)
    failed=solve_cg(lambda x:h@x,b,np.zeros_like(b),1,atol=1e-12,rtol=0.)
    assert cold.converged and norm(cold.reconstruction-direct)<1e-9 and warm.iterations==0
    assert not failed.converged and failed.stopping_reason=='inner_budget_exhausted'
    EVIDENCE['cg'] = dict(cold=cold.diagnostics(),warm=warm.diagnostics(),failed=failed.diagnostics())


@pytest.mark.parametrize('dtype', [torch.float64, torch.float32])
def test_joint_objective_and_structure(dtype):
    torch.manual_seed(7)
    shape=(1,4,4); n=16
    op=Matrix(torch.eye(n,dtype=dtype),shape)
    y=torch.randn(1,n,dtype=dtype)
    tv=TVRegularizer('anisotropic'); haar=HaarL1Regularizer(1)
    d=explicit(GradientOperator(shape),shape).numpy()
    w=explicit(HaarOperator(shape,1),shape).numpy()
    mask=haar.coefficient_mask(torch.ones(1,*shape)).flatten().numpy()
    k=np.concatenate([.08*d,.06*mask[:,None]*w]); yn=y.double().numpy().ravel()
    # Independent dual bound-constrained quadratic reference for analysis L1.
    fun=lambda p: (.5*np.linalg.norm(k.T@p-yn)**2, k@(k.T@p-yn))
    ref=minimize(fun,np.zeros(k.shape[0]),jac=True,bounds=[(-1,1)]*k.shape[0],method='L-BFGS-B',options=dict(ftol=1e-15,gtol=1e-12,maxiter=20000,maxls=100))
    xr=yn-k.T@ref.x
    ref_kkt=np.linalg.norm(ref.x-np.clip(ref.x+k@xr,-1,1))
    assert ref_kkt < 1e-5
    tol=1e-8 if dtype==torch.float64 else 2e-5
    # Float32 can reach its arithmetic floor before an arbitrarily small k^-2 target.
    eta0=1e-5 if dtype==torch.float64 else 3e-2
    result=quadratic_composite(op,y,[(tv,.08,1.),(haar,.06,.8)],x_init=torch.zeros(1,*shape,dtype=dtype),tolerance=tol,eta0=eta0,max_iterations=1500)
    x=result.reconstruction.flatten().double().numpy()
    objective=lambda a: .5*np.linalg.norm(a-yn)**2+np.sum(abs(k@a))
    gap=abs(objective(x)-objective(xr))/max(1,abs(objective(xr)))
    p=[]
    # Recover certificate from primal x via independent dual reference.
    stationarity=np.sqrt(np.linalg.norm(x-yn+k.T@ref.x)**2+np.linalg.norm(ref.x-np.clip(ref.x+k@x,-1,1))**2)/max(1,np.linalg.norm(yn))
    threshold=1e-5 if dtype==torch.float64 else 1e-3
    assert gap < threshold and stationarity < threshold
    assert abs(result.final_objective-(result.metadata['data_value']+sum(result.metadata['regularizer_contributions']))) < 1e-6
    zero=quadratic_composite(op,y,[(tv,.08,1.),(haar,0.,.8)],x_init=torch.zeros(1,*shape,dtype=dtype),tolerance=tol,eta0=eta0)
    single=quadratic_composite(op,y,[(tv,.08,1.)],x_init=torch.zeros(1,*shape,dtype=dtype),tolerance=tol,eta0=eta0)
    assert torch.equal(zero.reconstruction,single.reconstruction) and zero.metadata['active_branches']==1
    split_value=.03*tv.value(result.reconstruction)+.05*tv.value(result.reconstruction)
    assert float(abs(split_value-.08*tv.value(result.reconstruction))) < 1e-6
    constrained=quadratic_composite(op,y,[(tv,.08,1.)],x_init=torch.zeros(1,*shape,dtype=dtype),min_value=0.,max_value=.5,tolerance=tol,eta0=eta0)
    assert bool(((constrained.reconstruction>=0)&(constrained.reconstruction<=.5)).all())
    exhausted=quadratic_composite(op,y,[(tv,.08,1.)],x_init=torch.zeros(1,*shape,dtype=dtype),cg_budget=1,eta0=1e-12)
    assert exhausted.status!='converged' and exhausted.stopping_reason=='inner_budget_exhausted'
    EVIDENCE['joint_'+str(dtype)] = dict(objective_gap=gap,optimality=stationarity,reference_kkt=ref_kkt,status=result.status,reason=result.stopping_reason,iterations=result.actual_iterations,objective=result.final_objective,metadata={k:v for k,v in result.metadata.items() if k!='cg_subproblems'},last_cg=result.metadata['cg_subproblems'][-1],budget_exit=exhausted.stopping_reason)


def test_huber_and_quadratic_complex_definitions():
    torch.manual_seed(4); x=torch.randn(1,1,4,4,dtype=torch.complex128); direction=torch.randn_like(x)
    for reg in (TikhonovRegularizer(),TikhonovRegularizer(GradientOperator((1,4,4))),GradientHuberRegularizer(.4,grouped=True)):
        fd=(reg.value(x+1e-6*direction)-reg.value(x-1e-6*direction))/2e-6
        assert abs(float(fd)-float(inner(direction,reg.gradient(x)).real))<1e-7
    for mode in ('isotropic','anisotropic'):
        reg=TVRegularizer(mode); z=torch.randn(1,2,1,4,4,dtype=x.dtype)
        prox=reg.coefficient_proximal(z,.2)
        assert torch.isfinite(prox).all()


def test_steps_and_scaled_stacks():
    from inv_framework.solvers import PDHGSolver, FISTASolver
    from inv_framework.core.regularizer import ZeroRegularizer
    torch.manual_seed(19)
    shape=(1,4,4); x=torch.randn(1,*shape,dtype=torch.float64)
    d=GradientOperator(shape,spacing=(.6,1.7)); w=HaarOperator(shape,1)
    stack=torch.cat([.3*explicit(d,shape).to(torch.complex128),2j*explicit(w,shape)],0)
    exact=float(torch.linalg.matrix_norm(stack,2)**2); bound=.3**2*d.differences.bound+4
    assert exact<=bound
    a=torch.diag(torch.linspace(.5,2,16,dtype=x.dtype)); op=Matrix(a,shape); y=torch.randn(1,16,dtype=x.dtype)
    pdhg=PDHGSolver(TVRegularizer(spacing=(.6,1.7)),num_iterations=2).solve_detailed(y,op)
    md=pdhg.metadata
    assert md['tau']*md['sigma']*md['norm_bound_squared']<1
    fista=FISTASolver(TikhonovRegularizer(),num_iterations=3,step_size=10.).solve_detailed(y,op)
    assert fista.metadata['smooth_majorizer_checked']
    zero=PDHGSolver(ZeroRegularizer(),num_iterations=2).solve_detailed(y,Matrix(torch.zeros_like(a),shape))
    assert zero.status!='numerical_error' and zero.metadata['norm_bound_squared']==0
    EVIDENCE['steps']=dict(stacked_exact=exact,stacked_bound=bound,pdhg_bound=md['norm_bound_squared'],tau=md['tau'],sigma=md['sigma'],product=md['tau']*md['sigma']*md['norm_bound_squared'],fista_step=fista.metadata['step_size'],zero_step=zero.metadata['tau'])


def test_restarted_warm_cg_sequence():
    import time
    rng=np.random.default_rng(11); a=rng.normal(size=(40,40)); h=a.T@a+np.eye(40)*5
    b=rng.normal(size=40); directions=rng.normal(size=(6,40))*1e-4
    totals=[]; durations=[]
    for warm in (False, True):
        x=np.zeros_like(b); steps=0; started=time.perf_counter()
        for direction in directions:
            rhs=b+direction
            result=solve_cg(lambda v:h@v,rhs,x if warm else np.zeros_like(b),100,atol=1e-6,rtol=0.)
            assert result.converged
            x=result.reconstruction; steps+=result.iterations
        totals.append(steps); durations.append(time.perf_counter()-started)
    assert totals[1]<totals[0]
    EVIDENCE['warm_sequence']=dict(cold_iterations=totals[0],warm_iterations=totals[1],cold_seconds=durations[0],warm_seconds=durations[1],target=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_residency_and_classical_graph():
    a=torch.diag(torch.linspace(1,3,16,device='cuda',dtype=torch.float64))
    b=torch.ones(16,device='cuda',dtype=torch.float64,requires_grad=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
        result=solve_cg(lambda v:a@v,b,torch.zeros_like(b),50,atol=1e-10,rtol=0.)
        torch.cuda.synchronize()
    assert result.converged and result.reconstruction.device==b.device and result.reconstruction.dtype==b.dtype
    assert not result.reconstruction.requires_grad
    # Scalar reductions may synchronize; no full image host/device copies.
    transfers=[e for e in prof.events() if e.name=='aten::_to_copy']
    assert all(not e.input_shapes[0] or int(np.prod(e.input_shapes[0]))==1 for e in transfers)
    EVIDENCE['cuda']=dict(cg=result.diagnostics(),device=str(b.device),dtype=str(b.dtype),scalar_copy_events=len(transfers),profile_cpu_us=sum(e.self_cpu_time_total for e in prof.key_averages()))

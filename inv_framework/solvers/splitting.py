"""Composite two-block observation-split ADMM and primal-dual reconstruction."""
import math
import time
import torch
from ..instrumentation import OperatorBudgetExceeded
from ..operators.gradient import GradientOperator
from ..regularizers import TikhonovRegularizer
from .base import InverseProblemSolver, IterationRecorder, ConsecutiveStoppingMonitor, resolve_control
from ._utils import prepare_initial_image, require_linear_operator, validate_measurement_shape

class Penalty:
    def __init__(self, reg, shape):
        self.reg = reg
        self.accounting_operator = None
        self.calls = dict(transform_forward_calls=0, transform_adjoint_calls=0, coefficient_prox_calls=0,
                          transform_seconds=0., coefficient_prox_seconds=0.)
        if not all(callable(getattr(reg, name, None)) for name in
                   ('split_transform', 'coefficient_value', 'coefficient_proximal')):
            raise ValueError('unsupported split regularizer: missing mathematical capabilities')
        self.transform, self.bound = reg.split_transform(shape)

    def apply(self, x, adjoint=False):
        if self.bound == 0:
            return torch.zeros((x.shape[0], *self.shape), dtype=x.dtype, device=x.device) if adjoint else x.new_zeros((x.shape[0], 0))
        if x.is_cuda: torch.cuda.synchronize(x.device)
        start = time.perf_counter()
        self.calls['transform_adjoint_calls' if adjoint else 'transform_forward_calls'] += 1
        result = x if self.transform is None else (self.transform.adjoint(x) if adjoint else self.transform.forward(x))
        if x.is_cuda: torch.cuda.synchronize(x.device)
        self.calls['transform_seconds'] += time.perf_counter()-start
        return result

    def mask(self, z):
        mask = getattr(self.reg, 'coefficient_mask', None)
        return mask(z) if callable(mask) else torch.ones_like(z)

    def value(self, z):
        if self.bound == 0: return z.new_tensor(0.)
        return self.reg.coefficient_value(z).sum()

    def prox(self, z, t):
        if self.bound == 0: return z
        if z.is_cuda: torch.cuda.synchronize(z.device)
        start = time.perf_counter()
        self.calls['coefficient_prox_calls'] += 1
        failed=True
        try:
            out=self.reg.coefficient_proximal(z,t)
            failed=False
            return out
        finally:
            if z.is_cuda: torch.cuda.synchronize(z.device)
            elapsed=time.perf_counter()-start
            self.calls['coefficient_prox_seconds']+=elapsed
            record=getattr(self.accounting_operator,'record_prox',None)
            if callable(record):record(iterations=0,elapsed_seconds=elapsed,failed=failed)

    def dual_prox(self, z, sigma, weight):
        if self.bound == 0: return z
        # Moreau also handles zero-weight LL: its conjugate is the indicator {0}.
        result = z-sigma*self.prox(z/sigma,weight/sigma)
        result = result*self.mask(result)
        if weight == 0: result = torch.zeros_like(result)
        return result


def norm_bound(operator, x):
    """Schur bound using absolute row/column sums; signed generic matrices use columns."""
    return operator.squared_norm_upper_bound(x)


def solve_split_detailed(operator, measurement, *, algorithm, regularizer=None, strength=None,
        reg_strength=None, num_iterations=200, tolerance=1e-5, rho=1., cg_max_iterations=100,
        cg_tolerance=1e-6, tau=None, sigma=None, min_value=None, max_value=None,
        x_init=None, control=None, callback=None, endpoint_state=None, fidelity=None):
    from .detailed import _parameter_failure_result, _common_parameter_errors, _policy_status
    require_linear_operator(operator,algorithm)
    validate_measurement_shape(measurement,operator,algorithm)
    from ..fidelity import QuadraticFidelity
    weight = strength if strength is not None else reg_strength if reg_strength is not None else .01
    reg = regularizer if regularizer is not None else TikhonovRegularizer()
    from ..regularizers import ZeroRegularizer
    from ._utils import apply_box_constraints
    if weight == 0: reg = ZeroRegularizer()
    if reg == "h1": reg = TikhonovRegularizer(GradientOperator(operator.domain_shape))
    errors = _common_parameter_errors(num_iterations,min_value,max_value)
    if strength is not None and reg_strength is not None and strength != reg_strength:
        errors.append('strength and reg_strength conflict')
    for name,v,positive in [('strength',weight,False),('tolerance',tolerance,False),('rho',rho,True),('cg_tolerance',cg_tolerance,True)]:
        if not math.isfinite(float(v)) or (v<=0 if positive else v<0): errors.append(name+' invalid')
    for name,v in [('tau',tau),('sigma',sigma)]:
        if v is not None and (not math.isfinite(float(v)) or v<=0): errors.append(name+' must be positive finite')
    if isinstance(cg_max_iterations,bool) or not isinstance(cg_max_iterations,int) or cg_max_iterations<=0:
        errors.append('cg_max_iterations must be positive integer')
    try: penalty = Penalty(reg,operator.domain_shape)
    except ValueError as e: errors.append(str(e))
    if errors:
        return _parameter_failure_result(algorithm,measurement,operator,x_init=x_init,errors=errors,max_iterations=num_iterations,parameters={})
    penalty.accounting_operator = operator
    penalty.shape = operator.domain_shape
    control = resolve_control(control,default_iterations=num_iterations,default_tolerance=tolerance,callback=callback)
    x = prepare_initial_image(measurement,operator,x_init=x_init,initial_value=0.)
    nonnegative = min_value == 0
    constrained = min_value is not None or max_value is not None
    project = lambda v: apply_box_constraints(v, min_value=min_value, max_value=max_value)
    x = project(x)
    recorder = IterationRecorder(control,measurement,operator,algorithm=algorithm); recorder.set_initial(x)
    monitor = ConsecutiveStoppingMonitor(control)
    norm = lambda v: float(v.norm())
    fidelity_before = {k:getattr(fidelity,k,0) for k in ("prox_calls","inner_iterations","inner_failures","prox_seconds","weight_multiplications","mask_calls")}
    cg_started = None
    actual=0; cg_total=0; cg_seconds=0.; cg_failures=0
    status,reason='max_iterations','maximum_iterations_reached'
    prediction=residual=objective=metric=None
    endpoint=False; interruption=None; bound=None; primal=dual=None; inner_relative=None
    tol=float(control.tolerance or 0)
    try:
        if not bool(torch.isfinite(x).all() and torch.isfinite(measurement).all()):
            raise FloatingPointError('non_finite_input')
        fidelity = fidelity if fidelity is not None else QuadraticFidelity(measurement)
        observation_split = algorithm == 'admm'
        if hasattr(fidelity,"accounting_operator"): fidelity.accounting_operator=operator
        z=penalty.apply(x); u=torch.zeros_like(z); q=x.clone(); v=torch.zeros_like(x)
        if algorithm=='pdhg':
            try:
                bound=norm_bound(operator,x)+penalty.bound
            except ValueError as error:
                return _parameter_failure_result(algorithm,measurement,operator,x_init=x_init,errors=[str(error)],max_iterations=num_iterations,parameters={})
            tau=float(tau) if tau is not None else .99/math.sqrt(bound)
            sigma=float(sigma) if sigma is not None else .99/math.sqrt(bound)
            if tau*sigma*bound>=1:
                return _parameter_failure_result(algorithm,measurement,operator,x_init=x_init,
                    errors=['tau*sigma*norm_bound_squared must be < 1'],max_iterations=num_iterations,parameters={})
            p=torch.zeros_like(measurement); d=torch.zeros_like(z); bar=x.clone()
        else:
            if not nonnegative and callable(getattr(fidelity,'validate_normal',None)):
                try:fidelity.validate_normal(operator,penalty.transform,x)
                except ValueError as error:
                    return _parameter_failure_result(algorithm,measurement,operator,x_init=x_init,errors=[str(error)],max_iterations=num_iterations,parameters={})
            if observation_split:
                observation=operator.forward(x); observation_dual=torch.zeros_like(observation)
            else: aty=operator.adjoint(fidelity.linear_rhs())
        for iteration in range(1,min(num_iterations,control.max_iterations)+1):
            old=x
            if algorithm=='admm':
                rhs=(rho*operator.adjoint(observation-observation_dual) if observation_split else aty)+rho*penalty.apply(z-u,True)+rho*(q-v)
                def normal(a):
                    prediction_a=operator.forward(a)
                    return operator.adjoint(rho*prediction_a if observation_split else fidelity.normal_gradient(prediction_a))+rho*penalty.apply(penalty.apply(a),True)+rho*a
                cg_started=time.perf_counter()
                candidate=x.clone(); r=rhs-normal(candidate); direction=r.clone(); rr=(r*r).sum()
                # Summable inner relative tolerance, bounded by user requested accuracy.
                target=cg_tolerance/iteration**2*max(norm(rhs),1.)
                inner=0
                while norm(r)>target and inner<cg_max_iterations:
                    nd=normal(direction); curvature=(direction*nd).sum()
                    if not bool(torch.isfinite(curvature)) or float(curvature)<=0:
                        raise FloatingPointError('cg_nonpositive_curvature_or_breakdown')
                    alpha=rr/curvature; candidate=candidate+alpha*direction; r=r-alpha*nd
                    newrr=(r*r).sum(); direction=r+(newrr/rr)*direction; rr=newrr
                    inner+=1; cg_total+=1
                cg_seconds+=time.perf_counter()-cg_started
                cg_started=None
                inner_relative=norm(r)/max(norm(rhs),1.)
                if norm(r)>target:
                    cg_failures+=1; status,reason='max_iterations','cg_inner_max_iterations'; break
                tx=penalty.apply(candidate)
                nz=penalty.prox(tx+u,weight/rho)
                nq=project(candidate+v)
                nu=u+tx-nz; nv=v+candidate-nq
                obs_primal=0.
                ds=rho*(penalty.apply(nz-z,True)+(nq-q))
                if observation_split:
                    ax=operator.forward(candidate)
                    next_observation=fidelity.proximal(ax+observation_dual, 1/rho)
                    next_observation_dual=observation_dual+ax-next_observation
                    obs_primal=norm(ax-next_observation)**2
                    ds=ds+rho*operator.adjoint(next_observation-observation)
                primal=math.sqrt(norm(tx-nz)**2+norm(candidate-nq)**2+obs_primal)
                dual=norm(ds)
                eps_p=tol*(math.sqrt(tx.numel()+candidate.numel()+(ax.numel() if observation_split else 0))+max(
                    math.sqrt(norm(tx)**2+norm(candidate)**2+(norm(ax)**2 if observation_split else 0)),
                    math.sqrt(norm(nz)**2+norm(nq)**2+(norm(next_observation)**2 if observation_split else 0))))
                multiplier=rho*(penalty.apply(nu,True)+nv)
                if observation_split: multiplier=multiplier+rho*operator.adjoint(next_observation_dual)
                eps_d=tol*(math.sqrt(x.numel())+norm(multiplier))
                # Commit all split variables together; return feasible q for nonnegative runs.
                x,z,u,q,v=candidate,nz,nu,nq,nv
                if observation_split: observation,observation_dual=next_observation,next_observation_dual
                metric=max(primal/max(eps_p,1e-30),dual/max(eps_d,1e-30))
                criteria={'primal':primal<=eps_p,'dual':dual<=eps_d}
            else:
                np=fidelity.dual_prox(p+sigma*operator.forward(bar),sigma)
                nd=penalty.dual_prox(d+sigma*penalty.apply(bar),sigma,weight)
                nx=project(x-tau*(operator.adjoint(np)+penalty.apply(nd,True)))
                # Complete dual and primal update before committing.
                bar=2*nx-x; x,p,d=nx,np,nd
                criteria={}
            if endpoint_state is not None: endpoint_state.clear()
            endpoint=False
            actual=iteration
            returned=q if algorithm=='admm' and constrained else x
            # Independent returned-point fixed-point / KKT check.
            prediction=operator.forward(returned); residual=prediction-measurement
            txr=penalty.apply(returned)
            objective=float(fidelity.value(prediction))+weight*float(penalty.value(txr))
            if algorithm=='pdhg':
                pg=operator.adjoint(p)+penalty.apply(d,True)
                rx=(returned-project(returned-tau*pg))/tau
                rp=(p-fidelity.dual_prox(p+sigma*prediction,sigma))/sigma
                rd=(d-penalty.dual_prox(d+sigma*txr,sigma,weight))/sigma
                metric=math.sqrt(norm(rx)**2+norm(rp)**2+norm(rd)**2)/max(1.,norm(returned),norm(p),norm(d))
                criteria={'primal_dual_fixed_point':metric<=tol}
            if not bool(torch.isfinite(returned).all()) or not math.isfinite(objective+metric):
                raise FloatingPointError('non_finite_split_state')
            if endpoint_state is not None:
                endpoint_state.clear()
                endpoint_state.update(p=p.detach().clone() if algorithm=='pdhg' else (rho*observation_dual if observation_split else fidelity.gradient(prediction)).detach().clone(),
                    d=d.detach().clone() if algorithm=='pdhg' else (rho*u).detach().clone(),
                    penalty=penalty, fidelity=fidelity, tau=tau if algorithm=='pdhg' else 1.,
                    sigma=sigma if algorithm=='pdhg' else 1., weight=weight, nonnegative=nonnegative,
                    min_value=min_value, max_value=max_value)
            endpoint=True
            if control.discrepancy_target is not None:
                criteria["discrepancy"] = norm(residual)/max(norm(measurement),1e-12)<=control.discrepancy_target
            decision=monitor.observe(iteration,criteria=criteria,relative_change=norm(x-old)/max(norm(old),1.),monitor_value=objective)
            if not recorder.record(iteration,returned,residual=residual,objective=objective,algorithm_residual=metric,
                    criteria=criteria,stopping_candidate=decision.converged,consecutive_criteria_count=decision.consecutive,
                    native_criterion_name='split_residual',native_criterion_value=metric,
                    metadata={'primal_residual':primal,'dual_residual':dual,'complete_objective':True}):
                status,reason='cancelled','callback_cancelled'; break
            terminal, terminal_reason = _policy_status(decision)
            if terminal:
                status, reason = terminal, terminal_reason
                break
            if decision.converged and control.stop_on_convergence:
                status,reason='converged','split_residual_patience'; break
    except OperatorBudgetExceeded as e:
        status,reason='resource_exhausted','wall_time_budget_exhausted' if e.kind=='wall' else 'operator_call_budget_exhausted'
        interruption={'kind':e.kind,'limit':e.limit,'attempted':e.attempted}
    except RuntimeError as e:
        if str(e) not in {'poisson_prox_inner_max_iterations','poisson_prox_returned_precision'}: raise
        status,reason='max_iterations','poisson_prox_inner_max_iterations'
        endpoint=False
    except FloatingPointError as e:
        status,reason='numerical_error',str(e); recorder.mark_numerical_error(reason)
    if cg_started is not None: cg_seconds += time.perf_counter()-cg_started
    returned=q if algorithm=='admm' and constrained and 'q' in locals() else x
    return recorder.finish(returned,actual_iterations=actual,status=status,stopping_reason=reason,
        final_objective=objective if endpoint else None,final_residual=norm(residual)/max(norm(measurement),1e-12) if endpoint else None,
        predicted_measurement=prediction if endpoint else None,
        resources={**{'fidelity_'+k:getattr(fidelity,k,0)-v for k,v in fidelity_before.items()},**penalty.calls,'transform_kind': 'I' if penalty.transform is None else getattr(penalty.transform,'transform_kind','linear'),'cg_iterations':cg_total,'cg_seconds':cg_seconds,'cg_unconverged_subproblems':cg_failures},
        metadata={'native_termination_managed':True,'endpoint_diagnostics_complete':endpoint,
            'budget_interruption':interruption,'strength':weight,'rho':rho,'tau':tau,'sigma':sigma,
            'norm_bound_squared':bound,'norm_bound_is_upper_bound':True,'primal_residual':primal,'dual_residual':dual,
            'split_residual':metric if endpoint else None,'cg_relative_residual':inner_relative,
            'fidelity_precision':fidelity.precision_report() if callable(getattr(fidelity,'precision_report',None)) else None,
            'regularizer':type(reg).__name__,'delta':getattr(reg,'delta',None),'theta':1})

class ADMMSolver(InverseProblemSolver):
    algorithm='admm'
    def __init__(self,regularizer=None,**parameters): self.parameters=dict(regularizer=regularizer,**parameters)
    def solve_detailed(self,measurement,operator,x_init=None,control=None,callback=None,**kwargs):
        self.endpoint_state = {}
        return solve_split_detailed(operator,measurement,algorithm=self.algorithm,endpoint_state=self.endpoint_state,x_init=x_init,control=control,callback=callback,**{**self.parameters,**kwargs})
    def solve(self,measurement,operator,x_init=None,**kwargs):
        result=self.solve_detailed(measurement,operator,x_init=x_init,**kwargs)
        if result.status=='invalid_parameters': raise ValueError(result.stopping_reason)
        if result.status=='numerical_error': raise FloatingPointError(result.stopping_reason)
        if result.status=='resource_exhausted': raise OperatorBudgetExceeded(result.stopping_reason,**result.metadata['budget_interruption'])
        return result.reconstruction

class PDHGSolver(ADMMSolver):
    algorithm='pdhg'

def admm(operator,y,**kwargs): return ADMMSolver(**kwargs).solve(y,operator)
def pdhg(operator,y,**kwargs): return PDHGSolver(**kwargs).solve(y,operator)

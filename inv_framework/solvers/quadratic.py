"""Matrix-free quadratic baselines with one shared Hermitian augmentation."""
from dataclasses import replace
import math
import torch
from inv_framework.core.array_math import norm, inner
from inv_framework.core.operator import GradientOperator
from inv_framework.core.krylov import solve_cg
from inv_framework.solvers.base import resolve_control, IterationRecorder, ConsecutiveStoppingMonitor
from inv_framework.solvers._utils import prepare_initial_image

METHODS = ('landweber', 'cgls', 'lsqr', 'cg')


def validate_quadratic(algorithm, p):
    if p.get('regularizer','none') not in ('none','tikhonov','h1'):
        raise ValueError('Linear baselines require none, tikhonov or h1; TV/Haar/Huber are not quadratic')
    if p.get('min_value') is not None or p.get('max_value') is not None:
        raise ValueError('Shared linear baselines are unconstrained; CT legacy Landweber alone supports projection')
    if p.get('damping',0) and p.get('strength',0) and p.get('regularizer')!='none':
        raise ValueError('LSQR damping and explicit regularization cannot both be active')
    if algorithm=='cg' and not (p.get('regularizer')=='tikhonov' and p.get('strength',0)>0):
        raise ValueError('Independent CG requires positive identity Tikhonov; H1/unregularized nullspace positivity is not established by these public operators')


class AugmentedQuadratic:
    """Packed B=(sqrt(Q) A, sqrt(lambda) L); no inactive regularizer branch."""
    def __init__(self, operator, y, weights, regularizer, strength, boundary='neumann'):
        self.operator=operator; self.y=y; self.sqrt_q=weights.real.sqrt()
        self.strength=strength if regularizer!='none' else 0.
        self.transform=GradientOperator(operator.domain_shape,boundary=boundary) if self.strength and regularizer=='h1' else None
        self.m=y.numel()
        self.shape=(y.shape[0],*operator.domain_shape)
        self.reg_shape=None
        if self.strength:
            self.reg_shape=(y.shape[0],*self.transform.range_shape) if self.transform else self.shape
    def forward(self,x):
        data=(self.sqrt_q*self.operator.forward(x)).reshape(-1)
        if not self.strength:return data
        z=self.transform.forward(x) if self.transform else x
        return torch.cat((data,math.sqrt(self.strength)*z.reshape(-1)))
    def adjoint(self,z):
        x=self.operator.adjoint(self.sqrt_q*z[:self.m].reshape(self.y.shape))
        if self.strength:
            r=z[self.m:].reshape(self.reg_shape)
            x=x+math.sqrt(self.strength)*(self.transform.adjoint(r) if self.transform else r)
        return x
    def rhs(self):
        data=(self.sqrt_q*self.y).reshape(-1)
        return torch.cat((data,self.y.new_zeros(math.prod(self.reg_shape)))) if self.strength else data
    def bound(self,x):
        return float(self.sqrt_q.square().max())*self.operator.squared_norm_upper_bound(x)+self.strength*(self.transform.squared_norm_upper_bound(x) if self.transform else 1.)


class QuadraticSolver:
    def __init__(self,algorithm,**params):
        self.algorithm=algorithm;self.parameters=params;self.fidelity=None
    def solve(self,y,operator,**kwargs):return self.solve_detailed(y,operator,**kwargs).reconstruction
    @torch.no_grad()
    def solve_detailed(self,measurement,operator,x_init=None,control=None,callback=None,**kwargs):
        p={**self.parameters,**kwargs}
        fidelity=p.pop('fidelity',None) or self.fidelity
        q=fidelity.weights if fidelity is not None else torch.ones_like(measurement.real)
        strength=p.get('strength',0.);reg=p.get('regularizer','none')
        if p.get('damping',0):strength=p['damping']**2;reg='tikhonov'
        b=AugmentedQuadratic(operator,measurement,q,reg,strength,p.get('boundary','neumann'))
        self.quadratic=b
        x=prepare_initial_image(measurement,operator,x_init); rhs=b.rhs(); normal_rhs=b.adjoint(rhs)
        scale=max(1.,norm(normal_rhs)); normal=lambda v:b.adjoint(b.forward(v))
        tolerance=p.get('tolerance')
        if tolerance is None:tolerance=p.get('tol',p.get('atol',0.)+p.get('btol',0.))
        supplied_control=control
        control=resolve_control(control,default_iterations=p.get('num_iterations',200),default_tolerance=tolerance,callback=callback)
        default_stopping = supplied_control is None or (
            control.metadata.get('source')=='ct.api'
            and control.metadata.get('effective_stopping_policy') is None)
        if default_stopping:control=replace(control,patience=1)
        tolerance=control.normalized_normal_residual_tolerance if control.normalized_normal_residual_tolerance is not None else float(control.tolerance or 0.)
        recorder=IterationRecorder(control,measurement,operator,algorithm=self.algorithm);recorder.set_initial(x)
        monitor=ConsecutiveStoppingMonitor(control)
        status='max_iterations';reason='maximum_iterations_reached';actual=0;step=None;bound=None
        if self.algorithm=='landweber':
            bound=b.bound(x);step=p.get('step_size')
            if step is not None and bound>0 and step*bound>=2:raise ValueError('step_size requires alpha * augmented_squared_norm_bound < 2; value was not clipped')
            if step is None:step=1/bound if bound else 1.
        def checkpoint(k,v,previous, *, initial=False):
            nonlocal status,reason,actual
            actual=k;r=b.forward(v)-rhs;g=b.adjoint(r);nr=norm(g)/scale
            objective=.5*norm(r)**2
            criteria=dict(normal_residual=nr<=tolerance)
            relative_change=norm(v-previous)/max(1.,norm(previous))
            if control.relative_iterate_tolerance is not None:criteria['relative_iterate_change']=relative_change<=control.relative_iterate_tolerance
            data=operator.forward(v)-measurement
            if control.discrepancy_target is not None:criteria['discrepancy']=norm(data)/max(norm(measurement),1e-12)<=control.discrepancy_target
            decision=monitor.observe(k,criteria=criteria,relative_change=norm(v-previous)/max(1.,norm(previous)),monitor_value=objective)
            # Default stopping accepts an already solved initial point without
            # an update. Explicit controls retain their iteration requirements.
            converged=decision.converged or (initial and default_stopping and all(criteria.values()))
            if not recorder.record(k,v,residual=data,objective=objective,algorithm_residual=nr,step_size=step,criteria=criteria,stopping_candidate=converged,consecutive_criteria_count=decision.consecutive):
                status='cancelled';reason='callback_cancelled';return False
            from inv_framework.solvers._iteration import _policy_status
            terminal,why=_policy_status(decision)
            if terminal:status=terminal;reason=why;return False
            if not math.isfinite(nr):status='numerical_error';reason='nonfinite_residual';return False
            if converged and control.stop_on_convergence:status='converged';reason='normalized_normal_residual';return False
            return True
        budget=min(p.get('num_iterations',200),control.max_iterations)
        if self.algorithm=='cg':
            previous=x
            def observe(k,v):
                nonlocal previous
                keep=checkpoint(k,v,previous);previous=v.clone();return keep
            solved=solve_cg(normal,normal_rhs,x,budget,atol=tolerance*scale if control.patience==1 and control.min_iterations==1 and control.check_every==1 and control.stop_on_convergence else 0.,rtol=0.,callback=observe)
            x=solved.reconstruction
            # The kernel can certify the initial point before any callback.
            # Record that point once, at its actual update count of zero.
            if solved.converged and solved.iterations==0:
                checkpoint(0,x,previous,initial=True)
            if status=='max_iterations':
                if solved.converged:
                    # Numerical completion cannot satisfy missing policy
                    # observations. Do not manufacture further CG updates.
                    status='stalled';reason='cg_numerical_target_reached_policy_unmet'
                elif solved.stopping_reason=='nonpositive_curvature':
                    status='numerical_error';reason=solved.stopping_reason
                elif solved.stopping_reason=='precision_limit':
                    status='stalled';reason=solved.stopping_reason
                else:
                    reason='maximum_iterations_reached' if solved.stopping_reason=='inner_budget_exhausted' else solved.stopping_reason
            actual=solved.iterations
        else:
            r=rhs-b.forward(x);s=b.adjoint(r);direction=s.clone();ss=float(inner(s,s).real)
            if self.algorithm=='lsqr':
                beta=norm(r);u=r/beta if beta else r;v=b.adjoint(u);alpha=norm(v);v=v/alpha if alpha else v
                w=v.clone();phi=beta;rho_bar=alpha
            for k in range(1,budget+1):
                previous=x
                if norm(normal(x)-normal_rhs)/scale<=tolerance:
                    checkpoint(k,x,previous)
                    if status!='max_iterations':break
                    continue
                if self.algorithm=='landweber':x=x+step*b.adjoint(rhs-b.forward(x))
                elif self.algorithm=='cgls':
                    bd=b.forward(direction);den=norm(bd)**2
                    if den==0:reason='precision_limit';break
                    a=ss/den;x=x+a*direction;r=r-a*bd;s=b.adjoint(r);new_ss=float(inner(s,s).real)
                    direction=s+(new_ss/ss)*direction;ss=new_ss
                else:
                    u=b.forward(v)-alpha*u;beta=norm(u);u=u/beta if beta else u
                    v=b.adjoint(u)-beta*v;alpha=norm(v);v=v/alpha if alpha else v
                    rho=math.hypot(rho_bar,beta)
                    if rho==0:reason='precision_limit';break
                    c=rho_bar/rho;sn=beta/rho;theta=sn*alpha;rho_bar=-c*alpha
                    x=x+(c*phi/rho)*w;phi=sn*phi;w=v-(theta/rho)*w
                if not checkpoint(k,x,previous):break
                if norm(x-previous)==0:reason='precision_limit';break
        residual=operator.forward(x)-measurement;r=b.forward(x)-rhs;nr=norm(b.adjoint(r))/scale
        return recorder.finish(x,actual_iterations=actual,status=status,stopping_reason=reason,final_objective=.5*norm(r)**2,
            final_residual=norm(residual)/max(1.,norm(measurement)),predicted_measurement=operator.forward(x),
            metadata=dict(native_termination_managed=True,endpoint_diagnostics_complete=True,parameters=p,
                **({'cg_numerical':solved.diagnostics()} if self.algorithm=='cg' else {}),
                normalized_normal_residual=nr,normal_residual_tolerance=tolerance,normal_residual_definition='||A*Q(Ax-y)+lambda L*Lx|| / max(1,||A*Qy||)',
                data_residual=norm(residual),weighted_data_residual=norm(b.sqrt_q*residual),
                objective_definition='0.5 sum Q |Ax-y|^2 + 0.5 lambda ||Lx||^2',
                weights_definition='nonnegative diagonal Q; no correlated covariance',regularizer_definition=dict(name=reg,strength=strength,boundary=p.get('boundary','neumann'),spacing=[1.,1.],spatial_axes=[-2,-1],complex='modulus squared, Hermitian adjoint'),
                step_size=step,augmented_squared_norm_bound=bound,step_basis='upper_bound' if bound is not None else None))

"""Shared public composition assembly and fixed-block two-block ADMM."""
import math
import torch
from inv_framework.core.array_math import norm
from inv_framework.core.krylov import solve_cg
from inv_framework.core.operator import GradientOperator
from inv_framework.core.regularizer import (TVRegularizer, HaarL1Regularizer,
    TikhonovRegularizer, GradientHuberRegularizer, ZeroRegularizer)
from inv_framework.solvers.base import IterationRecorder, resolve_control, ConsecutiveStoppingMonitor
from inv_framework.solvers.splitting import Penalty, solve_fista_detailed, solve_split_detailed
from inv_framework.solvers._utils import prepare_initial_image, apply_box_constraints

REGULARIZERS = ('none', 'tv', 'haar_l1', 'tikhonov', 'h1', 'huber', 'gradient_huber')
from inv_framework.experiments import COMPOSABLE_METHODS as METHODS


def make_regularizer(name, shape, params):
    boundary = params.get('boundary', 'neumann')
    if name == 'tv':
        return TVRegularizer(params.get('tv_mode', 'isotropic'),
            params.get('tv_num_iterations', 2000), params.get('tv_tolerance', 1e-5), boundary=boundary)
    if name == 'haar_l1':
        reg = HaarL1Regularizer(params.get('level', 2), params.get('penalize_lowpass', False))
        reg.num_iterations = params.get('tv_num_iterations', 50)
        reg.tolerance = params.get('tv_tolerance', 1e-5)
        return reg
    if name == 'h1':
        return TikhonovRegularizer(GradientOperator(shape, boundary=boundary))
    if name in ('huber', 'gradient_huber'):
        reg = GradientHuberRegularizer(params.get('delta', .01), grouped=name == 'gradient_huber')
        # Huber's established discrete operator is Neumann, unit spacing.
        if boundary != 'neumann':
            raise ValueError('Huber currently requires Neumann differences')
        return reg
    return TikhonovRegularizer() if name == 'tikhonov' else ZeroRegularizer()


class CompositeSolver:
    """One numerical implementation per method, shared by physical domains."""
    def __init__(self, algorithm, **params):
        self.algorithm, self.parameters = algorithm, params
        self.fidelity = None

    @torch.no_grad()
    def solve_detailed(self, measurement, operator, x_init=None, control=None, callback=None, **kwargs):
        from inv_framework.core.fidelity import QuadraticFidelity
        self.endpoint_state={}
        p = {**self.parameters, **kwargs}
        selected_fidelity=p.pop('fidelity',None)
        name = p.get('regularizer', 'tikhonov')
        joint = name == 'tv_haar'
        reg = make_regularizer('tv' if joint else name, operator.domain_shape, p)
        strength = p.get('strength')
        if strength is None:strength=p.get('reg_strength')
        if strength is None:strength=.01
        p['strength']=strength
        common = dict(x_init=x_init, control=control, callback=callback,
            num_iterations=p.get('num_iterations', 200), tolerance=p.get('tolerance', 1e-5),
            min_value=p.get('min_value'), max_value=p.get('max_value'))
        fidelity = selected_fidelity or self.fidelity or QuadraticFidelity(measurement)
        if self.algorithm == 'fista':
            result = solve_fista_detailed(operator, measurement, regularizer=reg, strength=strength,
                step_size=p.get('step_size'), power_iterations=p.get('power_iterations',20), fidelity=fidelity, **common)
        elif self.algorithm == 'pdhg':
            result = solve_split_detailed(operator, measurement, algorithm='pdhg', regularizer=reg,
                strength=strength, tau=p.get('tau'), sigma=p.get('sigma'), fidelity=fidelity, endpoint_state=self.endpoint_state, **common)
        else:
            terms = [(('tv' if joint else name), reg, strength, p.get('rho_regularizer') or p.get('rho', 1.))]
            if joint:
                terms.append(('haar_l1', make_regularizer('haar_l1', operator.domain_shape, p),
                              p.get('haar_strength', .01), p.get('rho_haar') or p.get('rho', 1.)))
            result = fixed_block_admm(operator, measurement, fidelity, terms, p, endpoint_state=self.endpoint_state, **common)
        prediction=operator.forward(result.reconstruction)
        result.metadata['data_value']=float(fidelity.value(prediction))
        if self.algorithm != 'admm':
            result.metadata['regularizer_contributions']={name:strength*float(reg.value(result.reconstruction).sum())}
        result.metadata['constraints']=dict(min_value=p.get('min_value'),max_value=p.get('max_value'))
        result.metadata['regularizer_definition'] = dict(name=name, strength=strength,
            haar_strength=p.get('haar_strength') if joint else None,
            spatial_axes=[-2,-1], spacing=[1.,1.], boundary=p.get('boundary', 'neumann'),
            channels='independent', tv_mode=p.get('tv_mode','isotropic'),
            haar_level=p.get('level',2), penalize_lowpass=p.get('penalize_lowpass',False),
            huber_delta=p.get('delta',.01), huber_scaling='r^2/(2 delta) else r-delta/2')
        return result

    def solve(self, measurement, operator, **kwargs):
        return self.solve_detailed(measurement, operator, **kwargs).reconstruction


@torch.no_grad()
def fixed_block_admm(operator, measurement, fidelity, terms, params, *, x_init,
                     control, callback, num_iterations, tolerance, min_value, max_value, endpoint_state):
    """Split Ax, all K_j x, and identity into one joint auxiliary block.

    The identity split is an exact indicator (identity prox if unconstrained),
    making the CG system positive definite without changing the objective.
    """
    rho = params.get('rho', 1.)
    rd = params.get('rho_data') or rho
    rc = params.get('rho_constraint') or rho
    branches = [(name, Penalty(reg, operator.domain_shape), weight, r)
                for name, reg, weight, r in terms if weight > 0 and name != 'none']
    control = resolve_control(control, default_iterations=num_iterations,
                              default_tolerance=tolerance, callback=callback)
    project = lambda x: apply_box_constraints(x, min_value, max_value)
    x = project(prepare_initial_image(measurement, operator, x_init))
    q = x.clone(); v = torch.zeros_like(x)
    observation = operator.forward(x); od = torch.zeros_like(observation)
    zs = [p.apply(x) for _,p,_,_ in branches]; us = [torch.zeros_like(z) for z in zs]
    recorder = IterationRecorder(control, measurement, operator, algorithm='admm')
    recorder.set_initial(x)
    monitor=ConsecutiveStoppingMonitor(control)
    tolerance=float(control.tolerance or 0.)
    def normal(a):
        out = rd*operator.adjoint(operator.forward(a))+rc*a
        for _, p, _, r in branches:
            out = out+r*p.apply(p.apply(a),True)
        return out
    denominator = None
    requested = params.get('subproblem_solver', 'auto')
    diagonal = hasattr(operator, 'otf') and all(
        p.transform is None or getattr(p.transform, 'transform_kind', None) == 'W' or
        (getattr(p.transform, 'transform_kind', None) == 'D' and p.transform.differences.boundary == 'periodic')
        for _,p,_,_ in branches)
    if requested == 'fft' and not diagonal:
        raise ValueError('FFT requires periodic convolution and simultaneously diagonalizable splits')
    if diagonal and requested in ('auto', 'fft'):
        denominator = rd*operator.otf.abs().square()+rc
        h,w=operator.domain_shape[-2:]
        fy=torch.arange(h,device=x.device,dtype=x.real.dtype)
        fx=torch.arange(w,device=x.device,dtype=x.real.dtype)
        lap=4*torch.sin(math.pi*fy/h).square()[:,None]+4*torch.sin(math.pi*fx/w).square()[None,:]
        for _,p,_,r in branches:
            denominator=denominator+r*(lap if getattr(p.transform,'transform_kind',None)=='D' else 1.)
    factor = None
    if requested == 'cholesky':
        # Explicit fixed normal system for small repeated solves. The same
        # forward/adjoint operations assemble it and certify every solution.
        size = math.prod(operator.domain_shape)
        columns = []
        for start in range(0, size, 64):
            count = min(64, size-start)
            basis = x.new_zeros((count, size))
            basis[torch.arange(count, device=x.device), torch.arange(start, start+count, device=x.device)] = 1
            columns.append(normal(basis.reshape(count, *operator.domain_shape)).reshape(count, size).T)
        factor = torch.linalg.cholesky(torch.cat(columns, dim=1))
        del columns
    reports=[]; scale=None; primal_blocks={}; primal=dual=None; actual=0
    reason='maximum_iterations_reached'; status='max_iterations'
    for k in range(1,min(num_iterations,control.max_iterations)+1):
        rhs=rd*operator.adjoint(observation-od)+rc*(q-v)
        for (_,p,_,r),z,u in zip(branches,zs,us):
            rhs=rhs+r*p.apply(z-u,True)
        if scale is None: scale=max(1.,norm(rhs))
        target=params.get('cg_tolerance',1e-6)*scale/k**2
        if denominator is None and factor is None:
            cg=solve_cg(normal,rhs,x,params.get('cg_max_iterations',100),atol=target,rtol=0.)
            reports.append(cg.diagnostics())
            if not cg.converged:
                reason=cg.stopping_reason
                break
            x=cg.reconstruction
        else:
            candidate = (torch.cholesky_solve(rhs.flatten(1).T, factor).T.reshape_as(x)
                if factor is not None else
                torch.fft.ifftn(torch.fft.fftn(rhs,dim=(-2,-1))/denominator,dim=(-2,-1)).real)
            true_residual=norm(normal(candidate)-rhs)
            reports.append(dict(residual=true_residual,target=target,iterations=0,converged=true_residual<=target))
            if true_residual>target:
                reason='inner_returned_precision';break
            x=candidate
        ax=operator.forward(x)
        try:
            no=fidelity.proximal(ax+od,1/rd)
        except RuntimeError as error:
            if str(error) not in ('poisson_prox_inner_max_iterations','poisson_prox_returned_precision'):raise
            reason=str(error);break
        od=od+ax-no
        ds=rd*operator.adjoint(no-observation); observation=no
        primal_blocks={'data':norm(ax-no)}
        for j,(name,p,weight,r) in enumerate(branches):
            tx=p.apply(x); nz=p.prox(tx+us[j],weight/r)
            us[j]=us[j]+tx-nz
            primal_blocks[name]=norm(tx-nz)
            ds=ds+r*p.apply(nz-zs[j],True); zs[j]=nz
        previous=q
        nq=project(x+v); v=v+x-nq
        primal_blocks['constraint']=norm(x-nq); ds=ds+rc*(nq-q); q=nq
        primal=math.sqrt(sum(r*r for r in primal_blocks.values())); dual=norm(ds)
        actual=k
        objective=float(fidelity.value(operator.forward(q)))+sum(w*float(p.reg.value(q).sum()) for _,p,w,_ in branches)
        threshold=tolerance*max(1.,norm(q),norm(measurement))
        residual = operator.forward(q)-measurement
        criteria = dict(primal=primal<=threshold, dual=dual<=threshold)
        if control.discrepancy_target is not None:
            criteria['discrepancy'] = (
                norm(residual)/max(norm(measurement), 1e-12) <= control.discrepancy_target
            )
        decision=monitor.observe(k,criteria=criteria,
            relative_change=norm(q-previous)/max(1.,norm(previous)),monitor_value=objective)
        converged=decision.converged
        if not recorder.record(k,q,residual=residual,objective=objective,criteria=criteria,
            algorithm_residual=max(primal,dual),stopping_candidate=converged,
            consecutive_criteria_count=decision.consecutive,
            metadata=dict(primal_blocks=dict(primal_blocks),dual_residual=dual)):
            status,reason='cancelled','callback_cancelled';break
        from inv_framework.solvers._iteration import _policy_status
        terminal,terminal_reason=_policy_status(decision)
        if terminal:
            status,reason=terminal,terminal_reason;break
        if converged and control.stop_on_convergence:
            status,reason='converged','split_residual';break
    endpoint_state.update(p=rd*od,branches=[(p,w,r*u) for (_,p,w,r),u in zip(branches,us)],
                          fidelity=fidelity,min_value=min_value,max_value=max_value)
    data=float(fidelity.value(operator.forward(q)))
    contributions={name:w*float(p.reg.value(q).sum()) for name,p,w,_ in branches}
    return recorder.finish(q,actual_iterations=actual,status=status,stopping_reason=reason,
        final_objective=data+sum(contributions.values()),
        final_residual=norm(operator.forward(q)-measurement)/max(1.,norm(measurement)),
        predicted_measurement=operator.forward(q),metadata=dict(data_value=data,
        regularizer_contributions=contributions,primal_blocks=primal_blocks,
        native_termination_managed=True,endpoint_diagnostics_complete=True,parameters=params,
        primal_residual=primal,dual_residual=dual,cg_subproblems=reports,
        cg_schedule='eta0*max(1,||b_1||)/k^2',subproblem_path=('dense_cholesky' if factor is not None else 'hermitian_cg' if denominator is None else 'periodic_fft'),
        normal_factorizations=1 if factor is not None else 0,
        normal_factor_dimension=math.prod(operator.domain_shape) if factor is not None else None,
        block_rho=dict(data=rd,constraint=rc,**{n:r for n,_,_,r in branches}),rho_fixed=True))

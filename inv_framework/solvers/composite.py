"""Internal two-block ADMM for small quadratic composite validation problems.

Not registered as a three-domain task API. The x block and joint auxiliary
block include each regularizer transform and the box indicator with fixed,
positive branch penalties. No ridge term is added to the objective.
"""
import math
import torch
from inv_framework.core.krylov import solve_cg
from inv_framework.core.array_math import norm
from inv_framework.solvers.splitting import Penalty
from inv_framework.solvers.base import SolveResult


@torch.no_grad()
def quadratic_composite(operator, measurement, terms, *, x_init, max_iterations=2000,
                        tolerance=1e-7, cg_budget=100, eta0=1e-8,
                        min_value=None, max_value=None, constraint_rho=1.):
    """Minimize .5||Ax-y||² + sum(weight*R(x)) + indicator_box(x).

    terms contains (regularizer, weight, rho). Zero-weight terms are discarded.
    Fixed-scale absolute CG targets are eta0*max(1,||b_1||)/k².
    The identity constraint split makes H positive definite without changing
    the target even for an unconstrained or rank-deficient data operator.
    """
    if constraint_rho <= 0 or any(w < 0 or r <= 0 for _, w, r in terms):
        raise ValueError('weights must be nonnegative and branch penalties positive')
    branches = [(Penalty(reg, operator.domain_shape), w, r) for reg, w, r in terms if w > 0]
    project = lambda x: x.clamp(min_value, max_value) if min_value is not None or max_value is not None else x
    x = project(x_init.detach().clone())
    zs = [p.apply(x) for p, _, _ in branches]; us = [torch.zeros_like(z) for z in zs]
    q = x.clone(); v = torch.zeros_like(x)
    aty = operator.adjoint(measurement)
    def normal(a):
        out = operator.adjoint(operator.forward(a))+constraint_rho*a
        for p, _, rho in branches:
            out = out+rho*p.apply(p.apply(a), True)
        return out
    reports = []; reference_scale = None; reason = 'maximum_iterations_reached'; status = 'max_iterations'
    for iteration in range(1, max_iterations+1):
        rhs = aty+constraint_rho*(q-v)
        for (p, _, rho), z, u in zip(branches, zs, us):
            rhs = rhs+rho*p.apply(z-u, True)
        if reference_scale is None:
            reference_scale = max(1., norm(rhs))
        cg = solve_cg(normal, rhs, x, cg_budget, atol=eta0*reference_scale/iteration**2, rtol=0.)
        reports.append(cg.diagnostics())
        if not cg.converged:
            reason = cg.stopping_reason; break
        x = cg.reconstruction
        primal_sq = 0.; dual_vector = torch.zeros_like(x)
        for j, (p, weight, rho) in enumerate(branches):
            tx = p.apply(x); z = p.prox(tx+us[j], weight/rho)
            us[j] = us[j]+tx-z
            primal_sq += norm(tx-z)**2
            dual_vector += rho*p.apply(z-zs[j], True)
            zs[j] = z
        nq = project(x+v); v += x-nq
        primal_sq += norm(x-nq)**2
        dual_vector += constraint_rho*(nq-q); q = nq
        primal = math.sqrt(primal_sq); dual = norm(dual_vector)
        if max(primal, dual) <= tolerance*max(1., norm(x)):
            status, reason = 'converged', 'split_residual'; break
    data = .5*norm(operator.forward(q)-measurement)**2
    contributions = [w*float(p.value(p.apply(q))) for p, w, _ in branches]
    metadata = dict(data_value=data, regularizer_contributions=contributions,
                    cg_subproblems=reports, cg_schedule='eta0*max(1,||b_1||)/k^2',
                    branch_rho=[r for _, _, r in branches], constraint_rho=constraint_rho,
                    active_branches=len(branches), primal_residual=locals().get('primal'),
                    dual_residual=locals().get('dual'))
    return SolveResult(reconstruction=q, status=status, stopping_reason=reason,
                       actual_iterations=iteration,
                       final_objective=data+sum(contributions), metadata=metadata)

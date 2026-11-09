"""Complex SENSE quadratic and analysis regularization solvers."""
import numpy as np
from inv_framework.core.numpy_math import conjugate_gradient, gradient, gradient_adjoint, shrink


from inv_framework.core.array_math import OrthogonalHaar


class Haar(OrthogonalHaar):
    """MRI Haar keeps its existing all-coefficient penalty in cs_reconstruct."""
    def __init__(self, shape, levels):
        super().__init__(shape, tuple(range(1, len(shape))), levels)


def cs_reconstruct(op, y, weights, initial, params, wavelet):
    axes = tuple(range(1,initial.ndim))
    transform = Haar(initial.shape,params.get('levels',1)) if wavelet else None
    forward = transform.forward if wavelet else lambda x: gradient(x,axes)
    adjoint = transform.adjoint if wavelet else lambda z: gradient_adjoint(z,axes)
    x = initial.copy(); z = forward(x); u = np.zeros_like(z)
    rho, lam = params['rho'],params['lam']
    rhs = op.adjoint(weights*y)
    normal = lambda v: op.adjoint(weights*op.forward(v))+rho*adjoint(forward(v))
    history=[]; reason='max_iterations'; inner_reports=[]; reference_scale=None
    for iteration in range(1,params['max_iter']+1):
        from inv_framework.core.krylov import solve_cg
        subproblem_rhs = rhs+rho*adjoint(z-u)
        if reference_scale is None:
            reference_scale = max(1., float(np.linalg.norm(subproblem_rhs)))
        target = params['cg_tolerance']*reference_scale/iteration**2
        inner = solve_cg(normal, subproblem_rhs, x, params['cg_max_iter'], atol=target, rtol=0.)
        inner_reports.append(inner.diagnostics())
        inner_reason = inner.stopping_reason
        if not inner.converged:
            reason = 'numerical_error' if inner_reason == 'nonpositive_curvature' else inner_reason
            break
        x = inner.reconstruction
        gx = forward(x); previous = z
        z = shrink(gx+u,lam/rho,grouped=not wavelet); u += gx-z
        primal = float(np.linalg.norm(gx-z)); dual = float(rho*np.linalg.norm(adjoint(z-previous)))
        history.append({'primal_residual':primal,'dual_residual':dual})
        inner_residual = inner.residual
        history[-1]['inner_normal_residual'] = float(inner_residual)
        if max(primal,dual,inner_residual) <= params['tolerance']*max(1,np.linalg.norm(x)):
            reason='converged'; break
    gx = forward(x)
    regularizer = float(np.sum(abs(gx))) if wavelet else float(np.sum(np.sqrt(np.sum(abs(gx)**2,axis=0))))
    residual = op.forward(x)-y
    return x,iteration,reason,dict(residual_history=history,data_fit=float(.5*np.sum(weights*abs(residual)**2)),
        regularizer_value=regularizer,regularizer_contribution=lam*regularizer,solver='admm',inner_stopping_reason=inner_reason, inner_subproblems=inner_reports, inner_schedule='eta0*max(1,||b_1||)/k^2')

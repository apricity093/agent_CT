"""Complex SENSE quadratic and analysis regularization solvers."""
import numpy as np
from inv_framework.core.numpy_math import conjugate_gradient, gradient, gradient_adjoint, shrink


class Haar:
    """Orthogonal separable Haar, periodized dyadic blocks; all coefficients penalized."""
    def __init__(self, shape, levels):
        self.shape, self.levels = shape, levels
        self.axes = tuple(range(1, len(shape)))

    def forward(self, x):
        out = x.copy()
        for level in range(self.levels):
            region = (slice(None),)+tuple(slice(0,n//2**level) for n in self.shape[1:])
            block = out[region].copy()
            for a in self.axes:
                even = np.take(block, range(0,block.shape[a],2),axis=a)
                odd = np.take(block, range(1,block.shape[a],2),axis=a)
                block = np.concatenate([(even+odd)/np.sqrt(2),(even-odd)/np.sqrt(2)],axis=a)
            out[region] = block
        return out

    def adjoint(self, z):
        out = z.copy()
        for level in reversed(range(self.levels)):
            region = (slice(None),)+tuple(slice(0,n//2**level) for n in self.shape[1:])
            block = out[region].copy()
            for a in reversed(self.axes):
                lo, hi = np.split(block,2,axis=a)
                merged = np.empty_like(block)
                even, odd = [slice(None)]*block.ndim, [slice(None)]*block.ndim
                even[a], odd[a] = slice(0,None,2),slice(1,None,2)
                merged[tuple(even)],merged[tuple(odd)] = (lo+hi)/np.sqrt(2),(lo-hi)/np.sqrt(2)
                block = merged
            out[region] = block
        return out


def cs_reconstruct(op, y, weights, initial, params, wavelet):
    axes = tuple(range(1,initial.ndim))
    transform = Haar(initial.shape,params.get('levels',1)) if wavelet else None
    forward = transform.forward if wavelet else lambda x: gradient(x,axes)
    adjoint = transform.adjoint if wavelet else lambda z: gradient_adjoint(z,axes)
    x = initial.copy(); z = forward(x); u = np.zeros_like(z)
    rho, lam = params['rho'],params['lam']
    rhs = op.adjoint(weights*y)
    normal = lambda v: op.adjoint(weights*op.forward(v))+rho*adjoint(forward(v))
    history=[]; reason='max_iterations'
    for iteration in range(1,params['max_iter']+1):
        # Summably decreasing inner tolerance for inexact ADMM.
        subproblem_rhs = rhs+rho*adjoint(z-u)
        x, _, inner_reason, inner_history = conjugate_gradient(normal,subproblem_rhs,x,
            params['cg_max_iter'],params['cg_tolerance']/iteration**2)
        if inner_reason == 'numerical_error':
            return x,iteration,'numerical_error',{'residual_history':history}
        gx = forward(x); previous = z
        z = shrink(gx+u,lam/rho,grouped=not wavelet); u += gx-z
        primal = float(np.linalg.norm(gx-z)); dual = float(rho*np.linalg.norm(adjoint(z-previous)))
        history.append({'primal_residual':primal,'dual_residual':dual})
        inner_residual = inner_history[-1]*np.linalg.norm(subproblem_rhs)
        history[-1]['inner_normal_residual'] = float(inner_residual)
        if max(primal,dual,inner_residual) <= params['tolerance']*max(1,np.linalg.norm(x)):
            reason='converged'; break
    gx = forward(x)
    regularizer = float(np.sum(abs(gx))) if wavelet else float(np.sum(np.sqrt(np.sum(abs(gx)**2,axis=0))))
    residual = op.forward(x)-y
    return x,iteration,reason,dict(residual_history=history,data_fit=float(.5*np.sum(weights*abs(residual)**2)),
        regularizer_value=regularizer,regularizer_contribution=lam*regularizer,solver='admm',inner_stopping_reason=inner_reason)

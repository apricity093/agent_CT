"""Complex CG and periodic unit-spacing differences for array solvers."""
import numpy as np


from inv_framework.core.array_math import Differences, shrink as _shrink
from inv_framework.core.krylov import solve_cg


def conjugate_gradient(normal, rhs, initial, max_iter, tolerance):
    result = solve_cg(normal, rhs, initial, max_iter, rtol=tolerance)
    scale = max(float(np.linalg.norm(rhs)), np.finfo(float).tiny)
    return result.reconstruction, result.iterations, ('converged' if result.converged else
        'numerical_error' if result.stopping_reason in ('nonpositive_curvature', 'precision_limit') else 'max_iterations'), [r/scale for r in result.residual_history]


def gradient(x, axes):
    return Differences(axes, boundary='periodic').forward(x)


def gradient_adjoint(g, axes):
    return Differences(axes, boundary='periodic').adjoint(g)


def shrink(z, threshold, grouped=False):
    return _shrink(z, threshold, axis=0 if grouped else None)


def laplacian_spectrum(shape, axes):
    out = np.zeros(shape)
    for a in axes:
        view = [1]*len(shape)
        view[a] = shape[a]
        out += (2-2*np.cos(2*np.pi*np.arange(shape[a])/shape[a])).reshape(view)
    return out

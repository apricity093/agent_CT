"""Complex CG and periodic unit-spacing differences for array solvers."""
import numpy as np


def conjugate_gradient(normal, rhs, initial, max_iter, tolerance):
    x = initial.copy()
    r = rhs - normal(x)
    p = r.copy()
    rr = np.vdot(r, r).real
    scale = max(float(np.linalg.norm(rhs)), np.finfo(float).tiny)
    history = [float(np.sqrt(rr) / scale)]
    iteration = 0
    reason = "converged" if history[-1] <= tolerance else "max_iterations"
    for iteration in range(1, max_iter + 1) if reason != "converged" else ():
        ap = normal(p)
        curvature = np.vdot(p, ap).real
        if curvature <= 0 or not np.isfinite(curvature):
            return x, iteration - 1, "numerical_error", history
        alpha = rr / curvature
        x += alpha * p
        r -= alpha * ap
        next_rr = np.vdot(r, r).real
        history.append(float(np.sqrt(next_rr) / scale))
        if history[-1] <= tolerance:
            reason = "converged"
            break
        p = r + (next_rr / rr) * p
        rr = next_rr
    return x, iteration, reason, history


def gradient(x, axes):
    return np.stack([np.roll(x, -1, a)-x for a in axes])


def gradient_adjoint(g, axes):
    return sum(np.roll(v, 1, a)-v for v, a in zip(g, axes))


def shrink(z, threshold, grouped=False):
    magnitude = np.sqrt(np.sum(abs(z)**2, axis=0, keepdims=True)) if grouped else abs(z)
    return z * np.maximum(1-threshold/np.maximum(magnitude, np.finfo(float).tiny), 0)


def laplacian_spectrum(shape, axes):
    out = np.zeros(shape)
    for a in axes:
        view = [1]*len(shape)
        view[a] = shape[a]
        out += (2-2*np.cos(2*np.pi*np.arange(shape[a])/shape[a])).reshape(view)
    return out

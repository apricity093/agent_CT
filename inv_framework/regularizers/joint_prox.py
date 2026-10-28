"""Certified analysis L1/TV denoising on an optional image box."""
import math
import time
import torch


class ProximalAccuracyError(RuntimeError):
    pass


def joint_prox(reg, value, weight, *, min_value=None, max_value=None,
               tolerance=1e-5, max_iterations=1000, iteration=1, check_budget=None):
    """Dual FGP for .5||x-v||² + weight R(Tx) + indicator_C(x).

    The primal minimizer for a dual vector p is P_C(v-T*p). The certified
    gap bounds distance to the exact prox by sqrt(2 gap). Requested distance
    errors decay as k^-2.1, with a summable iteration-weighted error sequence.
    Inner arithmetic uses float64; the returned dtype error is included.
    """
    from ..solvers._utils import apply_box_constraints
    transform, bound = reg.split_transform(value.shape[1:])
    def apply(point, adjoint=False):
        started=time.perf_counter()
        out=transform.adjoint(point) if adjoint else transform.forward(point)
        name=('synthesis_calls' if adjoint else 'analysis_calls') if type(reg).__name__=='HaarL1Regularizer' else ('gradient_adjoint_calls' if adjoint else 'gradient_calls')
        setattr(reg,name,getattr(reg,name,0)+1)
        if type(reg).__name__=='HaarL1Regularizer':
            reg.transform_seconds+=time.perf_counter()-started
        return out
    project = lambda x: apply_box_constraints(x, min_value=min_value, max_value=max_value)
    target = tolerance * max(float(value.norm()), 1.) / iteration**2.1
    v = value.double()
    p = torch.zeros_like(apply(v))
    momentum = p.clone()
    acceleration = 1.
    reg.last_prox_iterations = 0
    for inner in range(1, max_iterations+1):
        if check_budget is not None: check_budget()
        x = project(v-apply(momentum,True))
        trial = momentum+apply(x)/bound
        next_p = trial-reg.coefficient_proximal(trial, weight)
        x = project(v-apply(next_p,True))
        tx = apply(x)
        gap = max(0., float(weight*reg.coefficient_value(tx).sum()-(next_p*tx).sum()))
        result = x.to(value.dtype)
        error = math.sqrt(2*gap)+float((result.double()-x).norm())
        reg.last_prox_iterations = inner
        reg.last_prox_gap = gap
        reg.last_prox_error_bound = error
        reg.last_prox_target = target
        if error <= target:
            return result
        next_acceleration = .5*(1+math.sqrt(1+4*acceleration**2))
        momentum = next_p+((acceleration-1)/next_acceleration)*(next_p-p)
        p, acceleration = next_p, next_acceleration
    raise ProximalAccuracyError('joint_prox_inner_max_iterations')

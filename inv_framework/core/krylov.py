"""Hermitian positive-definite CG with explicit absolute residual certification."""
from dataclasses import dataclass, field
import math
import torch
from inv_framework.core.array_math import clone, inner, norm


@dataclass
class CGResult:
    reconstruction: object
    budget: int
    target: float
    iterations: int
    residual: float
    converged: bool
    stopping_reason: str
    residual_history: list = field(default_factory=list)


    def diagnostics(self):
        return dict(budget=self.budget, target=self.target, iterations=self.iterations,
                    residual=self.residual, converged=self.converged, stopping_reason=self.stopping_reason)


@torch.no_grad()
def solve_cg(normal, rhs, initial, max_iterations, *, atol=0., rtol=1e-6, callback=None):
    """Target = atol + rtol ||rhs||. Caller establishes Hermitian positivity.

    Every call restarts from the current system's true residual. The final
    residual is recomputed; recursive residual drift never certifies success.
    """
    x = clone(initial)
    target = float(atol + rtol*norm(rhs))
    r = rhs-normal(x); p = clone(r); rr = float(inner(r, r).real)
    history = [math.sqrt(rr)]
    reason = 'inner_budget_exhausted'
    steps = 0
    for k in range(max_iterations+1):
        if math.sqrt(rr) <= target:
            r = rhs-normal(x); rr = float(inner(r, r).real)
            if math.sqrt(rr) <= target:
                reason = 'converged'; break
            p = clone(r)
        if k == max_iterations:
            break
        ap = normal(p); curvature = float(inner(p, ap).real)
        if not math.isfinite(curvature) or curvature <= 0:
            reason = 'nonpositive_curvature'; break
        update = (rr/curvature)*p
        next_x = x+update
        if norm(next_x-x) == 0:
            reason = 'precision_limit'; break
        x = next_x; steps += 1
        r = r-(rr/curvature)*ap
        next_rr = float(inner(r, r).real)
        history.append(math.sqrt(next_rr))
        if callback is not None and callback(steps,x) is False:
            reason = "callback_stopped"; break
        p = r+(next_rr/rr)*p; rr = next_rr
    residual = norm(rhs-normal(x))
    history[-1] = residual
    met = residual <= target
    return CGResult(x, max_iterations, target, steps, residual, met,
                    'converged' if met else reason, history)

"""Unwhitened Euclidean residual balls on explicitly selected measurements."""
import math
import torch
from .fidelity import _check, _timed_prox


def gaussian_whitened_radius(sample_count, alpha=.01):
    """Conservative chi-square upper radius for independent known variances."""
    if not isinstance(sample_count,int) or isinstance(sample_count,bool) or sample_count<=0:
        raise ValueError('sample_count must be a positive integer')
    if not 0<alpha<1: raise ValueError('alpha must lie in (0,1)')
    t=math.log(1/alpha)
    return math.sqrt(sample_count+2*math.sqrt(sample_count*t)+2*t)


class ResidualBall:
    """Indicator of ||mask * (v-y)||_2 <= epsilon, one joint measurement ball.

    Inactive coordinates are unrestricted in the primal and zero in the dual.
    The radius is explicit, in the same line-integral units as the observations.
    """

    def __init__(self, observed, epsilon, *, source, mask=None):
        if not observed.is_floating_point() or not bool(torch.isfinite(observed).all()):
            raise ValueError('observed must be a finite floating tensor')
        if isinstance(epsilon, bool) or not math.isfinite(float(epsilon)) or epsilon < 0:
            raise ValueError('epsilon must be nonnegative finite')
        if not isinstance(source, str) or not source.strip():
            raise ValueError('epsilon source must be explicit')
        self.observed = observed
        self.epsilon, self.source = float(epsilon), source
        self.mask = torch.ones_like(observed, dtype=torch.bool) if mask is None else mask
        if self.mask.shape != observed.shape or self.mask.device != observed.device:
            raise ValueError('mask must match observation shape and device')
        if not bool(((self.mask == 0) | (self.mask == 1)).all()) or not bool(self.mask.any()):
            raise ValueError('mask must be binary with nonempty effective measurement space')
        self.mask = self.mask.bool()
        self.prox_calls = self.inner_iterations = self.inner_failures = self.mask_calls = 0
        self.prox_seconds = 0.
        self.accounting_operator = None

    def residual(self, prediction):
        _check(prediction, self.observed)
        self.mask_calls += 1
        return torch.where(self.mask, prediction-self.observed, 0.)

    def diagnostics(self, prediction):
        residual = float(self.residual(prediction).norm())
        return dict(raw_residual_norm=residual, constraint_violation=max(0., residual-self.epsilon),
                    epsilon=self.epsilon, epsilon_source=self.source,
                    effective_measurements=int(self.mask.sum()))

    def value(self, prediction):
        return prediction.new_tensor(0. if float(self.residual(prediction).norm()) <= self.epsilon else float('inf'))

    def projection(self, value):
        residual = self.residual(value)
        norm = float(residual.norm())
        factor = 1. if norm <= self.epsilon else self.epsilon/norm
        return value-(1.-factor)*residual

    @_timed_prox
    def dual_prox(self, value, step):
        _check(value, self.observed)
        if not math.isfinite(float(step)) or step <= 0:
            raise ValueError('step must be positive finite')
        self.prox_calls += 1
        self.mask_calls += 1
        shifted = torch.where(self.mask, value-step*self.observed, 0.)
        norm = float(shifted.norm())
        factor = max(0., 1.-step*self.epsilon/norm) if norm else 0.
        return shifted*factor

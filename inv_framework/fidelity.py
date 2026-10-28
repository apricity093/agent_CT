"""Separable data terms in projection space (constants independent of x omitted)."""
import math
import time
from functools import wraps
import torch
from .measurement import _parameter, measurement_setting


def _check(value, reference):
    if value.shape != reference.shape or value.device != reference.device or value.dtype != reference.dtype:
        raise ValueError('prediction must match observation shape, dtype and device')


def _timed_prox(method):
    @wraps(method)
    def measured(self, value, step):
        if value.is_cuda: torch.cuda.synchronize(value.device)
        started = time.perf_counter()
        before = self.inner_iterations
        failed = True
        try:
            result = method(self, value, step)
            failed = False
            return result
        finally:
            if value.is_cuda: torch.cuda.synchronize(value.device)
            elapsed = time.perf_counter()-started
            self.prox_seconds += elapsed
            record = getattr(self.accounting_operator, 'record_prox', None)
            if callable(record):
                record(iterations=self.inner_iterations-before,elapsed_seconds=elapsed,failed=failed)
    return measured


class QuadraticFidelity:
    is_quadratic = True

    def validate_normal(self,operator,transform,reference):
        basis=getattr(transform,'nullspace_basis',None)
        if not bool((self.weights==0).any()) or not callable(basis):return
        projections=[operator.forward(vector) for vector in basis(reference)]
        if not projections:return
        columns=torch.stack([p.flatten(1) for p in projections],dim=1)
        weighted=torch.stack([self.normal_gradient(p).flatten(1) for p in projections],dim=1)
        gram=columns@weighted.transpose(1,2)
        if bool((torch.linalg.eigvalsh(gram) <= 0).any()):
            raise ValueError('weighted data and regularizer have a common nullspace; CG normal equation is not positive definite')

    def whiten_operator(self, operator):
        from .operators.weighted import WeightedLinearOperator
        result=WeightedLinearOperator(operator,self.weights)
        result.accounting_fidelity=self
        return result

    def _multiply(self,value):
        self.weight_multiplications+=1
        return self.weights*value

    def normal_gradient(self, prediction):
        _check(prediction,self.observed)
        return self._multiply(prediction)

    def linear_rhs(self):
        return self._multiply(self.observed)

    def __init__(self, observed, weights=1.):
        if not bool(torch.isfinite(observed).all()):
            raise ValueError('nonfinite observations')
        self.weight_multiplications=0
        self.prox_calls=0
        self.inner_iterations=0
        self.inner_failures=0
        self.prox_seconds=0.
        self.accounting_operator=None
        self.observed = observed
        self.weights = _parameter(weights, observed, 'weights')

    def value(self, prediction):
        _check(prediction, self.observed if hasattr(self, "observed") else self.counts)
        return .5*(self._multiply((prediction-self.observed).square())).sum()

    def measurement_diagnostics(self,prediction):
        _check(prediction,self.observed)
        self.weight_multiplications+=1
        return {"whitened_residual":self.weights.sqrt()*(prediction-self.observed)}

    def value_per_batch(self, prediction):
        _check(prediction, self.observed)
        return .5*(self._multiply((prediction-self.observed).square())).flatten(1).sum(1)

    def gradient(self, prediction):
        _check(prediction, self.observed if hasattr(self, "observed") else self.counts)
        return self._multiply(prediction-self.observed)

    @_timed_prox
    def proximal(self, value, step):
        _check(value,self.observed)
        if not math.isfinite(step) or step<=0: raise ValueError('step must be positive finite')
        self.prox_calls+=1
        return (value+step*self.weights*self.observed)/(1+step*self.weights)

    @_timed_prox
    def dual_prox(self, value, step):
        self.prox_calls+=1
        if not math.isfinite(step) or step <= 0:
            raise ValueError('step must be positive finite')
        _check(value, self.observed)
        return self._multiply(value-step*self.observed)/(self.weights+step)


class TransmissionPoissonFidelity:
    """Zero-background independent Poisson NLL, convex for I0>0, scale>0.

    The scalar primal proximal solves z-v+t*s*(c-I0*exp(-s*z))=0.
    Bisection brackets its unique root without clipping counts or exponentials.
    Failure to meet the requested optimality residual raises an explicit error.
    """
    def __init__(self, counts, I0, *, scale=1., tolerance=None, max_iterations=200, mask=None):
        counts = counts if counts.is_floating_point() else counts.to(torch.float64)
        self.counts = _parameter(counts, counts, 'counts')
        tolerance = (1e-10 if counts.dtype == torch.float64 else 8*torch.finfo(counts.dtype).eps) if tolerance is None else tolerance
        if tolerance < torch.finfo(counts.dtype).eps:
            raise ValueError('tolerance is below observation dtype precision')
        if bool((counts != counts.round()).any()):
            raise ValueError('pure Poisson counts must be integers')
        self.I0 = _parameter(I0, counts, 'I0', True)
        self.mask = torch.ones_like(counts) if mask is None else _parameter(mask, counts, 'mask')
        if bool(((self.mask != 0) & (self.mask != 1)).any()): raise ValueError('mask must be binary')
        if not math.isfinite(scale) or scale <= 0 or not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError('scale and tolerance must be positive finite')
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations <= 0:
            raise ValueError('max_iterations must be a positive integer')
        self.scale, self.tolerance, self.max_iterations = scale, tolerance, max_iterations
        self.prox_seconds = 0.
        self.accounting_operator = None
        self.mask_calls = 0
        self.prox_calls = 0
        self.inner_failures = 0
        self.inner_iterations = 0
        self.last_residual = None

    def precision_report(self):
        return {'working_dtype':'torch.float64','returned_dtype':str(self.counts.dtype),
                'requested_inner_tolerance':self.tolerance,'last_working_residual':self.last_residual,
                'last_returned_residual':getattr(self,'last_returned_residual',None),
                'returned_optimality_satisfied':getattr(self,'returned_optimality_satisfied',None),
                'returned_residual_threshold':getattr(self,'returned_residual_threshold',None),
                'inner_iterations_semantics':'batch-vector bisection iterations'}

    def validate_minimizer(self, regularizer, strength, max_value):
        positive_counts=bool((self.counts[self.mask.bool()]>0).all())
        coercive=regularizer=='tikhonov' and strength>0
        if not (positive_counts or coercive or max_value is not None):
            raise ValueError('finite_minimizer_not_established: zero counts require identity penalty or public box')

    def measurement_diagnostics(self,prediction):
        _check(prediction,self.counts)
        return {"predicted_counts":self.I0*torch.exp(-self.scale*prediction)}

    def value(self, prediction):
        _check(prediction, self.observed if hasattr(self, "observed") else self.counts)
        self.mask_calls += 2
        p = torch.where(self.mask.bool(),self.scale*prediction,torch.zeros_like(prediction))
        return (self.mask*(self.I0*torch.exp(-p)+self.counts*p)).sum()

    def value_per_batch(self, prediction):
        _check(prediction, self.counts)
        p = torch.where(self.mask.bool(), self.scale*prediction, torch.zeros_like(prediction))
        return (self.mask*(self.I0*torch.exp(-p)+self.counts*p)).flatten(1).sum(1)

    def gradient(self, prediction):
        _check(prediction, self.observed if hasattr(self, "observed") else self.counts)
        self.mask_calls += 2
        active_prediction=torch.where(self.mask.bool(),prediction,torch.zeros_like(prediction))
        return self.mask*self.scale*(self.counts-self.I0*torch.exp(-self.scale*active_prediction))

    @_timed_prox
    def proximal(self, value, step):
        self.last_residual = None
        self.last_returned_residual = None
        self.prox_calls += 1
        if not math.isfinite(step) or step <= 0:
            raise ValueError('proximal requires positive finite step')
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError('nonfinite proximal input')
        _check(value, self.counts)
        if value.dtype != torch.float64:
            working=TransmissionPoissonFidelity(self.counts.double(),self.I0.double(),scale=self.scale,tolerance=self.tolerance,max_iterations=self.max_iterations,mask=self.mask.double())
            try:
                root=working.proximal(value.double(),step)
            finally:
                self.mask_calls+=working.mask_calls
                self.inner_iterations+=working.inner_iterations
                self.inner_failures+=working.inner_failures
                self.last_residual=working.last_residual
            result=root.to(value.dtype)
            mask_before=working.mask_calls
            residual=result.double()-value.double()+step*working.gradient(result.double())
            self.last_returned_residual=float(residual.abs().max())
            self.mask_calls+=working.mask_calls-mask_before
            active=torch.where(working.mask.bool(),result.double(),0.)
            curvature=self.scale**2*working.I0*torch.exp(-self.scale*active)*working.mask
            threshold=self.tolerance*(1+value.double().abs())+4*torch.finfo(value.dtype).eps*(1+result.double().abs())*(1+step*curvature)
            self.returned_residual_threshold=float(threshold.max())
            self.returned_optimality_satisfied=bool((residual.abs()<=threshold).all())
            if not self.returned_optimality_satisfied:
                self.inner_failures+=1
                raise RuntimeError('poisson_prox_returned_precision')
            return result
        s = self.scale
        # In shifted coordinates u=s*(z-v+t*s*c), u+log(u)=log(t*s²*I0)-s*v+t*s²*c.
        # Solve in z directly using a guaranteed lower bound and an upper bound
        # from exp(-s*z)<=1 when z>=0.
        lower = value-step*s*self.counts
        upper = torch.maximum(torch.zeros_like(value), value+step*s*self.I0)
        self.mask_calls+=2
        lower=torch.where(self.mask.bool(),lower,value)
        upper=torch.where(self.mask.bool(),upper,value)
        for iteration in range(self.max_iterations):
            z = (lower+upper)/2
            residual = z-value+step*self.gradient(z)
            self.inner_iterations += 1
            if bool((residual.abs() <= self.tolerance*(1+value.abs())).all()):
                self.last_residual = float(residual.abs().max())
                self.last_returned_residual=self.last_residual
                self.returned_optimality_satisfied=True
                self.returned_residual_threshold=float((self.tolerance*(1+value.abs())).max())
                return z
            lower = torch.where(residual < 0, z, lower)
            upper = torch.where(residual >= 0, z, upper)
        self.last_residual = float(residual.abs().max())
        self.inner_failures += 1
        raise RuntimeError('poisson_prox_inner_max_iterations')

    def dual_prox(self, value, step):
        if not math.isfinite(step) or step <= 0:
            raise ValueError('step must be positive finite')
        result=value-step*self.proximal(value/step, 1/step)
        self.mask_calls+=1
        return torch.where(self.mask.bool(),result,torch.zeros_like(result))


def build_fidelity(model, observed, public, metadata, *, mask=None, weight_source=None,
                   tolerance=None, max_iterations=200):
    public = public or {}
    if metadata.get('noise_model') == 'poisson_gaussian' and 'counts' in public:
        raise ValueError('mixed observations cannot expose latent counts')
    if model == 'transmission_poisson_nll':
        if metadata.get('noise_model') != 'poisson' or 'c_obs' in public:
            raise ValueError('transmission_poisson_nll requires pure Poisson observations')
        if 'counts' not in public or 'I0' not in public:
            raise ValueError('transmission_poisson_nll requires raw counts and I0')
        if ('sigma_e' in public and bool((_parameter(public['sigma_e'],observed,'sigma_e')!=0).any())) or measurement_setting(metadata,'sigma_e',0)!=0:
            raise ValueError('pure Poisson NLL requires sigma_e=0')
        if measurement_setting(metadata,'background',0) != 0:
            raise ValueError('transmission NLL requires zero background')
        if mask is not None and mask.shape == observed.shape[1:]: mask=mask.unsqueeze(0).expand_as(observed)
        return TransmissionPoissonFidelity(public['counts'].to(observed), public['I0'],
            scale=measurement_setting(metadata,'scale',1.), tolerance=tolerance,max_iterations=max_iterations,mask=mask)
    if model == 'l2':
        weights=1.
    elif model == 'gaussian_nll':
        if 'sigma' not in public:
            raise ValueError('gaussian_nll requires known sigma')
        weights=_parameter(public['sigma'], observed, 'sigma', True).reciprocal().square()
    elif model == 'wls':
        if weight_source not in ('known_variance','provided','post_log_poisson','post_log_poisson_gaussian'):
            raise ValueError('wls requires explicit weight_source')
        if 'weights' not in public:
            raise ValueError('wls requires public weights')
        expected = {'post_log_poisson':'poisson','post_log_poisson_gaussian':'poisson_gaussian'}
        if weight_source in expected and metadata.get('noise_model') != expected[weight_source]:
            raise ValueError('weight_source does not match noise_model')
        weights=public['weights']
    else:
        raise ValueError('unknown data_fidelity: '+str(model))
    weights=_parameter(weights,observed,'weights')
    if mask is not None:
        if mask.shape == observed.shape[1:]: mask=mask.unsqueeze(0).expand_as(observed)
        if mask.shape != observed.shape: raise ValueError('mask must match observations')
        weights=weights*mask.to(observed)
    result=QuadraticFidelity(observed,weights)
    result.mask_compilation_calls=int(mask is not None)
    return result

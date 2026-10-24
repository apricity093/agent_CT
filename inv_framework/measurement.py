"""Public CT observations. Transmission log flooring never changes raw readout."""
import math
import torch


PUBLIC_FIELDS = ('counts', 'c_obs', 'I0', 'sigma', 'sigma_e', 'weights')


def measurement_setting(metadata,name,default=None):
    parameters=metadata.get('parameters',{}) or {}
    if name in metadata and name in parameters and metadata[name]!=parameters[name]:
        raise ValueError('conflicting measurement '+name)
    return parameters.get(name,metadata.get(name,default))


def _parameter(value, reference, name, positive=False):
    result = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if result.numel() != 1 and result.shape != reference.shape:
        raise ValueError(name + ' must be scalar or match observations')
    if not bool(torch.isfinite(result).all()) or bool((result <= 0 if positive else result < 0).any()):
        raise ValueError(name + ' outside valid range')
    return result.expand_as(reference).clone()


def simulate_observation(projection, *, model, seed, sigma=None, I0=None,
                         sigma_e=0., floor=1., scale=1.):
    """Return (log/line observations, public arrays, JSON metadata).

    Explicit scale models exp(-scale*p); new physical cases default to scale=1.
    Post-log weights are plug-in delta-method approximations using readout only.
    """
    if not projection.is_floating_point() or not bool(torch.isfinite(projection).all()):
        raise ValueError('projection must be finite floating point')
    generator = torch.Generator(device=projection.device).manual_seed(seed)
    metadata = dict(noise_model=model, seed=seed, observation_domain='line_integral' if model=='gaussian' else 'log_projection', observation_model='gaussian' if model == 'gaussian' else 'transmission_'+model)
    if model == 'gaussian':
        std = _parameter(sigma, projection, 'sigma', True)
        observed = projection + std*torch.randn(projection.shape, generator=generator,
                                                device=projection.device, dtype=projection.dtype)
        if torch.as_tensor(sigma).numel()==1: metadata['parameters']={'sigma':float(torch.as_tensor(sigma))}
        metadata.update(weight_source='known_variance',approximation=False,weight_cap=None)
        return observed, dict(sigma=std, weights=std.reciprocal().square()), metadata
    if model not in ('poisson', 'poisson_gaussian'):
        raise ValueError('unknown noise model')
    if not math.isfinite(floor) or floor <= 0 or not math.isfinite(scale) or scale <= 0:
        raise ValueError('floor and scale must be positive finite')
    incident = _parameter(I0, projection, 'I0', True)
    electronic = _parameter(sigma_e, projection, 'sigma_e')
    if model == 'poisson' and bool((electronic != 0).any()):
        raise ValueError('pure Poisson requires sigma_e=0')
    expectation = incident*torch.exp(-scale*projection)
    if not bool(torch.isfinite(expectation).all()):
        raise ValueError('nonfinite transmission expectation')
    readout = torch.poisson(expectation, generator=generator)
    arrays = dict(I0=incident)
    if model == 'poisson_gaussian':
        readout = readout + electronic*torch.randn(projection.shape, generator=generator,
                                                   device=projection.device, dtype=projection.dtype)
        arrays.update(c_obs=readout, sigma_e=electronic)
    else:
        arrays['counts'] = readout
    estimate = readout.clamp_min(floor)
    arrays['weights'] = scale**2*estimate.square()/(estimate+electronic.square())
    metadata.update(raw_observation_domain='counts' if model == 'poisson' else 'readout',
                    floor=floor, scale=scale, clipped_fraction=float((readout < floor).double().mean()),
                    weight_model='post_log_delta_method_approximation',
                    weight_source='post_log_poisson' if model=='poisson' else 'post_log_poisson_gaussian',
                    approximation=True, weight_cap=None, parameters={'floor':floor,'scale':scale})
    return -torch.log(estimate/incident)/scale, arrays, metadata

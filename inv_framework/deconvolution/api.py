"""Deconvolution single-experiment contract, independent of CT and Agent SDKs."""
from dataclasses import dataclass, field
from typing import Any
from inv_framework.experiments import ExperimentResult, planned_capabilities


@dataclass
class DeconvolutionTaskRequest:
    algorithm: str
    inputs: dict[str, Any]
    model: dict[str, Any]
    params: dict[str, Any] = field(default_factory=dict)
    task_type: str = "reconstruction"
    device: str = "cpu"


DeconvolutionTaskResult = ExperimentResult


def list_capabilities():
    """Return executable algorithms separately from documented future methods."""
    return planned_capabilities("deconvolution")


def run_deconvolution_task(request: DeconvolutionTaskRequest | dict) -> DeconvolutionTaskResult:
    import time
    import numpy as np
    from inv_framework.experiments import read_array, resolve_parameters
    from .operators import Convolution
    from .solvers import wiener_hunt, richardson_lucy
    if isinstance(request, dict):
        request = DeconvolutionTaskRequest(**request)
    result = DeconvolutionTaskResult("failed", "deconvolution", request.algorithm)
    started = time.perf_counter()
    try:
        params = resolve_parameters("deconvolution", request.algorithm, request.params)
        if request.algorithm in {'blind_dcp','blind_l0'}:
            from .blind_api import execute
            execute(request, params, result)
            result.runtime_sec = time.perf_counter()-started
            return result
        if request.device != "cpu" or request.task_type != "reconstruction":
            raise ValueError("Implemented deconvolution supports CPU reconstruction only")
        arrays = {key: read_array(value) for key, value in request.inputs.items()}
        if set(arrays) - {"observation", "psf", "initial_estimate"}:
            raise ValueError("Unsupported deconvolution input")
        y, psf = arrays["observation"], arrays["psf"]
        if any(np.iscomplexobj(a) or not np.isfinite(a).all() for a in (y, psf)):
            raise ValueError("observation and psf must be finite real arrays")
        y, psf = y.astype(float), psf.astype(float)
        model = request.model
        names = model["axes"]
        if names not in (["y", "x"], ["z", "y", "x"], ["y", "x", "channel"], ["z", "y", "x", "channel"]) or len(names) != y.ndim:
            raise ValueError("axes must declare 2D/3D spatial axes and optional last channel")
        axes = tuple(i for i, name in enumerate(names) if name != "channel")
        origin = model["psf_origin"]
        if psf.ndim != len(axes) or len(origin) != psf.ndim or any(not isinstance(o, int) or o < 0 or o >= n for o, n in zip(origin, psf.shape)):
            raise ValueError("psf_origin must index each PSF spatial axis")
        if any(psf.shape[i] > y.shape[a] for i, a in enumerate(axes)) or min(y.shape) < 1:
            raise ValueError("PSF support must not exceed image spatial shape")
        boundary = model["boundary"]
        if boundary not in {"zero", "periodic"}:
            raise ValueError("boundary must be zero or periodic")
        operator = Convolution(y.shape, psf, origin, axes, boundary)
        result.effective_params = params
        gain, background = float(model.get("gain", 1)), float(model.get("background", 0))
        if request.algorithm == "wiener_hunt":
            if boundary != "periodic" or model.get("noise_model", "gaussian") != "gaussian" or gain != 1 or background != 0:
                raise ValueError("wiener_hunt requires periodic Gaussian Hx model (gain=1, background=0)")
            if "initial_estimate" in arrays:
                raise ValueError("wiener_hunt is a direct method without initialization")
            x = wiener_hunt(operator, y, params["balance"])
            reason, iterations, history = "completed_valid", 0, []
        elif request.algorithm in {'tv_admm','rl_tv','hyper_laplacian'}:
            from .regularized import tv_admm, rl_tv, hyper_hqs
            if gain != 1 or not np.isfinite(background) or background<0:
                raise ValueError('Regularized methods require gain=1 and finite background>=0')
            noise=model.get('noise_model','gaussian')
            if noise not in {'gaussian','poisson'} or (request.algorithm=='rl_tv' and noise!='poisson') or (request.algorithm=='hyper_laplacian' and noise!='gaussian'):
                raise ValueError('Unsupported noise model for algorithm')
            if noise=='gaussian' and background!=0:
                raise ValueError('Gaussian methods require background=0')
            if boundary!='periodic':
                raise ValueError('Regularized methods currently require periodic convolution and differences')
            if noise=='poisson' and (np.any(y<0) or np.any(psf<0) or psf.sum()<=0):
                raise ValueError('Poisson requires nonnegative data and PSF with positive sum')
            default_initial = np.full(y.shape,params['initial_value']) if request.algorithm=='rl_tv' else np.maximum(y-background,0) if noise=='poisson' else y
            initial=arrays.get('initial_estimate',default_initial).astype(float)
            if initial.shape!=y.shape or not np.isfinite(initial).all() or (noise=='poisson' and np.any(initial<0)) or np.iscomplexobj(arrays.get('initial_estimate',initial)):
                raise ValueError('Invalid initial_estimate')
            if request.algorithm=='tv_admm':
                x,iterations,reason,diagnostics=tv_admm(operator,y,initial,params,noise=='poisson',background)
            elif request.algorithm=='rl_tv':
                x,iterations,reason,diagnostics=rl_tv(operator,y,initial,params,background)
            else:
                if not 0<params['p']<1 or params['beta_factor']<=1 or params['beta_max']<params['beta_initial']:
                    raise ValueError('HQS requires 0<p<1, beta_factor>1 and beta_max>=beta_initial')
                x,iterations,reason,diagnostics=hyper_hqs(operator,y,initial,params)
            result.diagnostics.update(diagnostics)
            history=[]
            from inv_framework.core.numpy_math import gradient
            g=gradient(x,axes)
            if request.algorithm=='hyper_laplacian':
                regularizer=float(np.sum(abs(g)**params['p']))
            else:
                smoothing=params['tv_epsilon'] if request.algorithm=='rl_tv' else 0
                regularizer=float(np.sum(np.sqrt(np.sum(g*g,axis=0)+smoothing**2)))
            mean=operator.forward(x)+background
            fit=float(np.sum(np.maximum(mean,1e-12)-y*np.log(np.maximum(mean,1e-12)))) if noise=='poisson' else float(.5*np.sum((mean-y)**2))
            result.diagnostics.update(data_fit=fit,regularizer_value=regularizer,regularizer_contribution=params['lam']*regularizer)
        else:
            if model.get("noise_model", "poisson") != "poisson" or not np.isfinite([gain, background]).all() or gain <= 0 or background < 0 or np.any(y < 0) or np.any(psf < 0) or psf.sum() <= 0:
                raise ValueError("RL requires nonnegative Poisson data/PSF/background and positive gain and PSF sum")
            initial = arrays.get("initial_estimate", np.full(y.shape, params["initial_value"]))
            if initial.shape != y.shape or np.iscomplexobj(initial) or not np.isfinite(initial).all() or np.any(initial < 0):
                raise ValueError("initial_estimate must be finite nonnegative and match observation")
            x, iterations, reason, history = richardson_lucy(operator, y, initial.astype(float),
                gain, background, params["max_iter"], params["tolerance"], params["epsilon"])
        result.diagnostics.update(device="cpu", dtype=str(x.dtype), psf_sum=float(psf.sum()),
            gain=gain, background=background, boundary=boundary, psf_origin=list(origin),
            data_residual=float(np.linalg.norm(gain*operator.forward(x)+background-y)),
            poisson_nll_history=history, minimum=float(x.min()), maximum=float(x.max()))
        result.outputs = {"reconstruction": x}
        result.iterations, result.stopping_reason = iterations, reason
        result.converged = reason == "converged"
        result.status = "success" if reason != "numerical_error" and np.isfinite(x).all() else "failed"
        if result.status == "failed":
            result.stopping_reason = "numerical_error"
            result.error = {"code": "numerical_error", "message": result.diagnostics.get("message", "Nonfinite reconstruction")}
    except (ValueError, KeyError, OSError, NotImplementedError) as error:
        code = "not_implemented" if isinstance(error, NotImplementedError) else "invalid_parameters"
        result.stopping_reason = code
        result.error = {"code": code, "message": str(error)}
    result.runtime_sec = time.perf_counter()-started
    return result

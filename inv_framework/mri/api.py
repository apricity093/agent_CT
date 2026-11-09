"""MRI single-experiment contract, independent of CT and Agent SDKs."""
from dataclasses import dataclass, field
from typing import Any
from inv_framework.experiments import ExperimentResult, planned_capabilities


@dataclass
class MRITaskRequest:
    algorithm: str
    inputs: dict[str, Any]
    model: dict[str, Any]
    params: dict[str, Any] = field(default_factory=dict)
    task_type: str = "reconstruction"
    device: str = "cpu"
    fidelity: str | None = None


MRITaskResult = ExperimentResult


def list_capabilities(context=None):
    """Return implemented algorithms and optionally screen recommendations."""
    return planned_capabilities("mri", context=context)


def run_mri_task(request: MRITaskRequest | dict) -> MRITaskResult:
    import time
    import numpy as np
    from inv_framework.experiments import read_array, resolve_parameters
    from .operators import SenseEncoding
    from .solvers import conjugate_gradient, cs_reconstruct
    if isinstance(request, dict):
        request = MRITaskRequest(**request)
    from inv_framework.experiments import COMPOSABLE_METHODS as METHODS
    if request.algorithm in METHODS:
        from inv_framework.composition_api import run_composition
        return run_composition(request, "mri")
    result = MRITaskResult("failed", "mri", request.algorithm)
    started = time.perf_counter()
    try:
        params = resolve_parameters("mri", request.algorithm, request.params)
        from inv_framework.experiments import resolve_fidelity
        record = next(r for r in list_capabilities()["algorithms"] if r["name"] == request.algorithm)
        fidelity = resolve_fidelity(record, request.fidelity, request.model, request.inputs)
        result.diagnostics["fidelity"] = fidelity
        result.diagnostics["generation_model"] = request.model.get("noise_model", "unknown")
        if request.algorithm != "varnet" and request.device != "cpu":
            raise ValueError("MRI currently supports device=cpu only")
        if request.task_type != ("calibration" if request.algorithm == "espirit" else "reconstruction"):
            raise ValueError("espirit requires calibration; other methods require reconstruction")
        model = request.model
        shape = tuple(model["spatial_shape"])
        spatial_names = ["y", "x"] if len(shape) == 2 else ["z", "y", "x"]
        if len(shape) not in (2, 3) or any(not isinstance(n, int) or n < 1 for n in shape):
            raise ValueError("spatial_shape must contain two or three positive integers")
        if model["axes"] != ["batch", "coil"] + spatial_names:
            raise ValueError("axes must declare batch, coil, then spatial axes")
        if model.get("fft_norm", "ortho") != "ortho" or model.get("fft_centered", True) is not True:
            raise ValueError("Only centered ortho FFT is supported")
        arrays = {key: read_array(value) for key, value in request.inputs.items()}
        if request.algorithm == 'varnet':
            from .varnet_inference import infer
            if request.device != 'cpu' and not request.device.startswith('cuda'):
                raise ValueError('VarNet device must be cpu or cuda[:index]')
            result.effective_params = params
            result.outputs, diagnostics = infer(arrays, model, params, request.device)
            result.diagnostics.update(diagnostics)
            result.iterations = params['num_cascades']
            result.status = 'success' if all(np.isfinite(a).all() for a in result.outputs.values()) else 'failed'
            result.stopping_reason = 'completed_valid' if result.status == 'success' else 'numerical_error'
            if result.status == 'failed':
                result.error = dict(code='numerical_error', message='Nonfinite VarNet output')
            result.runtime_sec = time.perf_counter()-started
            return result

        if set(arrays) - {"kspace", "sampling_mask", "trajectory", "sensitivity_maps", "data_weights", "initial_estimate", "calibration_data"}:
            raise ValueError("Unsupported MRI input (density compensation must be explicitly supplied as data_weights)")
        if request.algorithm in {"grappa", "espirit"}:
            from .calibration import grappa, espirit
            if len(shape) != 2 or model['sampling'] != 'cartesian':
                raise ValueError('GRAPPA/ESPIRiT require 2D Cartesian data')
            allowed = {'calibration_data'} if request.algorithm == 'espirit' else {'calibration_data','kspace','sampling_mask'}
            if set(arrays) != allowed:
                raise ValueError(f'{request.algorithm} inputs must be {allowed}')
            acs=arrays['calibration_data'].astype(complex)
            width=params['kernel_width']
            if width % 2 != 1 or acs.ndim != 4 or min(acs.shape) < 1 or min(acs.shape[-2:]) < width or not np.isfinite(acs).all() or any(a>b for a,b in zip(acs.shape[-2:],shape)):
                raise ValueError('calibration_data must be finite (batch,coil,acs_y,acs_x), fit spatial_shape and contain odd kernel_width')
            if request.algorithm == 'espirit' and (params['svd_threshold'] >= 1 or params['crop'] > 1 or not np.any(acs)):
                raise ValueError('ESPIRiT needs nonzero ACS, 0<svd_threshold<1 and 0<=crop<=1')
            result.effective_params=params
            if request.algorithm == 'espirit':
                maps,values,diagnostics=espirit(acs,shape,width,params['svd_threshold'],params['crop'])
                result.outputs=dict(sensitivity_maps=maps,eigenvalues=values)
            else:
                y=arrays['kspace'].astype(complex); mask=arrays['sampling_mask']
                if y.shape != acs.shape[:2]+shape or mask.shape != y.shape or not np.isfinite(y).all() or not np.isin(mask,[0,1]).all() or not np.all(mask==mask[:,:1]) or y.shape[1]<2:
                    raise ValueError('GRAPPA requires multicoil kspace and shared binary coil masks of matching shape')
                filled,diagnostics=grappa(y*mask,mask,acs,width,params['ridge'])
                result.outputs=dict(kspace=filled,reconstruction=np.sqrt(np.sum(abs(SenseEncoding(np.ones_like(filled)).fourier(filled,inverse=True))**2,axis=1)))
            result.diagnostics.update(diagnostics)
            result.status='success'; result.stopping_reason='completed_valid'
            result.runtime_sec=time.perf_counter()-started
            return result
        if 'calibration_data' in arrays:
            raise ValueError('Calibration is explicit; this reconstruction does not use calibration_data')
        y = arrays["kspace"].astype(np.complex128)
        if y.ndim < 3 or min(y.shape) < 1 or not np.isfinite(y).all():
            raise ValueError("kspace must be finite and have batch/coil axes")
        maps = arrays.get("sensitivity_maps")
        if maps is None:
            if request.algorithm in {"cg_sense", "cs_tv", "cs_wavelet"} or params.get("coil_combination") == "sense":
                raise ValueError("sensitivity_maps are required, including explicit ones for single coil")
            maps = np.ones(y.shape[:2]+shape, dtype=complex)
        if maps.shape != y.shape[:2]+shape or not np.isfinite(maps).all():
            raise ValueError("sensitivity_maps shape must be (batch,coil,*spatial_shape)")
        trajectory, mask = None, None
        if model["sampling"] == "cartesian":
            mask = arrays["sampling_mask"]
            if y.shape[2:] != shape or mask.shape != y.shape or not np.isin(mask, [0, 1]).all():
                raise ValueError("Cartesian mask must be binary and match kspace shape")
            y = y * mask
        elif model["sampling"] == "noncartesian":
            if request.algorithm == "zero_filled":
                raise ValueError("zero_filled requires Cartesian sampling")
            trajectory = arrays["trajectory"].astype(float)
            if y.ndim != 3 or trajectory.shape != (y.shape[-1], len(shape)) or not np.isfinite(trajectory).all():
                raise ValueError("trajectory must have shape (sample, spatial_dimension)")
            units = model["trajectory_units"]
            if units == "cycles/voxel":
                trajectory *= 2*np.pi
            elif units == "grid_indices":
                trajectory *= 2*np.pi/np.asarray(shape)
            elif units != "radians/voxel":
                raise ValueError("Unknown trajectory_units")
            if np.any(trajectory < -np.pi) or np.any(trajectory >= np.pi):
                raise ValueError("trajectory must lie in [-pi,pi) after conversion")
        else:
            raise ValueError("sampling must be cartesian or noncartesian")
        weights = arrays.get("data_weights", np.ones(y.shape))
        if weights.shape != y.shape or np.iscomplexobj(weights) or not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("data_weights must be finite nonnegative real weights matching kspace")
        if fidelity == "l2" and "data_weights" in arrays:
            raise ValueError("l2 is unweighted; use gaussian_nll or wls with data_weights")
        operator = SenseEncoding(maps, mask, trajectory)
        result.effective_params = params
        history = []
        if request.algorithm == "zero_filled":
            if "data_weights" in arrays or "initial_estimate" in arrays:
                raise ValueError("zero_filled does not use weights or initialization")
            coils = operator.fourier(y, inverse=True)
            if params["coil_combination"] == "rss":
                x = np.sqrt(np.sum(abs(coils)**2, axis=1))
            else:
                power = np.sum(abs(maps)**2, axis=1)
                x = np.divide(np.sum(maps.conj()*coils, axis=1), power,
                              out=np.zeros_like(coils[:, 0]), where=power > 0)
            reason, iterations, history = "completed_valid", 0, []
        else:
            initial = arrays.get("initial_estimate", np.zeros((y.shape[0],)+shape, complex))
            if initial.shape != (y.shape[0],)+shape or not np.isfinite(initial).all():
                raise ValueError("initial_estimate shape must be (batch,*spatial_shape)")
            if request.algorithm in {'cs_wavelet','cs_tv'}:
                if request.algorithm == 'cs_wavelet' and any(n % 2**params['levels'] for n in shape):
                    raise ValueError('Haar spatial dimensions must be divisible by 2**levels')
                x, iterations, reason, diagnostics = cs_reconstruct(operator,y,weights,initial.astype(complex),params,request.algorithm=='cs_wavelet')
                result.diagnostics.update(diagnostics)
            else:
                rhs = operator.adjoint(weights*y)
                normal = lambda x: operator.adjoint(weights*operator.forward(x)) + params["lam"]*x
                x, iterations, reason, history = conjugate_gradient(normal, rhs,
                    initial.astype(complex), params["max_iter"], params["tolerance"])
                result.diagnostics["normal_residual"] = float(np.linalg.norm(normal(x)-rhs)/max(np.linalg.norm(rhs), np.finfo(float).tiny))
        history = result.diagnostics.get("residual_history", history)
        result.outputs = {"reconstruction": x}
        # RSS is a magnitude target, not an encoded complex object.
        residual = operator.forward(x)-y if request.algorithm in {"cg_sense", "cs_tv", "cs_wavelet"} or params["coil_combination"] == "sense" else mask*operator.fourier(coils)-y
        result.diagnostics.update(device="cpu", dtype=str(x.dtype), residual_history=history,
            sampling=model["sampling"], spatial_shape=list(shape), fft_norm="ortho", fft_centered=True,
            trajectory_units=model.get("trajectory_units"),
            data_residual=float(np.linalg.norm(residual)),
            weighted_data_residual=float(np.linalg.norm(np.sqrt(weights)*residual)),
            residual_domain="encoded_object" if request.algorithm in {"cg_sense", "cs_tv", "cs_wavelet"} or params.get("coil_combination") == "sense" else "coil_images",
            coil_combination=params.get("coil_combination", "sense"))
        result.iterations, result.stopping_reason = iterations, reason
        result.converged = reason == "converged"
        result.status = "success" if reason != "numerical_error" and np.isfinite(x).all() else "failed"
        if result.status == "failed":
            result.error = {"code": "numerical_error", "message": "CG failed numerically"}
    except (ValueError, KeyError, OSError, NotImplementedError, ImportError) as error:
        code = "missing_checkpoint" if isinstance(error, FileNotFoundError) and request.algorithm == "varnet" else "not_implemented" if isinstance(error, NotImplementedError) else "unavailable" if isinstance(error, ImportError) else "invalid_parameters"
        result.stopping_reason = code
        result.error = {"code": code, "message": str(error)}
    result.runtime_sec = time.perf_counter()-started
    return result

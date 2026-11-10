"""Public boundary for shared MRI and deconvolution composition tasks."""
import time
import numpy as np
import torch
from inv_framework.experiments import ExperimentResult, read_array, resolve_fidelity
from inv_framework.solvers.composition import CompositeSolver, make_regularizer
from inv_framework.core.fidelity import QuadraticFidelity
from inv_framework.core.imaging_operators import CartesianSense, KnownConvolution


def run_composition(request, domain):
    started=time.perf_counter(); result=ExperimentResult('failed',domain,request.algorithm)
    try:
        from inv_framework.experiments import planned_capabilities
        from inv_framework.experiments import resolve_execution_parameters
        p=resolve_execution_parameters(domain,request,result)
        record=next(r for r in planned_capabilities(domain)['algorithms'] if r['name']==request.algorithm)
        f=resolve_fidelity(record,request.fidelity,request.model,request.inputs)
        if request.device!='cpu' or request.task_type!='reconstruction':
            raise ValueError('Shared imaging composition supports CPU reconstruction')
        arrays={k:read_array(v) for k,v in request.inputs.items()}; model=request.model
        if domain=='mri':
            if model['sampling']!='cartesian' or len(model['spatial_shape'])!=2 or model['axes']!=['batch','coil','y','x']:
                raise ValueError('Shared MRI requires native complex 2D Cartesian SENSE')
            if model.get('fft_norm','ortho')!='ortho' or not model.get('fft_centered',True):
                raise ValueError('MRI requires centered orthonormal FFT')
            if set(arrays)-{'kspace','sensitivity_maps','sampling_mask','data_weights','initial_estimate'}:
                raise ValueError('Unsupported MRI input')
            y=np.asarray(arrays['kspace'],dtype=np.complex128); maps=arrays['sensitivity_maps']; mask=arrays['sampling_mask']
            if y.ndim!=4 or maps.shape!=y.shape or mask.shape!=y.shape or tuple(y.shape[2:])!=tuple(model['spatial_shape']) or not np.isin(mask,[0,1]).all():
                raise ValueError('SENSE maps, binary mask and kspace must have matching (B,C,H,W) shapes')
            if p['min_value'] is not None or p['max_value'] is not None:
                raise ValueError('Complex MRI does not accept real box constraints')
            operator=CartesianSense(torch.as_tensor(maps,dtype=torch.complex128),torch.as_tensor(mask,dtype=torch.float64))
            y=y*mask; shape=(y.shape[0],*y.shape[2:])
        else:
            if model['axes']!=['y','x'] or model['boundary'] not in ('periodic','zero'):
                raise ValueError('Shared deconvolution requires scalar 2D known PSF, periodic or zero boundary')
            if set(arrays)-{'observation','psf','data_weights','initial_estimate'}:
                raise ValueError('Unsupported deconvolution input')
            y=arrays['observation'];psf=arrays['psf'];origin=model['psf_origin']
            if np.iscomplexobj(y) or np.iscomplexobj(psf) or y.ndim!=2 or psf.ndim!=2 or any(n>m for n,m in zip(psf.shape,y.shape)) or len(origin)!=2 or any(type(o)!=int or o<0 or o>=n for o,n in zip(origin,psf.shape)):
                raise ValueError('Require finite real image and 2D PSF with explicit in-support origin')
            if model.get('gain',1)!=1 or model.get('background',0)!=0:
                raise ValueError('Shared quadratic path requires gain=1, background=0; Poisson retains its dedicated paths')
            operator=KnownConvolution(y.shape,torch.as_tensor(psf,dtype=torch.float64),origin,model['boundary'])
            y=np.asarray(y,dtype=float)[None];shape=y.shape
        if not all(np.isfinite(a).all() for a in arrays.values()):raise ValueError('Inputs must be finite')
        weights=arrays.get('data_weights',np.ones_like(y.real))
        if weights.shape!=y.shape and not (domain=='deconvolution' and weights.shape==y.shape[1:]):
            raise ValueError('Inverse-variance data_weights must match observations')
        if np.iscomplexobj(weights) or np.any(weights<0):raise ValueError('Weights must be real nonnegative inverse variances')
        if domain=='deconvolution' and weights.shape==y.shape[1:]:weights=weights[None]
        if f=='l2' and 'data_weights' in arrays:raise ValueError('L2 is unweighted; select Gaussian/WLS')
        if p['regularizer']=='tv_haar' and request.algorithm!='admm':raise ValueError('TV+Haar is implemented for ADMM only')
        initial=arrays.get('initial_estimate',np.zeros(shape,dtype=y.dtype))
        if domain=='deconvolution' and initial.shape==shape[1:]:initial=initial[None]
        if initial.shape!=shape:raise ValueError('initial_estimate must match the reconstruction shape')
        reg=make_regularizer('tv' if p['regularizer']=='tv_haar' else p['regularizer'],operator.domain_shape,p)
        if p["strength"]>0:reg.validate_shape(operator.domain_shape)
        if p['regularizer']=='tv_haar' and p['haar_strength']>0:make_regularizer('haar_l1',operator.domain_shape,p).validate_shape(operator.domain_shape)
        if p['min_value'] is not None and p['max_value'] is not None and p['min_value']>p['max_value']:raise ValueError('Invalid box')
        from inv_framework.solvers.quadratic import METHODS, QuadraticSolver, validate_quadratic
        if request.algorithm in METHODS:validate_quadratic(request.algorithm,p)
        solver=(QuadraticSolver if request.algorithm in METHODS else CompositeSolver)(request.algorithm,**p)
        solver.fidelity=QuadraticFidelity(torch.as_tensor(y),torch.as_tensor(weights,dtype=torch.float64))
        solved=solver.solve_detailed(torch.as_tensor(y),operator,x_init=torch.as_tensor(initial,dtype=torch.as_tensor(y).dtype))
        x=solved.reconstruction.cpu().numpy()
        result.outputs={'reconstruction':x if domain=='mri' else x[0]}
        for key in ('step_size','tau','sigma','rho'):
            if key in p and solved.metadata.get(key) is not None:
                p[key] = solved.metadata[key]
        result.effective_params=p;result.iterations=solved.actual_iterations
        result.stopping_reason=solved.stopping_reason;result.converged=solved.status=='converged'
        result.status='success' if solved.status not in ('numerical_error','invalid_parameters','cancelled','diverged','resource_exhausted') and np.isfinite(x).all() else 'failed'
        result.diagnostics.update(dict(solved.metadata,convergence_status=solved.status,objective=solved.final_objective,
            history=[r.to_dict() for r in solved.trajectory],fidelity=f,weights='inverse variance; 0.5 sum W |Ax-y|^2',
            generation_model=model.get('noise_model','unknown'),reconstruction_assumption='linear quadratic',
            preprocessing='mask kspace' if domain=='mri' else 'none',constraints=dict(min_value=p['min_value'],max_value=p['max_value'])))
        if result.status == 'failed':
            result.error=dict(code='numerical_error',message=solved.stopping_reason)
    except (ValueError,KeyError,OSError) as error:
        result.stopping_reason='invalid_parameters';result.error=dict(code='invalid_parameters',message=str(error))
    result.runtime_sec=time.perf_counter()-started
    return result

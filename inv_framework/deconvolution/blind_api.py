"""Blind-task boundary validation and execution."""
import numpy as np
from inv_framework.experiments import read_array
from .blind import reconstruct, normalize_kernel


def execute(request, p, result):
    model = request.model
    if request.task_type != 'blind_deconvolution' or request.device != 'cpu':
        raise ValueError('Blind methods require task_type=blind_deconvolution and device=cpu')
    if set(request.inputs)-{'observation','initial_psf'}:
        raise ValueError('Blind inputs are observation and optional initial_psf; truth/known psf is evaluation only')
    y = read_array(request.inputs['observation'])
    if model['axes'] not in (['y','x'],['y','x','channel']) or len(model['axes'])!=y.ndim or min(y.shape[:2])<3 or min(y.shape)<1 or np.iscomplexobj(y) or not np.isfinite(y).all():
        raise ValueError('Blind observation must be finite real 2D or channel-last 2D')
    if model['boundary']!='periodic' or model.get('noise_model','gaussian')!='gaussian' or model.get('gain',1)!=1 or model.get('background',0)!=0:
        raise ValueError('Blind methods support periodic Gaussian y=Hx only')
    support = model['kernel_shape']
    if len(support)!=2 or any(not isinstance(n,int) or n<3 or n%2!=1 or n>size for n,size in zip(support,y.shape)):
        raise ValueError('kernel_shape requires two odd sizes >=3 within observation shape')
    if model.get('psf_origin',[n//2 for n in support]) != [n//2 for n in support]:
        raise ValueError('Blind PSF origin is the center of kernel_shape')
    initial = None
    if 'initial_psf' in request.inputs:
        initial = read_array(request.inputs['initial_psf'])
        if initial.shape!=tuple(support) or np.iscomplexobj(initial) or not np.isfinite(initial).all() or np.any(initial<0) or initial.sum()<=0:
            raise ValueError('initial_psf must be nonnegative, positive-sum and match kernel_shape')
        initial = normalize_kernel(initial.astype(float))
    if not 0<p['scale_factor']<1 or p['coarse_kernel_size']<3:
        raise ValueError('Require 0<scale_factor<1 and coarse_kernel_size>=3')
    if request.algorithm=='blind_dcp' and (p['patch_size']%2!=1 or p['penalty_factor']<=1):
        raise ValueError('DCP requires odd patch_size and penalty_factor>1')
    x,k,iterations,diagnostics = reconstruct(y.astype(float),support,initial,p,request.algorithm)
    result.outputs = dict(reconstruction=x,psf=k)
    result.effective_params = p
    result.diagnostics.update(device='cpu',boundary='periodic',**diagnostics)
    result.iterations = iterations
    result.stopping_reason = 'max_iterations'
    result.status = 'success' if np.isfinite(x).all() and np.isfinite(k).all() else 'failed'
    if result.status=='failed':
        result.stopping_reason='numerical_error'
        result.error=dict(code='numerical_error',message='Nonfinite blind output')

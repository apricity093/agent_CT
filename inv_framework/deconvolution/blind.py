"""Uniform DCP (Pan 2016) and unnatural L0 (Xu 2013) kernel estimation.

Periodic image model; compact centered PSFs. Final natural-image recovery uses
the existing Hyper-Laplacian HQS, separately from the structural latent image.
"""
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
from scipy.optimize import minimize
from inv_framework.core.numpy_math import gradient, gradient_adjoint, laplacian_spectrum, conjugate_gradient
from inv_framework.experiments import resolve_parameters
from .operators import Convolution
from .regularized import hyper_hqs


def dark_channel(x, patch):
    """Return minima and flattened original-image indices (including channels).

    Periodic patches match the declared convolution boundary. Ties select the
    first encountered location; scatter-add is the exact transpose of gathering.
    """
    image = x[..., None] if x.ndim == 2 else x
    h, w, c = image.shape
    yy, xx = np.indices((h, w))
    values = np.full((h, w), np.inf)
    indices = np.zeros((h, w), dtype=int)
    for dy in range(-patch//2+1, patch//2+1):
        for dx in range(-patch//2+1, patch//2+1):
            row, col = (yy+dy) % h, (xx+dx) % w
            pixels = image[row, col]
            channel = pixels.argmin(-1)
            candidate = np.take_along_axis(pixels, channel[..., None], -1)[..., 0]
            use = candidate < values
            indices[use] = ((row*w+col)*c+channel)[use]
            values = np.minimum(values, candidate)
    return values, indices


def scatter(values, indices, shape):
    return np.bincount(indices.ravel(), weights=values.ravel(), minlength=np.prod(shape)).reshape(shape)


def resize_centered(x, shape, scales, periodic=False):
    """Resample about the declared integer origin, preserving displacement units."""
    coords = [(np.arange(n)-n//2)/scale+old//2
              for n, old, scale in zip(shape, x.shape, scales)]
    grid = np.meshgrid(*coords, indexing='ij')
    return map_coordinates(x, grid, order=1, mode='grid-wrap' if periodic else 'constant', cval=0)


def normalize_kernel(k):
    k = np.maximum(k, 0)
    if k.sum() <= 0:
        raise ValueError('Kernel estimate has no positive mass; observation cannot support this estimate')
    return k/k.sum()


def center_kernel(k):
    center = np.array(k.shape)//2
    com = np.array([(v*k).sum()/k.sum() for v in np.indices(k.shape)])
    shift = center-np.rint(com).astype(int)
    out = np.zeros_like(k)
    src = tuple(slice(max(-s,0), min(n-s,n)) for n,s in zip(k.shape,shift))
    dst = tuple(slice(max(s,0), min(n+s,n)) for n,s in zip(k.shape,shift))
    out[dst] = k[src]
    return normalize_kernel(out), shift


def latent_l0(op, y, x, p):
    """Xu equations 9--12: scalar thresholding and exact periodic FFT solve."""
    lap = laplacian_spectrum(x.shape, (0,1))
    rhs = op.otf.conj()*np.fft.fftn(y, axes=(0,1))
    for level in range(p['epsilon_levels']):
        epsilon = p['epsilon_initial']/2**level
        weight = p['lam']/epsilon**2
        for _ in range(2**level):
            g = gradient(x, (0,1))
            auxiliary = np.where(abs(g)>epsilon, g, 0)
            numerator = rhs+weight*np.fft.fftn(gradient_adjoint(auxiliary,(0,1)), axes=(0,1))
            x = np.fft.ifftn(numerator/(abs(op.otf)**2+weight*lap), axes=(0,1)).real
    return x, {}


def latent_dcp(op, y, x, p):
    """HQS with frozen min-index selection and its exact scatter transpose."""
    alpha, beta = p['penalty_initial'], p['penalty_initial']
    history = []
    for _ in range(p['latent_iter']):
        g = gradient(x,(0,1))
        g = np.where(g*g>p['gradient_weight']/alpha, g, 0)
        dark, indices = dark_channel(x,p['patch_size'])
        u = np.where(dark*dark>p['dark_weight']/beta,dark,0)
        counts = scatter(np.ones_like(dark),indices,x.shape)
        rhs = op.adjoint(y)+alpha*gradient_adjoint(g,(0,1))+beta*scatter(u,indices,x.shape)
        normal = lambda v: op.adjoint(op.forward(v))+alpha*gradient_adjoint(gradient(v,(0,1)),(0,1))+beta*counts*v
        x, _, reason, residuals = conjugate_gradient(normal,rhs,x,p['cg_max_iter'],p['cg_tolerance'])
        history.append(dict(alpha=alpha,beta=beta,normal_residual=residuals[-1],inner_stop=reason))
        alpha *= p['penalty_factor']; beta *= p['penalty_factor']
    return x, dict(latent_history=history)


def estimate_kernel(x,y,k,p,use_gradients):
    """Compact nonnegative ridge LS; FFT correlations give the exact derivative."""
    xx = gradient(x,(0,1)) if use_gradients else x[None]
    yy = gradient(y,(0,1)) if use_gradients else y[None]
    transforms = np.fft.fftn(xx,axes=(1,2))
    origin = tuple(n//2 for n in k.shape)
    rows, cols = np.indices(k.shape)
    rows, cols = (rows-origin[0])%x.shape[0], (cols-origin[1])%x.shape[1]
    def objective(flat):
        trial = flat.reshape(k.shape)
        op = Convolution(x.shape,trial,origin,(0,1),'periodic')
        residual = np.stack([op.forward(v) for v in xx])-yy
        correlation = np.fft.ifftn(transforms.conj()*np.fft.fftn(residual,axes=(1,2)),axes=(1,2)).real
        if x.ndim == 3:
            correlation = correlation.sum(-1)
        grad = correlation.sum(0)[rows,cols]+p['kernel_ridge']*trial
        value = .5*(np.sum(residual**2)+p['kernel_ridge']*np.sum(trial**2))
        return value,grad.ravel()
    fit = minimize(objective,k.ravel(),jac=True,bounds=[(0,None)]*k.size,
                   method='L-BFGS-B',options=dict(maxiter=p['kernel_max_iter'],ftol=1e-10))
    return normalize_kernel(fit.x.reshape(k.shape)), dict(kernel_solver_stop=str(fit.message),kernel_solver_iterations=int(fit.nit))


def reconstruct(y,support,initial,p,method):
    scales = [1.0]
    while min(np.array(support)*scales[-1])>p['coarse_kernel_size']:
        scales.append(scales[-1]*p['scale_factor'])
    k = initial
    previous_image_shape = y.shape[:2]
    history = []
    for scale in reversed(scales):
        shape = tuple(max(s, int(round(n*scale))) for n,s in zip(y.shape[:2],(3,3)))
        factors = np.array(shape)/y.shape[:2]
        kernel_shape = tuple(min(2*((n-1)//2)+1, max(3,2*int(round((s//2)*f))+1)) for s,f,n in zip(support,factors,shape))
        filtered = gaussian_filter(y,tuple(max(0,(1/f-1)*.5) for f in factors)+( (0,) if y.ndim==3 else ()),mode='wrap')
        low = resize_centered(filtered,shape+y.shape[2:],tuple(factors)+( (1,) if y.ndim==3 else ()),True)
        if k is None:
            k = np.zeros(kernel_shape); k[kernel_shape[0]//2,kernel_shape[1]//2]=1
        else:
            ratios = np.array(shape)/previous_image_shape
            k = normalize_kernel(resize_centered(k,kernel_shape,ratios))
        previous_image_shape = shape
        x = low.copy()
        for iteration in range(p['max_iter']):
            op = Convolution(low.shape,k,tuple(n//2 for n in k.shape),(0,1),'periodic')
            x, diagnostics = (latent_dcp if method=='blind_dcp' else latent_l0)(op,low,x,p)
            k, kernel_diagnostics = estimate_kernel(x,low,k,p,method=='blind_dcp')
            k,shift = center_kernel(k)
            x = np.roll(x,tuple(-shift),(0,1))
            history.append(dict(scale=float(scale),image_shape=list(shape),kernel_shape=list(k.shape),iteration=iteration+1,center_shift=shift.tolist(),**diagnostics,**kernel_diagnostics))
    op = Convolution(y.shape,k,tuple(n//2 for n in k.shape),(0,1),'periodic')
    final = resolve_parameters('deconvolution','hyper_laplacian',dict(lam=p['final_lam'],p=.5,max_iter=p['final_max_iter']))
    image,iterations,reason,diagnostics = hyper_hqs(op,y,y,final)
    return image,k,len(history),dict(scale_history=history,scales=len(scales),psf_shape=list(k.shape),
        psf_sum=float(k.sum()),psf_minimum=float(k.min()),psf_origin=[n//2 for n in k.shape],
        data_residual=float(np.linalg.norm(op.forward(image)-y)),final_recovery='hyper_laplacian_p0.5_hqs',
        final_iterations=iterations,final_stopping_reason=reason,final_params=final,
        final_relative_change=diagnostics['residual_history'][-1]['relative_change'])

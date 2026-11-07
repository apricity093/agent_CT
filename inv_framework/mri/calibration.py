"""Cartesian patch calibration for GRAPPA and single-map ESPIRiT."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from .operators import fft


def patches(acs, width):
    # rows: valid spatial windows; columns: coil, kernel y, kernel x.
    return sliding_window_view(acs,(width,width),axis=(-2,-1)).transpose(1,2,0,3,4).reshape(-1,acs.shape[0],width,width)


def grappa(kspace, mask, calibration, width, ridge):
    result = kspace.copy(); errors=[]; patterns=0
    pad=width//2
    for batch in range(len(kspace)):
        acs=patches(calibration[batch],width)
        target=acs[:,:,pad,pad]
        source=np.pad(kspace[batch],((0,0),(pad,pad),(pad,pad)))
        sampled=np.pad(mask[batch,0].astype(bool),pad)
        groups={}
        for y,x in np.argwhere(~mask[batch,0].astype(bool)):
            pattern=sampled[y:y+width,x:x+width]
            groups.setdefault(tuple(pattern.ravel()),[]).append((y,x))
        for pattern, locations in groups.items():
            selection=np.asarray(pattern).reshape(width,width)
            if not selection.any():
                raise ValueError('GRAPPA kernel has missing targets without acquired neighbors; increase kernel_width')
            design=acs[:,:,selection].reshape(len(acs),-1)
            gram=design.conj().T@design
            scale=np.linalg.norm(gram)/len(gram)
            # Augmented least squares also defines the zero-ridge rank-deficient case.
            penalty=np.sqrt(ridge*scale)*np.eye(design.shape[1])
            kernel=np.linalg.lstsq(np.concatenate([design,penalty]),
                np.concatenate([target,np.zeros((design.shape[1],target.shape[1]),complex)]),rcond=None)[0]
            errors.append(float(np.linalg.norm(design@kernel-target)/max(np.linalg.norm(target),1e-30)))
            for y,x in locations:
                result[batch,:,y,x]=source[:,y:y+width,x:x+width][:,selection].ravel()@kernel
            patterns+=1
    return result,dict(calibration_relative_residuals=errors,sampling_patterns=patterns,coil_combination='rss')


def espirit(calibration, shape, width, threshold, crop):
    maps=[]; eigenvalues=[]; ranks=[]
    for acs in calibration:
        matrix=patches(acs,width).reshape(-1,acs.shape[0]*width**2)
        _, singular, vh=np.linalg.svd(matrix,full_matrices=False)
        kernels=vh[singular>threshold*singular[0]].reshape(-1,acs.shape[0],width,width)
        covariance=np.zeros(shape+(acs.shape[0],acs.shape[0]),complex)
        for kernel in kernels:
            padded=np.zeros((acs.shape[0],)+shape,complex)
            start=tuple(n//2-width//2 for n in shape)
            padded[:,start[0]:start[0]+width,start[1]:start[1]+width]=kernel
            image=np.moveaxis(fft(padded,2,inverse=True),0,-1)
            covariance+=image[..., :,None]*image[...,None,:].conj()
        covariance*=np.prod(shape)/width**2
        values,vectors=np.linalg.eigh(covariance)
        vector=vectors[...,-1]
        # Fix a display gauge with the strongest coil; subspace is gauge invariant.
        anchor=np.take_along_axis(vector,np.argmax(abs(vector),axis=-1)[...,None],axis=-1)[...,0]
        vector*=np.exp(-1j*np.angle(anchor))[...,None]
        vector*= (values[...,-1]>crop)[...,None]
        maps.append(np.moveaxis(vector,-1,0)); eigenvalues.append(values[...,-1]); ranks.append(len(kernels))
    return np.asarray(maps),np.asarray(eigenvalues),dict(calibration_ranks=ranks,map_sets=1)

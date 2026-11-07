"""Write small external-input experiments; no reconstruction runs implicitly."""
import argparse
from pathlib import Path
import numpy as np
import yaml


def generate(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    yy, xx = np.mgrid[-1:1:24j, -1:1:28j]
    truth = .1 + .8*((xx/.55)**2+(yy/.7)**2 < 1)
    object_image = (truth*np.exp(.3j*xx))[None]
    maps = np.stack([np.ones_like(xx), (.7+.2*xx)*np.exp(1j*yy)])[None]
    coils = maps*object_image[:,None]
    kspace = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(coils, axes=(-2,-1)),
                                      axes=(-2,-1), norm='ortho'), axes=(-2,-1))
    mask = np.ones(kspace.shape)
    mask[:,:,::3] = 0
    mri_model = dict(sampling='cartesian', axes=['batch','coil','y','x'],
                     spatial_shape=[24,28], fft_norm='ortho', fft_centered=True)
    arrays = dict(kspace=kspace,mask=mask,maps=maps,mri_truth=object_image,truth=truth)
    psf = np.array([[.05,.1,.05],[.1,.4,.1],[.05,.1,.05]])
    # Independent circular spatial sum used for synthesis.
    blurred = sum(psf[j,i]*np.roll(truth,(j-1,i-1),(0,1)) for j in range(3) for i in range(3))
    arrays.update(psf=psf,blurred=blurred+rng.normal(0,.01,truth.shape),counts=rng.poisson(30*blurred+.2).astype(float))
    for name, array in arrays.items():
        np.save(destination/f'{name}.npy',array)
    ref = lambda name: dict(path=f'{name}.npy')
    for algorithm in ('zero_filled','cg_sense'):
        task=dict(modality='mri',algorithm=algorithm,
            inputs=dict(kspace=ref('kspace'),sampling_mask=ref('mask'),sensitivity_maps=ref('maps')),
            model=mri_model,params={},evaluation=dict(reference=ref('mri_truth'),data_range=.9,mode='magnitude'))
        (destination/f'{algorithm}.yaml').write_text(yaml.safe_dump(task),encoding='utf-8')
    trajectory = rng.uniform(-.49,.49,(800,2))
    positions = np.stack(np.meshgrid(np.arange(24)-12,np.arange(28)-14,indexing='ij'),-1).reshape(-1,2)
    direct_dft = np.exp(-2j*np.pi*trajectory@positions.T)/np.sqrt(24*28)
    noncart = np.stack([direct_dft@coil.ravel() for coil in coils[0]])[None]
    np.save(destination/'trajectory.npy',trajectory)
    np.save(destination/'noncart.npy',noncart)
    task=dict(modality='mri',algorithm='cg_sense',inputs=dict(kspace=ref('noncart'),
        trajectory=ref('trajectory'),sensitivity_maps=ref('maps')),model=dict(
        sampling='noncartesian',axes=['batch','coil','y','x'],spatial_shape=[24,28],
        trajectory_units='cycles/voxel'),params=dict(lam=.01),evaluation=dict(
        reference=ref('mri_truth'),data_range=.9,mode='magnitude'))
    (destination/'noncartesian_sense.yaml').write_text(yaml.safe_dump(task),encoding='utf-8')
    for algorithm in ('wiener_hunt','richardson_lucy'):
        rl = algorithm=='richardson_lucy'
        model=dict(axes=['y','x'],boundary='periodic',psf_origin=[1,1],noise_model='poisson' if rl else 'gaussian')
        if rl:
            model.update(gain=30,background=.2)
        task=dict(modality='deconvolution',algorithm=algorithm,
            inputs=dict(observation=ref('counts' if rl else 'blurred'),psf=ref('psf')),
            model=model,params={},evaluation=dict(reference=ref('truth'),data_range=.9))
        (destination/f'{algorithm}.yaml').write_text(yaml.safe_dump(task),encoding='utf-8')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    generate(parser.parse_args().output)

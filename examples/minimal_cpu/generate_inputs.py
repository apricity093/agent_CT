"""Generate the bundled 32-pixel CPU inputs; run from the library root.

Uses the existing CT projector, and the synthesis conventions of
tests/core/fixtures/imaging_inputs.py. Existing input files are preserved.
"""
from pathlib import Path
import numpy as np
import torch
import yaml
from inv_framework.ct.backends import ParallelBeamRadon2D


def generate():
    root = Path(__file__).resolve().parent
    yy, xx = np.mgrid[-1:1:32j, -1:1:32j]
    image = .1 + .8 * ((xx/.55)**2 + (yy/.7)**2 < 1)
    ref = lambda key: dict(path='input.npz', key=key)
    for domain in ('ct', 'mri', 'deconvolution'):
        folder = root / domain
        folder.mkdir(exist_ok=True)
        if (folder / 'input.npz').exists() or (folder / 'request.yaml').exists():
            raise FileExistsError(f'Inputs already exist in {folder}')
        if domain == 'ct':
            truth = (.014*((xx/.65)**2+(yy/.8)**2 < 1)
                     + .006*(((xx-.2)/.18)**2+((yy+.15)/.3)**2 < 1)).astype('float32')[None,None]
            angles = torch.arange(90, dtype=torch.float32)*torch.pi/90
            operator = ParallelBeamRadon2D(image_size=32, angles=angles, device='cpu')
            observation = operator.forward(torch.from_numpy(truth)).numpy()
            arrays = dict(observation=observation, reference=truth)
            task = dict(modality=domain, algorithm='fbp', backend='torch', device='cpu',
                inputs=dict(observation=ref('observation')), model=dict(geometry=dict(
                    type='parallel_2d', domain_shape=[1,32,32], range_shape=[1,90,32],
                    angles_rad=angles.tolist()), measurement=dict(noise_model='gaussian',
                    observation_model='gaussian', observation_domain='line_integral')),
                params={}, evaluation=dict(reference=ref('reference'), data_range=.02))
        elif domain == 'mri':
            truth = (image*np.exp(.3j*xx))[None]
            maps = np.stack([np.ones_like(xx), (.7+.2*xx)*np.exp(1j*yy)])[None]
            # Fix the sensitivity convention explicitly: sum_coil |S|^2 = 1.
            maps /= np.sqrt(np.sum(abs(maps)**2, axis=1, keepdims=True))
            coils = maps*truth[:,None]
            kspace = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(coils, axes=(-2,-1)),
                axes=(-2,-1), norm='ortho'), axes=(-2,-1))
            mask = np.ones(kspace.shape, dtype='uint8'); mask[:,:,::3] = 0
            arrays = dict(kspace=kspace*mask, sampling_mask=mask, sensitivity_maps=maps, reference=truth)
            task = dict(modality=domain, algorithm='cg_sense', device='cpu', task_type='reconstruction',
                inputs={key:ref(key) for key in ('kspace','sampling_mask','sensitivity_maps')},
                model=dict(sampling='cartesian', axes=['batch','coil','y','x'], spatial_shape=[32,32],
                    fft_norm='ortho', fft_centered=True), params=dict(lam=.001),
                stopping_policy=dict(max_iterations=50, tolerance=1e-6),
                evaluation=dict(reference=ref('reference'), data_range=.9, mode='magnitude'))
        else:
            psf = np.array([[.05,.1,.05],[.1,.4,.1],[.05,.1,.05]])
            blurred = sum(psf[j,i]*np.roll(image,(j-1,i-1),(0,1)) for j in range(3) for i in range(3))
            counts = np.random.default_rng(42).poisson(30*blurred+.2)
            arrays = dict(observation=counts, psf=psf, reference=image)
            task = dict(modality=domain, algorithm='richardson_lucy', device='cpu', task_type='reconstruction',
                inputs=dict(observation=ref('observation'), psf=ref('psf')),
                model=dict(axes=['y','x'], boundary='periodic', psf_origin=[1,1], noise_model='poisson',
                    gain=30., background=.2), params=dict(epsilon=1e-12, initial_value=.5),
                stopping_policy=dict(max_iterations=50, tolerance=.003), problem_constraints=dict(nonnegative=True),
                evaluation=dict(reference=ref('reference'), data_range=.9))
        np.savez(folder / 'input.npz', **arrays)
        (folder / 'request.yaml').write_text(yaml.safe_dump(task, sort_keys=False), encoding='utf-8')


if __name__ == '__main__':
    generate()

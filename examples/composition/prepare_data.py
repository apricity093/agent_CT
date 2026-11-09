"""Prepare the three small, fixed development acquisitions used by the examples."""
import json
from pathlib import Path
import argparse
import runpy

import torch
import numpy as np
from inv_framework.ct.backends import ParallelBeamRadon2D
from inv_framework.core.imaging_operators import CartesianSense, KnownConvolution

ROOT = Path(__file__).resolve().parents[2]


DIRECTORIES = dict(ct='ct_seeded_dev',mri='mri_standard_dev_seed0',deconvolution='camera_dev_seed0')


def prepare(domain):
    destination=ROOT/'examples/composition/data'/DIRECTORIES[domain]
    files=('public.npz','reference.npz','source.json')
    if all((destination/name).exists() for name in files):
        print(f'Already prepared: {destination.relative_to(ROOT)}')
        return
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f'Incomplete data directory: {destination}')
    rng=np.random.default_rng(0)
    if domain=='ct':
        generator=runpy.run_path(str(ROOT/'examples/train_diffusion_for_dps.py'))['random_ellipse_phantom']
        with torch.random.fork_rng():
            torch.manual_seed(2026110801)
            truth=generator(64)[None,None].double().numpy()*.02
        geometry=json.loads((ROOT/'tests/data/cases/parallel_2d__ellipses_sparse_gaussian/case.json').read_text())['geometry']
        operator=ParallelBeamRadon2D(64,angles=torch.tensor(geometry['angles_rad']))
        clean=operator.forward(torch.as_tensor(truth)).numpy()
        sigma=float(np.sqrt(np.mean(clean**2)))*.01
        public=dict(observation=clean+sigma*rng.normal(size=clean.shape))
        source=dict(generator='examples/train_diffusion_for_dps.py:random_ellipse_phantom',
            object_seed=2026110801,num_ellipses=6,scale=.02,geometry=geometry,data_range=.02,
            image_unit='normalized_inverse_length',observation_unit='pixel-summed line integral')
    else:
        cid='standard_mr_dev' if domain=='mri' else 'public_camera_levin'
        case=ROOT/'tests/data'/domain/'cases'/cid
        arrays=dict(np.load(case/'public.npz'))
        truth=np.load(case/'reference.npz')['object']
        metadata=json.loads((case/'case.json').read_text())
        source=dict(source_case=f'tests/data/{domain}/cases/{cid}',source_metadata=metadata,data_range=1.)
        if domain=='mri':
            public={k:arrays[k] for k in ('sensitivity_maps','sampling_mask')}
            operator=CartesianSense(torch.as_tensor(public['sensitivity_maps'],dtype=torch.complex128),torch.as_tensor(public['sampling_mask']))
            clean=operator.forward(torch.as_tensor(truth,dtype=torch.complex128)).numpy()
            sigma=.002
            noisy=(clean+sigma/np.sqrt(2)*(rng.normal(size=clean.shape)+1j*rng.normal(size=clean.shape)))*public['sampling_mask']
            start,end=metadata['acs_columns']
            public.update(kspace=noisy,calibration_data=noisy[...,start:end])
            source.update(noise_model='proper complex Gaussian',variance_definition='E|noise|^2=sigma^2',acs_columns=[start,end])
        else:
            public=dict(psf=arrays['psf'])
            operator=KnownConvolution(truth.shape,torch.as_tensor(public['psf'],dtype=torch.float64),metadata['psf_origin'],'periodic')
            clean=operator.forward(torch.as_tensor(truth)[None]).numpy()[0]
            sigma=.005
            public['observation']=clean+sigma*rng.normal(size=clean.shape)
            source.update(physical_boundary='periodic',psf_origin=metadata['psf_origin'])
    source.update(role='development',noise_seed=0,rng='numpy.random.default_rng',sigma=sigma,
        truth_shape=list(truth.shape),preprocessing='matched forward model, then fixed noise; reference evaluated after solving')
    destination.mkdir(parents=True,exist_ok=True)
    np.savez(destination/'public.npz',**public)
    np.savez(destination/'reference.npz',object=truth)
    (destination/'source.json').write_text(json.dumps(source,indent=2)+'\n',encoding='utf-8')
    print(destination.relative_to(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain',choices=tuple(DIRECTORIES),action='append')
    args=parser.parse_args();torch.set_num_threads(1)
    for domain in args.domain or DIRECTORIES:prepare(domain)


if __name__ == '__main__':
    main()

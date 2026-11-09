"""Small deterministic public composition calls, without an Agent service.

Run from the library root:
    python examples/composition_quickstart.py --domain mri --algorithm admm
Select --regularizer tv_haar for ADMM's joint objective.
"""
import argparse
import json
import numpy as np
from inv_framework.runner import run_experiment


def configuration(domain, algorithm, regularizer):
    image=np.array([[.3,1.2],[.8,.2]])
    params=dict(regularizer=regularizer,strength=.08,level=1,num_iterations=300)
    if regularizer=='tv_haar':params.update(haar_strength=.06,rho_data=1.1,rho_regularizer=.7,rho_haar=1.3,rho_constraint=.9)
    config=dict(modality=domain,algorithm=algorithm,params=params,fidelity='l2')
    if domain=='mri':
        image=image*np.exp(1j*np.array([[.2,.8],[-.5,1.7]]))
        kspace=np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(image),norm='ortho'))[None,None]
        config.update(inputs=dict(kspace=kspace,sampling_mask=np.ones(kspace.shape),sensitivity_maps=np.ones(kspace.shape,complex)),
            model=dict(axes=['batch','coil','y','x'],spatial_shape=[2,2],sampling='cartesian'))
    elif domain=='deconvolution':
        config.update(inputs=dict(observation=image,psf=np.ones((1,1))),model=dict(axes=['y','x'],boundary='periodic',psf_origin=[0,0]))
    else:
        import torch
        from inv_framework.ct.backends import ParallelBeamRadon2D
        angles=torch.tensor([0.,.4,1.,1.6]);operator=ParallelBeamRadon2D(2,angles=angles)
        observation=operator.forward(torch.tensor(image,dtype=torch.float32)[None,None]).numpy()
        if algorithm=='admm':
            params['cg_tolerance']=.1
        config.update(backend='torch',case_id='external',model={},inputs=dict(observation=observation,
            geometry=dict(type='parallel_2d',domain_shape=[1,2,2],range_shape=list(operator.range_shape),angles_rad=angles.tolist()),
            measurement=dict(domain='sinogram',observation_model='line_integral',noise_model='none')))
    return config


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain',choices=['ct','mri','deconvolution'],default='mri')
    parser.add_argument('--algorithm',choices=['admm','pdhg','fista'],default='admm')
    parser.add_argument('--regularizer',default='tv_haar')
    parser.add_argument('--output')
    args=parser.parse_args()
    result=run_experiment(configuration(args.domain,args.algorithm,args.regularizer),args.output)
    print(json.dumps(dict(status=result.status,iterations=result.iterations,converged=result.converged,stopping_reason=result.stopping_reason,error=result.error)))
    raise SystemExit(0 if result.status=='success' else 1)

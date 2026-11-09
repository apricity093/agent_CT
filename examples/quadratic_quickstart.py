"""Run one quadratic baseline through a public domain task, without Agent services."""
import argparse
import numpy as np
from inv_framework.runner import run_experiment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain',choices=('ct','mri','deconvolution'),default='mri')
    parser.add_argument('--method',choices=('landweber','cgls','lsqr','cg'),default='lsqr')
    parser.add_argument('--output', help='Optional directory for result.json and reconstruction.npy')
    args=parser.parse_args()
    params=dict(regularizer='tikhonov',strength=.08,num_iterations=200,tolerance=1e-5)
    if args.domain=='ct':
        config=dict(modality='ct',algorithm=args.method,backend='torch',
                    case_id='parallel_2d__ellipses_dense_clean',params=params)
    else:
        grid=np.arange(16).reshape(4,4)
        image=(1+grid/16)*np.exp(1j*grid/5) if args.domain=='mri' else 1+grid/16
        if args.domain=='mri':
            from inv_framework.mri.operators import SenseEncoding
            maps=np.ones((1,1,4,4),dtype=complex);mask=np.ones_like(maps.real)
            y=SenseEncoding(maps,mask).forward(image[None])
            inputs=dict(kspace=y,sensitivity_maps=maps,sampling_mask=mask)
            model=dict(axes=['batch','coil','y','x'],sampling='cartesian',spatial_shape=[4,4])
        else:
            from inv_framework.deconvolution.operators import Convolution
            psf=np.array([[.8,.2]])
            y=Convolution((4,4),psf,[0,0],(0,1),'periodic').forward(image)
            inputs=dict(observation=y,psf=psf)
            model=dict(axes=['y','x'],boundary='periodic',psf_origin=[0,0])
        config=dict(modality=args.domain,algorithm=args.method,inputs=inputs,model=model,params=params)
    result=run_experiment(config, args.output)
    print(dict(status=result.status,converged=result.converged,iterations=result.iterations,
               stopping_reason=result.stopping_reason,error=result.error))


if __name__=='__main__':main()

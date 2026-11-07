"""Generate independent external observations; checkpoint preparation is explicit."""
import argparse
from pathlib import Path
import numpy as np
import yaml


def generate(destination, checkpoint=None):
    root=Path(destination);root.mkdir(parents=True,exist_ok=True)
    yy,xx=np.mgrid[:64,:80]
    truth=np.zeros((64,80));truth[10:48,16:59]=.8
    truth[((yy-40)/12)**2+((xx-25)/15)**2<1]=.3
    truth[20:25,30:70]=1
    kernel=np.zeros((9,9));kernel[4,2:7]=.2
    observation=sum(kernel[i,j]*np.roll(truth,(i-4,j-4),(0,1)) for i,j in np.ndindex(kernel.shape))
    for name,array in dict(truth=truth,true_psf=kernel,observation=observation).items():np.save(root/(name+'.npy'),array)
    ref=lambda name:dict(path=name+'.npy')
    for method in ('blind_dcp','blind_l0'):
        task=dict(modality='deconvolution',algorithm=method,task_type='blind_deconvolution',
            inputs=dict(observation=ref('observation')),model=dict(axes=['y','x'],boundary='periodic',noise_model='gaussian',kernel_shape=[9,9]),params={},
            evaluation=dict(reference=ref('truth'),data_range=1.))
        (root/(method+'.yaml')).write_text(yaml.safe_dump(task),encoding='utf8')
    # Synthetic MRI is an inference/encoding example, outside the knee training distribution.
    image=truth*np.exp(.3j*np.sin(xx/20))
    maps=np.stack([np.ones_like(image),np.exp(1j*xx/40)])/np.sqrt(2)
    coils=maps*image
    k=np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(coils,axes=(-2,-1)),norm='ortho'),axes=(-2,-1))[None]
    mask=np.zeros(k.shape,bool);mask[...,::4]=1;mask[...,28:52]=1
    np.save(root/'kspace.npy',k*mask);np.save(root/'mask.npy',mask)
    task=dict(modality='mri',algorithm='varnet',inputs=dict(kspace=ref('kspace'),sampling_mask=ref('mask')),
        model=dict(axes=['batch','coil','y','x'],spatial_shape=[64,80],sampling='cartesian'),
        params=dict(checkpoint=str(Path(checkpoint).resolve()) if checkpoint else None,num_low_frequencies=24))
    (root/'varnet.yaml').write_text(yaml.safe_dump(task),encoding='utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True);parser.add_argument('--checkpoint')
    args=parser.parse_args();generate(args.output,args.checkpoint)

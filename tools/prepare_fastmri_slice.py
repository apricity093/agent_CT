"""Prepare one locally authorized fully sampled multicoil fastMRI slice."""
import argparse
from pathlib import Path
import h5py
import numpy as np
import yaml


def prepare(source, index, destination, checkpoint):
    with h5py.File(source) as handle:
        dataset=handle['kspace']
        if dataset.ndim!=4 or dataset.shape[1]<2 or not np.issubdtype(dataset.dtype,np.complexfloating) or not 0<=index<dataset.shape[0]:
            raise ValueError('Expected fully sampled (slice,coil,y,x) kspace and a valid slice index')
        kspace=np.asarray(dataset[index],dtype=np.complex64)[None]
    if not np.isfinite(kspace).all():raise ValueError('kspace must be finite')
    destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    width=kspace.shape[-1];low=24;start=(width-low+1)//2
    if width<low:raise ValueError('Acquisition width must contain the declared 24-column ACS')
    columns=np.zeros(width,bool);columns[::4]=True;columns[start:start+low]=True
    mask=np.broadcast_to(columns,kspace.shape).copy()
    # Evaluation on the native full FOV; this is not the challenge's cropped score.
    coils=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace,axes=(-2,-1)),norm='ortho'),axes=(-2,-1))
    reference=np.sqrt(np.sum(abs(coils)**2,axis=1)).astype(np.float32)
    np.savez_compressed(destination/'public.npz',kspace=kspace*mask,sampling_mask=mask,calibration_data=kspace[...,start:start+low])
    np.save(destination/'reference.npy',reference)
    config=dict(modality='mri',algorithm='varnet',inputs={key:dict(path='public.npz',key=key) for key in ('kspace','sampling_mask')},
                model=dict(axes=['batch','coil','y','x'],spatial_shape=list(kspace.shape[-2:]),sampling='cartesian'),
                params=dict(checkpoint=str(Path(checkpoint).resolve()),num_low_frequencies=low),
                evaluation=dict(reference=dict(path='reference.npy'),data_range=float(reference.max()),mode='magnitude'))
    (destination/'varnet.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf8')
    (destination/'README.md').write_text(f'Locally authorized fastMRI acquisition {Path(source).name}, slice {index}.\nNative shape {kspace.shape}; no spatial crop, resize or intensity normalization.\nR4 column sampling plus 24-column measured ACS; reference is full-acquisition native-FOV RSS.\nThis is not the official cropped fastMRI challenge score. Data and derivatives remain local under the source agreement; do not redistribute without permission.\n',encoding='utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('--slice',type=int,default=0)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    args=parser.parse_args();prepare(args.source,args.slice,args.output,args.checkpoint)

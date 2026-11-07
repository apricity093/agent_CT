"""Generate editable external-input tasks for classical MRI and deconvolution."""
import argparse
from pathlib import Path
import numpy as np
import yaml
from imaging_inputs import generate as generate_basics


def generate(destination):
    destination=Path(destination);generate_basics(destination)
    ref=lambda name:dict(path=name+'.npy')
    base=yaml.safe_load((destination/'cg_sense.yaml').read_text())
    k=np.load(destination/'kspace.npy');np.save(destination/'acs.npy',k[:,:,4:20,4:24])
    for algorithm in ['cs_wavelet','cs_tv','grappa','espirit']:
        task=dict(base,algorithm=algorithm,params={})
        if algorithm in {'grappa','espirit'}:
            task['inputs']=dict(calibration_data=ref('acs'))
        if algorithm=='grappa':
            task['inputs'].update(kspace=ref('kspace'),sampling_mask=ref('mask'))
            truth=np.load(destination/'mri_truth.npy');maps=np.load(destination/'maps.npy')
            np.save(destination/'rss_truth.npy',abs(truth)*np.sqrt(np.sum(abs(maps)**2,axis=1)))
            task['evaluation']=dict(reference=ref('rss_truth'),data_range=1.2,mode='magnitude')
        if algorithm=='espirit':
            task['task_type']='calibration';task.pop('evaluation')
        (destination/(algorithm+'.yaml')).write_text(yaml.safe_dump(task))
    counts=np.load(destination/'counts.npy');truth=np.load(destination/'truth.npy')
    np.save(destination/'count_truth.npy',30*truth)
    for algorithm,noise in [('tv_admm','gaussian'),('tv_admm','poisson'),('rl_tv','poisson'),('hyper_laplacian','gaussian')]:
        name=algorithm+'_'+noise
        task=dict(modality='deconvolution',algorithm=algorithm,inputs=dict(observation=ref('counts' if noise=='poisson' else 'blurred'),psf=ref('psf')),
            model=dict(axes=['y','x'],boundary='periodic',psf_origin=[1,1],noise_model=noise,background=.2 if noise=='poisson' else 0),params={},
            evaluation=dict(reference=ref('count_truth' if noise=='poisson' else 'truth'),data_range=27. if noise=='poisson' else .9))
        (destination/(name+'.yaml')).write_text(yaml.safe_dump(task))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    generate(parser.parse_args().output)

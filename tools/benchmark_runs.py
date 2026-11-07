"""Single benchmark invocation, public input tasks and independent evaluation."""
import csv
import json
import os
from pathlib import Path
import time
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from inv_framework.runner import run_experiment
from inv_framework.evaluation.array_metrics import evaluate_array

ROOT=Path(__file__).resolve().parents[1]


def read_cases(domain):
    return json.loads((ROOT/'tests/data'/domain/'catalog.json').read_text())['cases']


def task(domain, case, algorithm, params=None, noise='gaussian'):
    folder=ROOT/'tests/data'/domain/case['path']
    ref=lambda key:dict(path=str(folder/'public.npz'),key=key)
    config=dict(modality=domain,algorithm=algorithm,params=params or {},device='cpu')
    if domain=='mri':
        config['model']=case['model']
        keys=['kspace','sensitivity_maps','trajectory' if case['model']['sampling']=='noncartesian' else 'sampling_mask']
        if algorithm=='espirit': keys=['calibration_data'];config['task_type']='calibration'
        if algorithm=='grappa': keys=['kspace','sampling_mask','calibration_data']
        if algorithm=='varnet': keys=['kspace','sampling_mask']
        config['inputs']={key:ref(key) for key in keys}
    else:
        config['inputs']=dict(observation=ref('counts' if noise=='poisson' else 'gaussian'))
        config['model']=dict(axes=['y','x'],boundary='periodic',noise_model=noise,background=case['background'] if noise=='poisson' else 0)
        if algorithm.startswith('blind'):
            config['task_type']='blind_deconvolution';config['model']['kernel_shape']=case['kernel_shape']
        else:
            config['inputs']['psf']=ref('psf');config['model']['psf_origin']=case['psf_origin']
    return config


class Bench:
    def __init__(self, output):
        self.output=output
        self.rows=[]

    def run(self, domain, case, algorithm, params=None, noise='gaussian', label='quality', config=None):
        config=config or task(domain,case,algorithm,params,noise)
        name=f'{label}/{domain}/{case["case_id"]}/{noise}_{algorithm}'
        dest=self.output/name
        dest.mkdir(parents=True,exist_ok=True)
        # Persist each explicit task as editable YAML, using task-relative file paths.
        def relative(value):
            if isinstance(value,dict):
                return {k:os.path.relpath(v,dest).replace('\\','/') if k in ('path','checkpoint') and isinstance(v,str) else relative(v) for k,v in value.items()}
            if isinstance(value,list): return [relative(v) for v in value]
            return value
        config_path=dest/'task.yaml'
        config_path.write_text(yaml.safe_dump(relative(config),sort_keys=False),encoding='utf8')
        started=time.perf_counter()
        result=run_experiment(config_path,dest)
        wall=time.perf_counter()-started
        row=dict(run=name,domain=domain,case=case['case_id'],split=case['split'],algorithm=algorithm,noise=noise,status=result.status,iterations=result.iterations,converged=result.converged,stop=result.stopping_reason,runtime_sec=result.runtime_sec,wall_sec=wall,device=result.diagnostics.get('device','cpu'),dtype=result.diagnostics.get('dtype','float64'),data_residual=result.diagnostics.get('data_residual'))
        if algorithm=='grappa' and result.status=='success':
            acquired=np.load(ROOT/'tests/data/mri'/case['path']/'public.npz')
            row['data_residual']=float(np.linalg.norm((result.outputs['kspace']-acquired['kspace'])*acquired['sampling_mask']))
        if 'reconstruction' in result.outputs:
            image=result.outputs['reconstruction']
            truth=np.load(ROOT/'tests/data'/domain/case['path']/'reference.npz')['object']
            gain=case['photon_gain'] if domain=='deconvolution' and noise=='poisson' else 1.
            image=image/gain
            metrics=evaluate_array(image,dict(reference=truth,data_range=1.,mode='magnitude') if domain=='mri' else dict(reference=truth,data_range=1.),spatial_axes=(-2,-1),mri=domain=='mri')
            row.update(metrics)
            error=abs(image)-abs(truth) if domain=='mri' else image-truth
            row['nmse']=float(np.sum(abs(error)**2)/np.sum(abs(truth)**2))
            fig,ax=plt.subplots(1,3,figsize=(9,3))
            for panel,data,title in zip(ax,[abs(truth).squeeze(),abs(image).squeeze() if domain=='mri' else image,abs(error).squeeze()],['reference','reconstruction','absolute error']):
                panel.imshow(data,cmap='gray',vmin=0,vmax=1 if title!='absolute error' else .3);panel.set_title(title);panel.axis('off')
            fig.tight_layout();fig.savefig(dest/'comparison.png',dpi=100);plt.close(fig)
        if 'psf' in result.outputs:
            k=result.outputs['psf'];true=np.load(ROOT/'tests/data'/domain/case['path']/'reference.npz')['true_psf']
            row.update(kernel_rmse=float(np.sqrt(np.mean((k-true)**2))),kernel_min=float(k.min()),kernel_mass=float(k.sum()),kernel_relative_error=float(np.linalg.norm(k-true)/np.linalg.norm(true)))
            fig,ax=plt.subplots(1,2,figsize=(4,2))
            for panel,data,title in zip(ax,[true,k],['true PSF','estimated PSF']):panel.imshow(data,cmap='gray');panel.set_title(title);panel.axis('off')
            fig.tight_layout();fig.savefig(dest/'kernel.png');plt.close(fig)
        for key in ('model_loading_sec','inference_sec'):
            if key in result.diagnostics:row[key]=result.diagnostics[key]
        (dest/'evaluation.json').write_text(json.dumps(row,indent=2),encoding='utf8')
        self.rows.append(row)
        print(name,result.status,row.get('psnr'),result.stopping_reason,flush=True)
        self.save()
        return result,row

    def save(self):
        self.output.mkdir(parents=True,exist_ok=True)
        (self.output/'summary.json').write_text(json.dumps(self.rows,indent=2),encoding='utf8')
        fields=list(dict.fromkeys(key for row in self.rows for key in row))
        with (self.output/'summary.csv').open('w',newline='',encoding='utf8') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(self.rows)

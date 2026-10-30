"""Render fixed-seed README image plates from real Agent outputs."""
import argparse
import csv
import json
from pathlib import Path
import sys

import torch
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_point(path):
    value=torch.load(path,map_location='cpu',weights_only=False)
    return (value if isinstance(value,torch.Tensor) else value['reconstruction']).numpy().squeeze()


def cone_case(project,work,case_dir):
    """Reconstruct the original README's exact saved 3D input."""
    from inverse_agent.core.models import (RunRequest,ProblemSpec,VariableSpec,
        ObservationSpec,ForwardModelSpec,CaseRef,ArtifactRef,DataProvenance,
        AlgorithmRequest,ComputeBudget,EvaluationRequest,MetricRequest,AccessPolicy,Partition)
    from inverse_agent.orchestration import IterisExperimentService
    manifest=json.loads((case_dir/'case.json').read_text())
    geometry=manifest['geometry']
    result_path=work/'fdk/result.json'
    if not result_path.exists():
        catalog=case_dir.parent/'catalog.json'
        catalog.write_text(json.dumps({'schema_version':'1.0','cases':[dict(
            case_id=manifest['case_id'],path=case_dir.name,dimension=3,
            geometry_type='cone_3d',noise_model='none',observation_domain='line_integral',has_ground_truth=True)]}))
        service=IterisExperimentService(ct_repo=project/'external/ct-benchlab')
        request=RunRequest(run_id='readme-fdk-original-input',
            problem=ProblemSpec(modality='ct',task_type='reconstruction',
                unknown=VariableSpec(quantity='voxel_line_attenuation',domain='volume_3d'),
                observation=ObservationSpec(domain='line_integral',axes=['detector_row','angle','detector_col']),
                forward_model=ForwardModelSpec(model_id='cone_beam',linear=True),
                geometry={**geometry,'type':'3d_cone_beam'}),
            case=CaseRef(case_id=manifest['case_id'],backend_id='ct-benchlab',partition=Partition.SMOKE,
                catalog=ArtifactRef(uri=str(catalog),role='case_catalog'),
                provenance=DataProvenance(source='unchanged original README cone HDF5 observations'),
                metadata={'dimension':3,'backend_capabilities':['cuda','astra','astra_cuda']}),
            algorithm=AlgorithmRequest(algorithm_id='fdk'),
            evaluation=EvaluationRequest(metrics=[MetricRequest(metric_id=k,parameters={'data_range':.004})
                for k in ['image.psnr','image.ssim','image.rmse']]),
            budget=ComputeBudget(device='cuda',cpu_threads=1,timeout_s=60,max_wall_seconds=50,
                                 max_forward_calls=1000,max_adjoint_calls=1000),
            output_dir=work/'fdk',access_policy=AccessPolicy(writable_roots=[work]))
        request=service.compose_ct_request(request,request.algorithm)
        result=service.execute(request)
        if result.status!='success': raise RuntimeError(result.model_dump_json())
    result=json.loads(result_path.read_text())
    with h5py.File(case_dir/'arrays.h5') as handle:
        truth=handle['truth/x'][...].squeeze()
        original=handle['measurement/y_observed'][...]
    point=load_point(result['artifacts']['reconstruction']['uri'])
    return truth,point,dict(case=manifest['case_id'],method='fdk',seed=manifest['provenance']['seed'],
        status=result['status'],convergence_status=result['convergence_status'],
        metrics={k:v['value'] for k,v in result['metrics'].items()},
        parameters={'filter_type':'ram-lak','short_scan':False,'voxel_supersampling':1},
        resource_usage=result['resource_usage'],measurement_norm=float(np.linalg.norm(original.astype(float))))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--qa-scripts',type=Path,required=True)
    parser.add_argument('--cone-source',type=Path,required=True)
    args=parser.parse_args()
    sys.path[:0]=[str(args.project/'src'),str(args.project/'external/ct-benchlab'),str(args.project/'external/iteris/src'),str(args.qa_scripts)]
    from audit_panel_alignment import require_matplotlib_panel_alignment
    args.out.mkdir(parents=True,exist_ok=True)
    args.work.mkdir(parents=True,exist_ok=True)
    rows=json.loads((args.study/'reports/quality_summary.json').read_text())
    selected=[]
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
                         'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})

    def plate(name,images,titles,nrows,ncols):
        fig,axes=plt.subplots(nrows,ncols,figsize=(ncols*3.0,nrows*3.25),squeeze=False)
        for index,(ax,image,title) in enumerate(zip(axes.flat,images,titles)):
            ax.imshow(image,cmap='gray',vmin=0,vmax=.004,interpolation='nearest')
            ax.set_axis_off()
            ax.set_title(title,fontsize=10,pad=10)
            ax.text(0,1.06,chr(97+index),transform=ax.transAxes,fontweight='bold',fontsize=10)
        fig.subplots_adjust(left=.035,right=.965,bottom=.025,top=.91,wspace=.13,hspace=.23)
        fig.canvas.draw()
        require_matplotlib_panel_alignment(fig,json_out=args.out/(name+'.alignment.json'),strict=True)
        fig.savefig(args.out/(name+'.png'),dpi=300)
        fig.savefig(args.out/(name+'.pdf'))
        fig.savefig(args.out/(name+'.svg'))
        plt.close(fig)

    def group(case,methods,labels):
        with h5py.File(args.study/'data'/case/'arrays.h5') as handle:truth=handle['truth/x'][...].squeeze()
        images=[truth];titles=['Reference']
        for method,label in zip(methods,labels):
            row=next(r for r in rows if r['case']==case and r['method']==method)
            assert row.get('reconstruction') and 'image.psnr' in row['metrics']
            images.append(load_point(row['reconstruction']))
            plt.imsave(args.out/(case.split('_')[1]+'_'+method+'.png'),images[-1],cmap='gray',vmin=0,vmax=.004)
            titles.append(label+'\n'+f"{row['metrics']['image.psnr']:.2f} dB | {row['metrics']['image.ssim']:.3f} SSIM")
            selected.append({k:row.get(k) for k in ['case','method','status','convergence_status','metrics','parameters','parameter_sources','resource_usage']})
        return images,titles

    images,titles=group('G2_gaussian_002_104729_128',
        ['fista_identity','admm_identity','pdhg_identity','fista_tv','fista_haar'],
        ['FISTA + identity','ADMM + identity','PDHG + identity','FISTA + TV','FISTA + Haar-L1'])
    plate('gaussian_components',images,titles,2,3)
    images,titles=group('G2_poisson_1000_104729_128',['pdhg_wls_tv','pdhg_poisson_tv'],
                        ['PDHG + TV | post-log WLS','PDHG + TV | pre-log NLL'])
    plate('poisson_fidelity',images,titles,1,3)
    truth,point,record=cone_case(args.project,args.work,args.cone_source)
    selected.append(record)
    slices=lambda x:[x[32,:,:],x[:,32,:],x[:,:,32]]
    for name,image in zip(['axial','coronal','sagittal'],slices(point)):
        plt.imsave(args.out/('fdk_'+name+'.png'),image,cmap='gray',vmin=0,vmax=.004)
    plate('cone_fdk',slices(truth)+slices(point),
        ['Reference | axial','Reference | coronal','Reference | sagittal',
         'FDK | axial','FDK | coronal','FDK | sagittal'],2,3)
    manifest=dict(seed_2d=104729,seed_3d=record['seed'],selection_rule='Fixed G2 intermediate Gaussian/Poisson conditions; component coverage, not best-seed or best-quality selection. 3D reuses the exact historical README input.',
        image_display=dict(window=[0,.004],gamma=1,colormap='gray',crop='none',interpolation='nearest',
            note='Identical fixed window across all panels; values outside the display window are saturated for display only; metrics use original values.'),
        case_count_2d=2,additional_3d_baselines=1,results=selected,
        limitations=['One seed; not a performance ranking.','2D returned points are budget-limited, not convergence certificates.',
            '3D FDK is a geometry-specific direct baseline, not evidence of arbitrary 3D solver/prior interchangeability.'])
    resource_fields=['runtime_s','wall_seconds','iterations_completed','forward_calls','adjoint_calls',
                     'gpu_runtime_seconds','peak_gpu_memory_mb','prox_iterations','backend_reconstruction_calls']
    for row in selected:
        row['resource_usage']={key:row.get('resource_usage',{}).get(key) for key in resource_fields}
    (args.out/'selected_results.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with (args.out/'selected_results.csv').open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['case','method','status','PSNR','SSIM','RMSE'])
        for row in selected:writer.writerow([row['case'],row['method'],row['convergence_status'],*[row['metrics'][k] for k in ['image.psnr','image.ssim','image.rmse']]])
    print((args.out/'selected_results.csv').read_text())


if __name__=='__main__': main()

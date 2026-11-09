"""Representative public configuration checks, not a quality benchmark."""
import json
from pathlib import Path
import numpy as np
import pytest
import yaml
from inv_framework.runner import run_experiment
from inv_framework.api.schema import normalize_request


def mri_task():
    rng=np.random.default_rng(7)
    return dict(modality='mri',algorithm='cg_sense',model=dict(sampling='cartesian',spatial_shape=[4,4],axes=['batch','coil','y','x']),
        inputs=dict(kspace=rng.normal(size=(1,1,4,4))+1j*rng.normal(size=(1,1,4,4)),sampling_mask=np.ones((1,1,4,4)),sensitivity_maps=np.ones((1,1,4,4))))


def deconv_task():
    return dict(modality='deconvolution',algorithm='tv_admm',model=dict(axes=['y','x'],boundary='periodic',psf_origin=[0,0],noise_model='poisson'),
        inputs=dict(observation=np.arange(16).reshape(4,4)/10,psf=np.ones((1,1))),params=dict(max_iter=8,lam=.1))


def test_poisson_and_quadratic_comparison_can_keep_the_same_constraint():
    counts=np.array([[0.,2.],[1.,3.]])
    for fidelity in ('poisson_nll','l2'):
        task=dict(modality='deconvolution',algorithm='tv_admm',fidelity=fidelity,
            inputs=dict(observation=counts if fidelity=='poisson_nll' else counts-.2,
                psf=np.ones((1,1)),initial_estimate=np.zeros((2,2))),
            model=dict(axes=['y','x'],boundary='periodic',psf_origin=[0,0],
                background=.2 if fidelity=='poisson_nll' else 0.,noise_model='poisson'),
            params=dict(lam=0.,max_iter=2000,tolerance=1e-8,nonnegative=True))
        result=run_experiment(task)
        assert result.status=='success' and result.converged, result.error
        assert result.diagnostics['nonnegative'] is True
        np.testing.assert_allclose(result.outputs['reconstruction'],np.maximum(counts-.2,0),atol=1e-6)


def test_precedence_aliases_and_actual_updates(tmp_path):
    task=mri_task()
    task['inputs']={k:dict(path=f'{k}.npy') for k in task['inputs']}
    for key,value in mri_task()['inputs'].items():np.save(tmp_path/f'{key}.npy',value)
    path=tmp_path/'task.yaml'
    task['params']=dict(lam=.2,max_iter=3,tolerance=1e-4)
    path.write_text(yaml.safe_dump(task))
    result=run_experiment(path,tmp_path/'run',params=dict(lam=.4,max_iter=1))
    assert result.status=='success',result.error
    assert result.effective_params==dict(lam=.4,max_iter=1,tolerance=1e-4)
    expected=np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(mri_task()['inputs']['kspace'],axes=(-2,-1)),axes=(-2,-1),norm='ortho'),axes=(-2,-1))[:,0]/1.4
    np.testing.assert_allclose(result.outputs['reconstruction'],expected)
    _,_,resolved=normalize_request(dict(case_id='unused',algorithm='fista',params=dict(reg_strength=.3)))
    assert resolved['strength']==.3
    from inv_framework.api import list_capabilities
    defaults=next(r['default_parameters'] for r in list_capabilities()['algorithms'] if r['name']=='fista')
    assert normalize_request(dict(case_id='unused',algorithm='fista',params=defaults))[2]['strength']==defaults['strength']
    ct=run_experiment(dict(modality='ct',algorithm='sirt',case_id='parallel_2d__ellipses_sparse_gaussian',params=dict(num_iterations=3)),params=dict(max_iter=1))
    assert ct.status=='success',ct.error
    assert ct.effective_params['num_iterations']==1
    selected=run_experiment(dict(modality='ct',algorithm='fista',case_id='parallel_2d__ellipses_sparse_gaussian',fidelity='l2',params=dict(max_iter=1)),params=dict(fidelity='gaussian_nll'))
    assert selected.status=='success',selected.error
    assert selected.effective_params['data_fidelity']=='gaussian_nll'
    assert selected.effective_params['step_size']>0

    with pytest.raises(ValueError,match='conflict'):
        normalize_request(dict(case_id='unused',algorithm='sirt',params=dict(max_iter=1,num_iterations=2)))
    (tmp_path/'evidence.json').write_text(json.dumps(dict(mri=result.effective_params,ct=ct.effective_params),indent=2))


def test_fidelity_is_an_update_and_fixed_methods_reject(tmp_path):
    task=mri_task();task['params']=dict(lam=.3,max_iter=10)
    unweighted=run_experiment(task)
    task['inputs']['data_weights']=np.arange(1,17).reshape(1,1,4,4)
    task['fidelity']='wls';weighted=run_experiment(task)
    assert weighted.status=='success',weighted.error
    assert np.linalg.norm(weighted.outputs['reconstruction']-unweighted.outputs['reconstruction'])>1e-3
    task['fidelity']='gaussian_nll';equivalent=run_experiment(task)
    np.testing.assert_allclose(weighted.outputs['reconstruction'],equivalent.outputs['reconstruction'])
    task=deconv_task();poisson=run_experiment(task)
    task['fidelity']='l2';quadratic=run_experiment(task)
    assert poisson.status==quadratic.status=='success'
    assert np.linalg.norm(poisson.outputs['reconstruction']-quadratic.outputs['reconstruction'])>1e-3
    task['fidelity']='gaussian_nll';equivalent=run_experiment(task)
    np.testing.assert_array_equal(equivalent.outputs['reconstruction'],quadratic.outputs['reconstruction'])
    task['algorithm']='richardson_lucy';task['params']={};rejected=run_experiment(task)
    assert rejected.status=='failed' and 'unsupported fidelity' in rejected.error['message']
    task=mri_task();task['fidelity']='wls';rejected=run_experiment(task)
    assert 'data_weights' in rejected.error['message']
    task=deconv_task();task['params']['lam']=1000 # valid beyond suggested range
    accepted=run_experiment(task);assert accepted.effective_params['lam']==1000
    task['params']['rho']=.2;changed=run_experiment(task)
    assert changed.effective_params['rho']==.2
    assert np.linalg.norm(changed.outputs['reconstruction']-accepted.outputs['reconstruction'])>1e-6
    task['params']['rho']=0;rejected=run_experiment(task)
    assert rejected.status=='failed' and 'rho' in rejected.error['message']
    task=deconv_task();task['algorithm']='richardson_lucy';task['params']=dict(max_iter=1)
    task['model'].update(gain=2.,background=.1,boundary='zero')
    scaled=run_experiment(task)
    assert scaled.status=='success',scaled.error
    np.testing.assert_allclose(scaled.outputs['reconstruction'], .5*task['inputs']['observation']/1.1)
    (tmp_path/'updates.json').write_text(json.dumps(dict(mri_weighted_difference=float(np.linalg.norm(weighted.outputs['reconstruction']-unweighted.outputs['reconstruction'])), deconvolution_fidelity_difference=float(np.linalg.norm(poisson.outputs['reconstruction']-quadratic.outputs['reconstruction'])),model=dict(gain=scaled.diagnostics['gain'],background=scaled.diagnostics['background'],boundary=scaled.diagnostics['boundary'])),indent=2))


def test_discovery_pages_and_defaults():
    from inv_framework.api import list_capabilities as ct
    from inv_framework.mri.api import list_capabilities as mri
    from inv_framework.deconvolution.api import list_capabilities as deconv
    root=Path(__file__).resolve().parents[1]
    for domain,discover,count in [('ct',ct,18),('mri',mri,14),('deconvolution',deconv,14)]:
        records=discover()['algorithms'];assert len(records)==count
        for record in records:
            assert (root/record['documentation']).is_file()
            assert record['fidelity']['mode'] in {'fixed','selectable','not_applicable'}
            defaults=record['default_parameters'] if domain=='ct' else record['parameters']
            for key in defaults:
                specs=record['parameter_specs'] if domain=='ct' else [dict(name=k,**v) for k,v in record['parameter_schema'].items()]
                parameter=next(p for p in specs if p['name']==key)
                assert parameter['units'] and parameter['description'] and parameter['recommended_search']

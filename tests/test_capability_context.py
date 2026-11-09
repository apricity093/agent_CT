"""Focused discovery and dedicated MRI diagnostic regressions."""
import numpy as np
import pytest
from inv_framework.mri.api import list_capabilities as mri_capabilities, run_mri_task
from inv_framework.deconvolution.api import list_capabilities as deconv_capabilities
from inv_framework.api import list_capabilities as ct_capabilities
from inv_framework.experiments import resolve_parameters


def algorithm(caps, name):
    return next(r for r in caps['algorithms'] if r['name'] == name)


def search(caps, name, param):
    record=algorithm(caps, name)
    if 'parameter_specs' in record:
        return next(p for p in record['parameter_specs'] if p['name']==param)['recommended_search']
    return record['parameter_schema'][param]['recommended_search']


def test_haar_and_calibration_candidates():
    static=search(mri_capabilities(),'cs_wavelet','levels')
    assert static['values']==[1,2,3] and static['resolution']=='pending'
    assert 'model.spatial_shape' in static['unresolved_conditions'][0]
    caps=mri_capabilities(context=dict(model=dict(spatial_shape=[4,4],sampling='cartesian'),calibration_shape=[4,4]))
    assert search(caps,'cs_wavelet','levels')['resolved_values']==[1,2]
    assert search(caps,'cs_wavelet','levels')['unresolved_conditions']==[]
    assert search(caps,'espirit','kernel_width')['resolved_values']==[] # existing fixed width is 5
    caps=mri_capabilities(context=dict(calibration_shape=[5,5]))
    assert search(caps,'grappa','kernel_width')['resolved_values']==[5]
    assert algorithm(caps,'cs_wavelet')['recommendation_context']['unresolved_conditions']==['model.sampling']
    assert search(mri_capabilities(),'espirit','kernel_width')['resolution']=='pending'
    # Candidate filtering does not change execution's legal Haar levels.
    assert resolve_parameters('mri','cs_wavelet',{'levels':4})['levels']==4


def test_ct_view_partitions_and_model_candidates():
    static=ct_capabilities()
    assert search(static,'sart','block_size')['resolution']=='pending'
    assert search(static,'fista','data_fidelity')['resolution']=='pending'
    caps=ct_capabilities(context=dict(num_views=3,model=dict(spatial_shape=[4,4],noise_model='gaussian'),available_inputs=['sigma']))
    assert search(caps,'fista','regularizer')['resolved_values']==search(static,'fista','regularizer')['values']
    assert search(caps,'fista','data_fidelity')['resolved_values']==['l2','gaussian_nll']
    assert search(caps,'constrained_pdhg','residual_metric')['resolved_values']==['raw','gaussian_whitened']
    assert search(caps,'fista','weight_source')['resolved_values']==[]
    assert search(ct_capabilities(context=dict(model=dict(spatial_shape=[3,4]))),'fista','regularizer')['resolved_values']==['none','tikhonov','h1','tv','huber','gradient_huber']
    caps=ct_capabilities(context=dict(num_views=3,params=dict(subset_count=2)))
    assert search(caps,'sart','block_size')['resolved_values']==[1]
    caps=ct_capabilities(context=dict(num_views=3,params=dict(block_size=2)))
    assert search(caps,'os_sart','subset_count')['resolved_values']==[]
    caps=ct_capabilities(context=dict(num_views=10,params=dict(block_size=2)))
    assert search(caps,'os_sart','subset_count')['resolved_values']==[]
    caps=ct_capabilities(context=dict(num_views=20,params=dict(block_size=2)))
    assert search(caps,'os_sart','subset_count')['resolved_values']==[10]
    caps=ct_capabilities(context=dict(model=dict(noise_model='poisson',sigma_e=1),available_inputs=['counts','I0','weights']))
    assert 'transmission_poisson_nll' not in search(caps,'fista','data_fidelity')['resolved_values']
    assert 'post_log_poisson_gaussian' not in search(caps,'fista','weight_source')['resolved_values']


def test_penalty_context_and_blind_minimum():
    caps=deconv_capabilities(context=dict(model=dict(boundary='zero'),params=dict(beta_max=2)))
    assert search(caps,'hyper_laplacian','beta_initial')['resolved_values']==[.1,1.]
    assert algorithm(caps,'tv_admm')['recommendation_context']['incompatible_conditions']
    for name in ('blind_dcp','blind_l0'):
        assert algorithm(caps,name)['parameter_schema']['coarse_kernel_size']['minimum']==3
        for value in (1,2):
            with pytest.raises(ValueError,match='coarse_kernel_size must be >= 3'):
                resolve_parameters('deconvolution',name,dict(coarse_kernel_size=value))
        assert resolve_parameters('deconvolution',name,dict(coarse_kernel_size=3))['coarse_kernel_size']==3


def test_calibration_diagnostics_survive():
    # Small explicit nonzero ACS, as in the existing calibration test.
    from inv_framework.mri.operators import fft
    yy,xx=np.mgrid[:6,:6]
    acs=fft(np.stack([np.ones((6,6)),np.exp(.2j*yy)])[None],2)
    result=run_mri_task(dict(algorithm='espirit',task_type='calibration',inputs=dict(calibration_data=acs),model=dict(sampling='cartesian',spatial_shape=[6,6],axes=['batch','coil','y','x'],noise_model='complex_gaussian'),params=dict(kernel_width=3)))
    assert result.status=='success',result.error
    assert result.diagnostics['generation_model']=='complex_gaussian'
    assert result.diagnostics['fidelity'] is None
    assert 'calibration_ranks' in result.diagnostics
    assert result.outputs['sensitivity_maps'].shape==(1,2,6,6)


def test_varnet_diagnostics_survive(monkeypatch):
    from inv_framework.mri import varnet_inference
    def infer(*args):
        return dict(reconstruction=np.ones((1,4,4))),dict(checkpoint='test-only')
    monkeypatch.setattr(varnet_inference,'infer',infer)
    result=run_mri_task(dict(algorithm='varnet',inputs={},model=dict(sampling='cartesian',spatial_shape=[4,4],axes=['batch','coil','y','x'])))
    assert result.status=='success',result.error
    assert result.diagnostics==dict(generation_model='unknown',fidelity=None,checkpoint='test-only')

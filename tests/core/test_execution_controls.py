"""Outer execution controls and persisted result contracts."""
import copy
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
import yaml
from inv_framework.runner import run_experiment, _resolve_paths
from inv_framework.experiments import execution_controls, planned_capabilities
from fixtures.imaging_inputs import generate


@pytest.fixture
def tasks(tmp_path):
    generate(tmp_path)
    configs = {}
    for name in ('cg_sense', 'richardson_lucy'):
        config = yaml.safe_load((tmp_path / f'{name}.yaml').read_text())
        config.pop('evaluation')
        configs[name] = _resolve_paths(config, tmp_path)
    configs['fista'] = copy.deepcopy(configs['richardson_lucy'])
    configs['fista'].update(algorithm='fista', fidelity='l2')
    configs['fista']['model'].update(gain=1, background=0, noise_model='gaussian')
    return configs


@pytest.mark.parametrize('algorithm', ['cg_sense', 'richardson_lucy', 'fista'])
@pytest.mark.parametrize('budget', [1,3,7])
def test_budget_native_equivalence(tasks, algorithm, budget):
    task = tasks[algorithm]
    native = 'num_iterations' if algorithm == 'fista' else 'max_iter'
    task['params'] = {native: budget, 'tolerance': 0.0}
    expected = run_experiment(task)
    task['params'] = {native: 20, 'tolerance': .5}
    task['stopping_policy'] = dict(max_iterations=budget, tolerance=0.0)
    actual = run_experiment(task, params={native: 12})
    assert actual.status == expected.status == 'success', actual.error
    assert actual.iterations == expected.iterations == budget
    assert actual.effective_params == expected.effective_params
    assert actual.stopping_reason == expected.stopping_reason
    assert not actual.converged
    np.testing.assert_array_equal(actual.outputs['reconstruction'], expected.outputs['reconstruction'])
    assert actual.diagnostics['parameter_overrides'][native]['previous'] == 12


@pytest.mark.parametrize('algorithm', ['cg_sense', 'richardson_lucy', 'fista'])
def test_real_early_stop_and_defaults(tasks, algorithm):
    task = tasks[algorithm]
    baseline = run_experiment(task)
    empty = run_experiment(dict(task, stopping_policy={}, problem_constraints={}))
    assert baseline.status == empty.status == 'success'
    np.testing.assert_array_equal(baseline.outputs['reconstruction'], empty.outputs['reconstruction'])
    task['stopping_policy'] = dict(max_iterations=100, tolerance=1.)
    stopped = run_experiment(task)
    assert stopped.status == 'success' and stopped.converged, stopped.error
    assert stopped.iterations < 100


@pytest.mark.parametrize('patch', [dict(stopping_policy={'unknown':1}),
    dict(problem_constraints={'min_value':0}), dict(stopping_policy={'patience':2})])
def test_reject_unknown_or_unsupported(tasks, patch):
    result = run_experiment(dict(tasks['cg_sense'], **patch))
    assert result.status == 'failed' and result.error


def test_direct_and_fixed_inference_reject_controls(tasks):
    for domain, algorithm in [('mri','zero_filled'), ('mri','varnet'), ('mri','espirit'), ('deconvolution','wiener_hunt')]:
        result = run_experiment(dict(modality=domain, algorithm=algorithm, inputs={}, model={},
            stopping_policy=dict(tolerance=.1)))
        assert result.status == 'failed'
        assert 'unsupported stopping_policy' in result.error['message']


def test_supported_box_enters_solver(tasks):
    task = tasks['fista']
    task['params'] = dict(num_iterations=7, tolerance=0., regularizer='tikhonov', strength=.1)
    native = run_experiment(task, params=dict(min_value=.2, max_value=.6))
    constrained = run_experiment(dict(task, problem_constraints=dict(min_value=.2, max_value=.6)))
    unconstrained = run_experiment(task)
    assert constrained.status == native.status == 'success', constrained.error
    np.testing.assert_array_equal(constrained.outputs['reconstruction'], native.outputs['reconstruction'])
    assert not np.allclose(constrained.outputs['reconstruction'], np.clip(unconstrained.outputs['reconstruction'], .2, .6))
    assert constrained.diagnostics['effective_problem_constraints'] == dict(min_value=.2,max_value=.6)


def test_rl_object_scale_and_precision(tasks):
    task = tasks['richardson_lucy']
    x = np.linspace(.2,2.,24*28).reshape(24,28)
    task['inputs'] = dict(observation=(5*x+2).astype(np.float32), psf=np.ones((1,1),np.float32))
    task['model'].update(gain=5, background=2, psf_origin=[0,0])
    task['stopping_policy'] = dict(max_iterations=300,tolerance=1e-12)
    result = run_experiment(task)
    assert result.status == 'success' and result.converged
    np.testing.assert_allclose(result.outputs['reconstruction'],x,atol=1e-7,rtol=2e-7)
    artifact = result.output_artifacts['reconstruction']
    assert artifact['dtype'] == 'float64' and artifact['path'] is None
    assert result.diagnostics['computation_precision'] == 'float64'
    assert 'no post-rescaling' in artifact['scale']


def test_file_paths_metadata_reference_and_missing(tasks, tmp_path, monkeypatch):
    task = tasks['cg_sense']
    # Include actual supported data_weights reference, with WLS selected.
    weights = np.linspace(.8,1.2,24*28*2).reshape(1,2,24,28)
    np.save(tmp_path/'weights.npy',weights)
    task.update(fidelity='wls',params=dict(max_iter=3))
    task['inputs']['data_weights'] = dict(path='weights.npy')
    for value in task['inputs'].values():
        if isinstance(value,dict): value['path'] = Path(value['path']).name
    file = tmp_path/'request.yaml'
    file.write_text(yaml.safe_dump(task))
    first = run_experiment(file,tmp_path/'saved')
    monkeypatch.chdir(tmp_path)
    from_mapping = run_experiment(task)
    assert from_mapping.status == 'success'
    np.testing.assert_array_equal(first.outputs['reconstruction'], from_mapping.outputs['reconstruction'])
    elsewhere = tmp_path/'elsewhere'; elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    second = run_experiment(file)
    assert first.status == second.status == 'success', first.error
    np.testing.assert_array_equal(first.outputs['reconstruction'], second.outputs['reconstruction'])
    saved = json.loads((tmp_path/'saved/result.json').read_text())
    for name, artifact in saved['output_artifacts'].items():
        loaded = np.load(tmp_path/'saved'/artifact['path'])
        assert artifact['shape'] == list(loaded.shape) and artifact['dtype'] == str(loaded.dtype)
        assert dict(artifact,path=None) == first.output_artifacts[name]
        assert saved['outputs'][name] == artifact['path']
    task['evaluation'] = dict(reference=dict(path='mri_truth.npy'), data_range=.9)
    file.write_text(yaml.safe_dump(task))
    evaluated = run_experiment(file)
    assert 'ssim' in evaluated.metrics and not first.metrics
    np.testing.assert_array_equal(first.outputs['reconstruction'],evaluated.outputs['reconstruction'])
    # Same filename in cwd cannot rescue a missing config-relative input.
    np.save(elsewhere/'weights.npy',weights)
    (tmp_path/'weights.npy').unlink()
    failed = run_experiment(file, tmp_path/'failed')
    assert failed.status == 'failed'
    assert json.loads((tmp_path/'failed/result.json').read_text())['error']


def test_failure_cli_and_io_errors(tmp_path):
    file=tmp_path/'bad.yaml'; file.write_text('modality: mri\nalgorithm: missing\n')
    out=tmp_path/'failure'
    command=subprocess.run([sys.executable,'-m','inv_framework.runner','--config',str(file),'--output','failure'],cwd=tmp_path,capture_output=True,text=True)
    assert command.returncode != 0
    saved=json.loads((out/'result.json').read_text())
    assert saved['status']=='failed' and saved['outputs']==saved['output_artifacts']=={}
    with pytest.raises(FileNotFoundError): run_experiment(tmp_path/'absent.yaml',tmp_path/'not_created')
    assert not (tmp_path/'not_created').exists()
    unwritable=tmp_path/'file';unwritable.write_text('existing file')
    with pytest.raises(FileExistsError): run_experiment(file,unwritable)


@pytest.mark.parametrize('section', ['inputs', 'model', 'params', 'evaluation'])
def test_null_sections_are_request_errors(section, tmp_path):
    task = dict(modality='mri', algorithm='cg_sense', inputs={}, model={}, params={})
    task[section] = None
    file = tmp_path / 'null.yaml'
    file.write_text(yaml.safe_dump(task))
    for name, request in [('mapping', task), ('file', file)]:
        out = tmp_path / name
        result = run_experiment(request, out)
        assert result.status == 'failed'
        assert result.error == dict(code='invalid_request', message=f'{section} must be a mapping')
        assert json.loads((out / 'result.json').read_text())['status'] == 'failed'
    command = subprocess.run([sys.executable, '-m', 'inv_framework.runner',
        '--config', str(file), '--output', str(tmp_path / 'cli')], capture_output=True, text=True)
    assert command.returncode == 1
    assert 'Traceback' not in command.stderr and 'AttributeError' not in command.stderr
    assert json.loads((tmp_path / 'cli/result.json').read_text())['error']['message'] == f'{section} must be a mapping'


def test_numerical_failure_is_not_success(tasks, monkeypatch, tmp_path):
    from inv_framework.deconvolution import solvers
    monkeypatch.setattr(solvers,'richardson_lucy',lambda *a: (np.full((24,28),np.nan),3,'numerical_error',[]))
    result=run_experiment(tasks['richardson_lucy'],tmp_path/'numeric')
    assert result.status=='failed' and not result.converged
    assert result.iterations==3 and result.stopping_reason=='numerical_error'
    assert json.loads((tmp_path/'numeric/result.json').read_text())['status']=='failed'


def test_all_declared_control_mappings_are_applied():
    from types import SimpleNamespace
    from inv_framework.experiments import resolve_execution_parameters, ExperimentResult
    for domain in ('mri','deconvolution'):
        for record in planned_capabilities(domain)['algorithms']:
            mappings=execution_controls(domain,record['name'],record['parameters'])
            for group,mapping in mappings.items():
                for field,target in mapping.items():
                    value=True if target is None or field=='nonnegative' else 3 if field=='max_iterations' else .1
                    request=SimpleNamespace(algorithm=record['name'],params={},stopping_policy={},problem_constraints={},fidelity=None,model={})
                    getattr(request,group)[field]=value
                    result=ExperimentResult('failed',domain,record['name'])
                    p=resolve_execution_parameters(domain,request,result)
                    if target: assert p[target]==value


def test_declared_paths_only(tmp_path):
    config=dict(data_root='cases',params=dict(checkpoint='weights.pt'),
        inputs=dict(geometry=dict(path='ordinary label'),measurement=dict(path='ordinary label'),
                    public_measurement=dict(weights=dict(path='w.npy'))),
        model=dict(checkpoint='ordinary label'),evaluation=dict(reference=dict(path='ref.npy')))
    resolved=_resolve_paths(config,tmp_path)
    assert resolved['data_root']==str(tmp_path/'cases')
    assert resolved['params']['checkpoint']==str(tmp_path/'weights.pt')
    assert resolved['inputs']['public_measurement']['weights']['path']==str(tmp_path/'w.npy')
    assert resolved['model']==config['model']
    assert resolved['inputs']['geometry']==config['inputs']['geometry']


@pytest.mark.parametrize('budget', [1,3,7])
def test_ct_sart_existing_budget_and_native_alias(budget):
    task=dict(modality='ct',algorithm='sart',backend='torch',device='cpu',
        case_id='parallel_2d__ellipses_sparse_gaussian',params=dict(num_iterations=budget))
    native=run_experiment(task)
    aliased=run_experiment(dict(task,params=dict(max_iter=budget)))
    assert native.status==aliased.status=='success',native.error
    assert native.iterations==aliased.iterations==budget
    assert native.effective_params==aliased.effective_params
    assert native.stopping_reason==aliased.stopping_reason
    np.testing.assert_array_equal(native.outputs['reconstruction'],aliased.outputs['reconstruction'])


def test_ct_data_root_relative_to_config(tmp_path,monkeypatch):
    import os
    root=Path(__file__).resolve().parents[2]
    config=dict(modality='ct',algorithm='fbp',backend='torch',device='cpu',
        case_id='parallel_2d__ellipses_dense_clean',data_root=os.path.relpath(root/'tests/data',tmp_path))
    path=tmp_path/'ct.yaml';path.write_text(yaml.safe_dump(config))
    a=run_experiment(path)
    other=tmp_path/'cwd';other.mkdir();monkeypatch.chdir(other)
    b=run_experiment(path)
    assert a.status==b.status=='success',a.error
    np.testing.assert_array_equal(a.outputs['reconstruction'],b.outputs['reconstruction'])
    config['data_root']='absent';path.write_text(yaml.safe_dump(config))
    assert run_experiment(path).status=='failed'


def test_outer_budget_does_not_replace_inner(tasks):
    task=copy.deepcopy(tasks['fista'])
    task['params']=dict(tv_num_iterations=17,tv_tolerance=1e-4)
    task['stopping_policy']=dict(max_iterations=1)
    result=run_experiment(task)
    assert result.status=='success',result.error
    assert result.effective_params['tv_num_iterations']==17
    assert result.effective_params['tv_tolerance']==1e-4
    from inv_framework.experiments import resolve_execution_parameters,ExperimentResult
    from inv_framework.mri.api import MRITaskRequest
    request=MRITaskRequest('cs_tv',{}, {},params=dict(cg_max_iter=19,cg_tolerance=2e-5),stopping_policy=dict(max_iterations=3))
    p=resolve_execution_parameters('mri',request,ExperimentResult('failed','mri','cs_tv'))
    assert (p['max_iter'],p['cg_max_iter'],p['cg_tolerance'])==(3,19,2e-5)


@pytest.mark.parametrize('domain', ['ct', 'mri', 'deconvolution'])
def test_minimal_cpu_cli_from_other_cwd(domain, tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    config = root / 'examples/minimal_cpu' / domain / 'request.yaml'
    baseline = run_experiment(config)
    assert baseline.status == 'success', baseline.error
    command = subprocess.run([sys.executable, '-m', 'inv_framework.runner',
        '--config', str(config), '--output', str(tmp_path / 'cli')],
        cwd=tmp_path, capture_output=True, text=True)
    assert command.returncode == 0, command.stderr
    saved = json.loads((tmp_path / 'cli/result.json').read_text())
    assert saved['status'] == 'success' and saved['diagnostics']['device'] == 'cpu'
    assert saved['effective_params'] == baseline.effective_params
    for name, artifact in saved['output_artifacts'].items():
        array = np.load(tmp_path / 'cli' / artifact['path'], allow_pickle=False)
        assert list(array.shape) == artifact['shape'] and str(array.dtype) == artifact['dtype']
        assert np.isfinite(array).all()
        np.testing.assert_array_equal(array, baseline.outputs[name])
    task = _resolve_paths(yaml.safe_load(config.read_text()), config.parent)
    task.pop('evaluation')
    monkeypatch.chdir(tmp_path)
    unreferenced = run_experiment(task)
    assert unreferenced.status == 'success'
    assert not {'rmse', 'psnr', 'ssim'} & unreferenced.metrics.keys()
    np.testing.assert_array_equal(unreferenced.outputs['reconstruction'], np.load(tmp_path / 'cli/reconstruction.npy'))

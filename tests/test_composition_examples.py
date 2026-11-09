"""File requests, candidate propagation and reference separation for the examples."""
from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from inv_framework.experiments import read_array, resolve_parameters
from inv_framework.runner import run_experiment, _resolve_paths

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = runpy.run_path(str(ROOT / 'examples/composition/requests.py'))


def test_json_runner_preserves_scientific_numeric_parameters(tmp_path):
    import json
    config = EXAMPLES['load']('deconvolution_tv_haar.yaml')
    config['params'].update(num_iterations=1, tolerance=1e-5)
    path = tmp_path / 'request.json'
    path.write_text(json.dumps(config), encoding='utf-8')
    result = run_experiment(path)
    assert result.status == 'success', result.error
    assert result.effective_params['tolerance'] == 1e-5


def test_ct_external_double_precision_reaches_certified_inner_updates():
    from inv_framework.api.schema import load_request_case
    from inv_framework.ct.api import CTTaskRequest
    import test_composition_adaptation as adaptation
    config, _, _, _ = adaptation.fixture('ct')
    config['inputs']['observation'] = config['inputs']['observation'].astype(np.float64)
    request = CTTaskRequest('external', 'admm', backend='torch', inputs=config['inputs'])
    case = load_request_case(request, 'cpu')
    assert case.measurement.dtype == case.truth.dtype == torch.float64
    config['inputs'].pop('public_measurement')
    config.update(algorithm='admm', fidelity='l2', params=dict(regularizer='tv_haar', level=1,
        strength=.01, haar_strength=.01, num_iterations=12, cg_tolerance=1e-10))
    result = run_experiment(config)
    assert result.status == 'success', result.error
    assert result.iterations == 12
    reports = result.diagnostics['convergence']['metadata']['cg_subproblems']
    assert len(reports) == 12 and all(r['converged'] for r in reports)


def test_candidates_are_complete_public_requests(tmp_path):
    from inv_framework.ct.api import CTTaskRequest
    from inv_framework.api.schema import validate_ct_request
    for domain in EXAMPLES['DOMAINS']:
        candidates = list(EXAMPLES['candidates'](domain))
        assert len(candidates) == 37
        for name, config in candidates:
            assert not {'truth', 'reference', 'object'} & config['inputs'].keys()
            for value in config['inputs'].values():
                if isinstance(value, dict) and 'path' in value:
                    assert np.isfinite(read_array(value)).all()
            assert read_array(config['evaluation']['reference']).size > 4
            if domain == 'ct':
                effective = validate_ct_request(CTTaskRequest(config['case_id'], config['algorithm'],
                    backend='torch', inputs=config['inputs'], params=config['params'], fidelity=config['fidelity']))
            else:
                effective = resolve_parameters(domain, config['algorithm'], config['params'])
            assert effective['strength'] == config['params']['strength']
            assert effective['haar_strength'] == config['params']['haar_strength']
            path = tmp_path / 'development' / (name + '.yaml')
            EXAMPLES['write'](config, path)
            restored = _resolve_paths(yaml.safe_load(path.read_text()), path.parent)
            assert restored == config


def test_quadratic_methods_keep_target_and_only_supported_parameters():
    from inv_framework.ct.registry import validate_parameter_values
    for domain in EXAMPLES['DOMAINS']:
        base = EXAMPLES['quadratic'](domain, 'cg')
        for method in EXAMPLES['METHODS']:
            config = EXAMPLES['quadratic'](domain, method)
            assert config['inputs'] == base['inputs']
            assert config['model'] == base['model']
            assert config['fidelity'] == base['fidelity'] == 'l2'
            assert all(config['params'][k] == v for k, v in base['params'].items())
            if domain == 'ct':
                assert not validate_parameter_values(method, config['params']).errors
            else:
                resolve_parameters(domain, method, config['params'])


def test_deconvolution_huber_boundaries_and_public_execution(tmp_path):
    candidates = list(EXAMPLES['candidates']('deconvolution'))
    huber = [(name, config) for name, config in candidates
             if config['params']['regularizer'] == 'gradient_huber']
    assert len(candidates) == 37 and len(huber) == 6
    for name, config in candidates:
        assert config['model']['boundary'] == 'periodic'
        expected = 'neumann' if config['params']['regularizer'] == 'gradient_huber' else 'periodic'
        assert config['params']['boundary'] == expected
    name, config = huber[2]
    # Two updates exercise real assembly/CG without a quality search.
    result = run_experiment(config, tmp_path / name, params={'num_iterations': 2})
    assert result.status == 'success', result.error
    assert result.iterations == 2
    assert result.effective_params['boundary'] == 'neumann'
    assert result.diagnostics['subproblem_path'] == 'hermitian_cg'
    image = result.outputs['reconstruction']
    assert np.isfinite(image).all() and np.linalg.norm(image) > 0


@pytest.mark.parametrize('dimension', [2, 3])
def test_ct_runner_evaluates_after_reconstruction_and_preserves_reason(monkeypatch, dimension):
    import inv_framework.ct.api as api
    from inv_framework.api.execution import _metrics
    from inv_framework.utils.metrics import ssim
    if dimension == 2:
        reference = np.linspace(0, .02, 12*16, dtype=np.float32).reshape(1, 1, 12, 16)
    else:
        reference = np.linspace(0, .02, 3*12*16, dtype=np.float32).reshape(1, 3, 12, 16)
    image = reference * .8 + .001
    bundle = dict(reconstruction=torch.as_tensor(image), truth=torch.as_tensor(reference),
        measurement=torch.ones(1, 1, 2, 2), predicted_measurement=torch.zeros(1, 1, 2, 2),
        solver='cg', case_id='fixed_arrays', dimension=dimension, data_range=.02)
    expected = _metrics(bundle)
    seen = []
    def run(request):
        assert not {'truth', 'reference', 'data_range'} & request.inputs.keys()
        seen.append(request)
        return SimpleNamespace(status='success', effective_params={}, metrics={}, runtime_sec=0.,
            iterations=2, converged=False, convergence_status='max_iterations',
            stopping_reason='inner_budget_exhausted', reconstruction=image,
            diagnostics={'ground_truth_available': False}, error_message=None)
    monkeypatch.setattr(api, 'run_ct_task', run)
    config = EXAMPLES['load']('ct_tv_haar.yaml')
    config['evaluation']['reference'] = reference
    config['inputs']['geometry']['type'] = 'cone_3d' if dimension == 3 else 'parallel_2d'
    result = run_experiment(config)
    assert len(seen) == 1
    assert result.stopping_reason == 'inner_budget_exhausted'
    for key in ('rmse', 'psnr', 'ssim', 'relative_error', 'ssim_definition'):
        assert result.metrics[key] == expected[key]
    assert result.diagnostics['ground_truth_available'] is False
    assert result.diagnostics['evaluation']['reference_provided'] is True
    assert result.diagnostics['evaluation']['stage'] == 'post_reconstruction'
    if dimension == 2:
        assert result.metrics['ssim'] == float(ssim(torch.as_tensor(image),
            torch.as_tensor(reference), data_range=.02).mean())
        assert result.metrics['ssim_definition'] == '2d_ssim'
    else:
        axial = torch.stack([ssim(torch.as_tensor(image[:, z:z+1]),
            torch.as_tensor(reference[:, z:z+1]), data_range=.02).mean() for z in range(3)]).mean()
        assert result.metrics['ssim'] == float(axial)
        assert result.metrics['ssim_definition'] == 'axial_mean_ssim'

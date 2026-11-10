"""Public discovery contracts without requiring benchmark data or optional GPUs."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from inv_framework.capabilities import list_capabilities


def algorithm(caps, name):
    return next(r for r in caps['algorithms'] if r['name'] == name)


def test_aggregate_and_domain_queries():
    from inv_framework import list_capabilities as top_level
    from inv_framework.ct.api import list_capabilities as ct
    from inv_framework.mri.api import list_capabilities as mri
    from inv_framework.deconvolution.api import list_capabilities as deconv
    all_domains = list_capabilities()
    assert set(all_domains) == {'ct','mri','deconvolution'}
    for domain, query in [('ct',ct),('mri',mri),('deconvolution',deconv)]:
        assert all_domains[domain] == query() == list_capabilities(domain)
    assert top_level() == ct()
    json.dumps(all_domains, allow_nan=False)
    with pytest.raises(ValueError, match='Unknown modality'):
        list_capabilities('unknown')


def test_case_query_is_independent(tmp_path, monkeypatch):
    from inv_framework.ct.cases import list_ct_cases
    from inv_framework.api import dispatcher
    monkeypatch.setenv('INVFRAMEWORK_CT_CASE_ROOT', str(tmp_path/'missing'))
    def forbidden(*args, **kwargs):
        raise AssertionError('Algorithm discovery read cases')
    monkeypatch.setattr(dispatcher, 'list_ct_cases', forbidden)
    assert 'cases' not in list_capabilities('ct')
    with pytest.raises(FileNotFoundError):
        list_ct_cases(data_root=tmp_path/'missing')
    with pytest.raises(FileNotFoundError):
        list_ct_cases()


def test_defaults_and_controls_match_registration():
    caps = list_capabilities()
    root = Path(__file__).resolve().parents[1]
    for domain, domain_caps in caps.items():
        for record in domain_caps['algorithms']:
            assert record['modality'] == domain
            assert record['task_types'] and record['input_specs'] and record['outputs']
            if domain == 'ct':
                defaults = record['default_parameters']
                schema = {r['name']: r for r in record['parameter_specs']}
            else:
                config = yaml.safe_load((root/'configs'/domain/'algorithms'/f"{record['name']}.yaml").read_text())
                defaults = record['parameters']
                assert defaults == config['parameters']
                schema = record['parameter_schema']
            assert set(schema) == set(defaults)
            for key, default in defaults.items():
                assert schema[key]['default'] == default
                assert schema[key]['type']
            controls = record['controls']
            native = controls['native']
            assert native['iteration_parameter'] is None or native['iteration_parameter'] in defaults
            assert set(native['threshold_parameters'] + native['inner_parameters']) <= defaults.keys()
            if domain != 'ct':
                mappings = controls['unified_entry']['field_mappings']
                assert controls['unified_entry']['stopping_policy'] == bool(mappings['stopping_policy'])
                assert controls['unified_entry']['problem_constraints'] == bool(mappings['problem_constraints'])
                assert set(v for v in mappings['stopping_policy'].values() if v) <= defaults.keys()
            assert record['runtime_status']['state'] == 'pending' or not record['dependency_installed']
            assert record['runnable'] is not True
    mri = caps['mri']
    assert algorithm(mri,'cg_sense')['controls']['native']['iteration_parameter'] == 'max_iter'
    assert algorithm(mri,'admm')['controls']['native']['iteration_parameter'] == 'num_iterations'
    for name in ('cg','cgls','lsqr','tikhonov'):
        assert algorithm(caps['ct'],name)['controls']['unified_entry']['problem_constraints'] is False
    assert algorithm(mri,'espirit')['outputs'] == ['sensitivity_maps','eigenvalues']
    assert [s['name'] for s in algorithm(mri,'espirit')['input_specs']] == ['calibration_data']
    assert {'kspace','reconstruction'} <= set(algorithm(mri,'grappa')['outputs'])
    for name in ('blind_dcp','blind_l0'):
        blind = algorithm(caps['deconvolution'], name)
        assert blind['outputs'] == ['reconstruction','psf']
        assert blind['task_types'] == ['blind_deconvolution']
        assert 'kernel_shape' in blind['model_requirements']
        assert [s['name'] for s in blind['input_specs']] == ['observation','initial_psf']


def test_missing_optional_dependencies_and_context(monkeypatch, tmp_path):
    import importlib.util
    from inv_framework.mri.api import list_capabilities as mri
    original = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, 'find_spec', lambda module: None if module in {'astra','leapctype','torchkbnufft'} else original(module))
    ct = list_capabilities('ct')
    fdk = algorithm(ct,'fdk')
    assert fdk['runtime_status']['state'] == 'unavailable'
    assert 'Missing dependency: astra' in fdk['runtime_status']['reasons']
    assert algorithm(mri(),'cg_sense')['dependency_installed']
    noncart = algorithm(mri(context=dict(model=dict(sampling='noncartesian'))),'cg_sense')
    assert 'Missing dependency: torchkbnufft' in noncart['runtime_status']['reasons']
    assert list_capabilities('deconvolution')['algorithms']
    missing = algorithm(mri(context=dict(available_inputs=[], device='cuda', params=dict(checkpoint=str(tmp_path/'absent.pt')))), 'varnet')
    assert missing['runtime_status']['state'] == 'unavailable'
    assert 'Checkpoint file does not exist' in missing['runtime_status']['reasons']
    assert 'Missing input: kspace' in missing['runtime_status']['reasons']
    incompatible = algorithm(mri(context=dict(model=dict(sampling='noncartesian', spatial_shape=[4,4,4]))),'varnet')
    assert incompatible['runtime_status']['state'] == 'unavailable'
    assert 'Unsupported spatial dimension' in incompatible['runtime_status']['reasons']


def test_single_domain_import_is_lazy():
    command = """
import sys
from inv_framework.capabilities import list_capabilities
list_capabilities('mri')
assert 'inv_framework.ct.api' not in sys.modules
assert 'inv_framework.deconvolution.api' not in sys.modules
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable,'-c',command], check=True)


def test_discovery_does_not_probe_hardware(monkeypatch):
    import torch
    def forbidden(*args, **kwargs):
        raise AssertionError('Discovery probed hardware')
    monkeypatch.setattr(torch.cuda,'is_available',forbidden)
    assert list_capabilities()['ct']['backends']['astra']['cuda_available'] is None

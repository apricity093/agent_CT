import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
import yaml
from inv_framework.ct.api import CTTaskRequest, run_ct_task
from inv_framework.api import run_ct_task as old_run
from inv_framework.ct.backends.radon_torch import ParallelBeamRadon2D
from inv_framework.runner import run_experiment


def test_external_no_truth_python_cli_and_parameter_precedence(tmp_path):
    assert run_ct_task is old_run
    angles = torch.linspace(0, 3.0, 10)
    operator = ParallelBeamRadon2D(12, angles=angles)
    image = torch.zeros(1, 1, 12, 12)
    image[:, :, 4:8, 4:8] = .02
    observed = operator.forward(image).numpy()
    np.save(tmp_path / 'observed.npy', observed)
    model = {'geometry': {'type': 'parallel_2d', 'domain_shape': [1,12,12],
        'range_shape': list(observed.shape[1:]), 'angles_rad': angles.tolist()},
        'measurement': {'noise_model': 'gaussian', 'observation_domain': 'line_integral',
                        'observation_model': 'gaussian'}}
    config = {'modality': 'ct', 'algorithm': 'sirt', 'backend': 'torch', 'device': 'cpu',
        'inputs': {'observation': {'path': 'observed.npy'}}, 'model': model,
        'params': {'num_iterations': 2}}
    path = tmp_path / 'task.yaml'
    path.write_text(yaml.safe_dump(config), encoding='utf-8')
    result = run_experiment(path, tmp_path / 'api', params={'num_iterations': 3})
    assert result.status == 'success', result.error
    assert result.effective_params['num_iterations'] == 3
    assert not {'psnr','ssim','rmse'} & result.metrics.keys()
    assert np.isfinite(result.outputs['reconstruction']).all()
    # CLI uses the same explicit config, outside the algorithm working directory.
    config['params']['num_iterations'] = 3
    path.write_text(yaml.safe_dump(config), encoding='utf-8')
    completed = subprocess.run([sys.executable, '-m', 'inv_framework.runner',
        '--config', str(path), '--output', str(tmp_path / 'cli')],
        cwd=tmp_path, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])})
    assert completed.returncode == 0, completed.stderr
    np.testing.assert_allclose(np.load(tmp_path/'cli/reconstruction.npy'), result.outputs['reconstruction'])
    payload = json.loads((tmp_path/'cli/result.json').read_text())
    assert payload['effective_params'] == result.effective_params
    # Arrays need no file or catalog registration either.
    direct = run_ct_task(CTTaskRequest('unregistered', 'sirt',
        inputs={'observation': observed, **model}, params={'max_iter': 3}))
    assert direct.status == 'success', direct.error_message
    np.testing.assert_allclose(direct.reconstruction, result.outputs['reconstruction'])

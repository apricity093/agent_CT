import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
import yaml
from inv_framework.runner import run_experiment


@pytest.mark.parametrize('algorithm',['zero_filled','cg_sense','noncartesian_sense','wiener_hunt','richardson_lucy'])
def test_external_files_evaluation_and_cli(tmp_path,algorithm):
    tool=Path(__file__).resolve().parent/'fixtures/imaging_inputs.py'
    subprocess.run([sys.executable,str(tool),'--output',str(tmp_path)],check=True)
    path=tmp_path/f'{algorithm}.yaml'
    evaluated=run_experiment(path,tmp_path/'evaluated')
    assert evaluated.status=='success',evaluated.error
    assert {'psnr','ssim','rmse'} <= evaluated.metrics.keys()
    task=yaml.safe_load(path.read_text())
    task.pop('evaluation')
    path.write_text(yaml.safe_dump(task))
    command=subprocess.run([sys.executable,'-m','inv_framework.runner','--config',str(path),'--output',str(tmp_path/'cli')],capture_output=True,text=True)
    assert command.returncode==0,command.stderr
    saved=json.loads((tmp_path/'cli/result.json').read_text())
    assert saved['metrics']=={}
    assert saved['effective_params']==evaluated.effective_params
    np.testing.assert_allclose(np.load(tmp_path/'cli/reconstruction.npy'),evaluated.outputs['reconstruction'])

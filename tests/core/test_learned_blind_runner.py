import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch
import yaml
from inv_framework.mri.models.varnet import VarNet
from inv_framework.runner import run_experiment


@pytest.mark.parametrize('method',['varnet','blind_dcp','blind_l0'])
def test_external_api_cli_without_truth(tmp_path,method):
    tool=Path(__file__).resolve().parent/'fixtures/learned_blind_inputs.py'
    subprocess.run([sys.executable,str(tool),'--output',str(tmp_path)],check=True)
    path=tmp_path/(method+'.yaml');task=yaml.safe_load(path.read_text())
    task.pop('evaluation',None)
    if method=='varnet':
        # Random weights validate serialization and execution only, never quality.
        p=dict(num_cascades=1,chans=2,pools=2,sens_chans=2,sens_pools=2)
        torch.save(VarNet(**p).state_dict(),tmp_path/'structure_only.pt')
        task['params'].update(p,checkpoint='structure_only.pt')
    else:
        task['params']=dict(max_iter=1,final_max_iter=3)
        if method=='blind_dcp':task['params'].update(patch_size=3,latent_iter=2)
    path.write_text(yaml.safe_dump(task))
    result=run_experiment(path,tmp_path/'api');assert result.status=='success',result.error
    run=subprocess.run([sys.executable,'-m','inv_framework.runner','--config',str(path),'--output',str(tmp_path/'cli')],cwd=tmp_path,capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    saved=json.loads((tmp_path/'cli/result.json').read_text())
    assert saved['metrics']=={}
    for name,value in result.outputs.items():
        np.testing.assert_allclose(np.load(tmp_path/'cli'/saved['outputs'][name]),value)

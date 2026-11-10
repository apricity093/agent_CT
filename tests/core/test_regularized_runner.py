import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import yaml
from inv_framework.runner import run_experiment


def test_external_classical_tasks_and_cli(tmp_path):
    tool=Path(__file__).resolve().parent/'fixtures/regularized_inputs.py'
    subprocess.run([sys.executable,str(tool),'--output',str(tmp_path)],check=True)
    names=['grappa','espirit','cs_wavelet','cs_tv','tv_admm_gaussian','tv_admm_poisson','rl_tv_poisson','hyper_laplacian_gaussian']
    for name in names:
        path=tmp_path/(name+'.yaml')
        result=run_experiment(path,tmp_path/(name+'_evaluated'))
        assert result.status=='success',(name,result.error)
        if name=='espirit':
            assert result.outputs['sensitivity_maps'].shape==(1,2,24,28)
        else:
            assert {'psnr','ssim','rmse'}<=result.metrics.keys()
        task=yaml.safe_load(path.read_text());task.pop('evaluation',None)
        path.write_text(yaml.safe_dump(task))
        output=tmp_path/(name+'_no_truth')
        command=subprocess.run([sys.executable,'-m','inv_framework.runner','--config',str(path),'--output',str(output)],capture_output=True,text=True)
        assert command.returncode==0,(name,command.stderr,command.stdout)
        saved=json.loads((output/'result.json').read_text())
        assert saved['metrics']=={}
        assert set(saved['output_artifacts']) == set(result.outputs)
        for output_name, artifact in saved['output_artifacts'].items():
            array=np.load(output/artifact['path'])
            assert artifact['shape']==list(array.shape) and artifact['dtype']==str(array.dtype)
            assert dict(artifact,path=None)==result.output_artifacts[output_name]
        for key,value in result.outputs.items():
            np.testing.assert_allclose(np.load(output/(key+'.npy')),value)

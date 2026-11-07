"""Compare adapted and downloaded upstream networks with one explicit checkpoint.

Pass a directory containing official varnet.py and unet.py. Only those network
files are loaded; the FFT/complex shim uses our independently checked primitives.
No reference package, extra environment or network access is required.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import types
import numpy as np
import torch
from inv_framework.mri.models import complex_math
from inv_framework.mri.models.varnet import VarNet
from inv_framework.mri.varnet_inference import load_checkpoint


def compare(source,checkpoint,kspace,mask,low):
    fast=types.ModuleType('fastmri');fast.__dict__.update(vars(complex_math))
    data=types.ModuleType('fastmri.data');data.transforms=complex_math
    sys.modules['fastmri']=fast;sys.modules['fastmri.data']=data
    package=types.ModuleType('reference_models');package.__path__=[str(source)]
    sys.modules['reference_models']=package
    spec=importlib.util.spec_from_file_location('reference_models.varnet',source/'varnet.py')
    reference=importlib.util.module_from_spec(spec);sys.modules[spec.name]=reference;spec.loader.exec_module(reference)
    local=VarNet().eval(); upstream=reference.VarNet().eval()
    load_checkpoint(local,checkpoint,'cpu');load_checkpoint(upstream,checkpoint,'cpu')
    y=torch.view_as_real(torch.as_tensor(np.load(kspace).astype(np.complex64)))
    sampled=torch.as_tensor(np.load(mask)[:,:1,:1,:,None].astype(bool))
    with torch.inference_mode():
        actual=local(y,sampled,low);expected=upstream(y,sampled,low)
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
    return dict(maximum_absolute_difference=float((actual-expected).abs().max()),
                state_tensors=len(local.state_dict()),shape=list(actual.shape),
                checkpoint=str(checkpoint),quality_validation=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','checkpoint','kspace','mask','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--low',type=int,required=True)
    args=parser.parse_args();result=compare(args.source,args.checkpoint,args.kspace,args.mask,args.low)
    args.output.write_text(json.dumps(result,indent=2));print(result)

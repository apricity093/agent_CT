import numpy as np
import pytest
import torch
from scipy.optimize import brentq
from inv_framework.fidelity import TransmissionPoissonFidelity

CUDA_DEVICE = pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))

@pytest.mark.parametrize('device',['cpu', CUDA_DEVICE])
@pytest.mark.parametrize('I0',[.5,20.5,100000.])
@pytest.mark.parametrize('scale',[.3,1.,2.])
@pytest.mark.parametrize('step',[.001,.3,10.])
def test_scalar_grid(device,I0,scale,step):
    counts=np.array([0.,1.,7.,100.]);value=np.array([-3.,0.,2.,4.])
    f=TransmissionPoissonFidelity(torch.tensor(counts,device=device),I0,scale=scale)
    result=f.proximal(torch.tensor(value,device=device),step).cpu().numpy()
    reference=[]
    for c,v in zip(counts,value):
        def equation(x):
            with np.errstate(over='ignore'):
                return x-v+step*scale*(c-I0*np.exp(-scale*x))
        root=brentq(equation,min(0.,v-step*scale*c)-1,max(0.,v+step*scale*I0)+1,xtol=1e-13,maxiter=1000)
        reference.append(root)
        assert abs(equation(result[len(reference)-1]))<1e-9
    assert np.max(abs(result-np.array(reference)))<1e-9

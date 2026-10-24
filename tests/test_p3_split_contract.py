import numpy as np
import pytest
import torch
from test_p1_composable import Matrix
from inv_framework.solvers import ADMMSolver, PDHGSolver

class QuadraticCoefficients:
    def __init__(self, transform):
        self.transform = transform
    def split_transform(self, shape):
        return self.transform, 4.
    def coefficient_value(self, z):
        return .5*z.square()
    def coefficient_proximal(self, z, step):
        return z/(1+step)

CUDA_DEVICE = pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))

@pytest.mark.parametrize('solver', [ADMMSolver, PDHGSolver])
@pytest.mark.parametrize('device', ['cpu', CUDA_DEVICE])
def test_replace_split_module(solver, device):
    a=np.diag([1.1,1.2,1.3,1.4])
    t=np.array([[1.,-.2,0.,0.],[0.,1.,.3,0.],[0.,0.,1.,-.1],[.2,0.,0.,1.]])
    y=np.array([-1.,.4,-.3,1.2]); weight=.17
    reference=np.linalg.solve(a.T@a+weight*t.T@t,a.T@y)
    tensor=lambda x: torch.tensor(x,dtype=torch.float64,device=device)
    reg=QuadraticCoefficients(Matrix(tensor(t),(1,2,2)))
    costs=[]
    for _ in range(2):
        instance=solver(reg,strength=weight,num_iterations=2000,tolerance=1e-8)
        result=instance.solve_detailed(tensor(y)[None],Matrix(tensor(a),(1,2,2)))
        x=result.reconstruction.flatten().cpu().numpy()
        assert result.status=='converged'
        assert np.linalg.norm(x-reference)<1e-6
        assert np.linalg.norm(a.T@(a@x-y)+weight*t.T@t@x)<1e-6
        assert result.resources['transform_kind']=='linear'
        costs.append(result.resources['coefficient_prox_calls'])
    assert costs[0]==costs[1]

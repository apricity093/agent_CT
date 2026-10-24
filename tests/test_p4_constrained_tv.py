import numpy as np
import pytest
import torch
from inv_framework.residual_ball import ResidualBall
from inv_framework.operators.ct.radon_torch import ParallelBeamRadon2D
from inv_framework.solvers.constrained_tv import ConstrainedTVSolver


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('radius', [0., .7, 100.])
def test_ball_independent(device, radius):
    y = torch.tensor([[[[.2, -.1, .3, 2.]]]], dtype=torch.float64, device=device)
    mask = torch.tensor([[[[True, False, True, True]]]], device=device)
    v = torch.tensor([[[[2., 9., -1., .4]]]], dtype=y.dtype, device=device)
    ball = ResidualBall(y, radius, source='explicit test', mask=mask)
    yy, vv, mm = [t.cpu().numpy() for t in (y, v, mask)]
    delta = (vv-yy)[mm]
    reference = vv.copy()
    reference[mm] = yy[mm]+delta*min(1., radius/np.linalg.norm(delta))
    np.testing.assert_allclose(ball.projection(v).cpu(), reference, rtol=0, atol=1e-12)
    step = .37
    q = (vv-step*yy)[mm]
    expected = np.zeros_like(vv)
    expected[mm] = q*max(0., 1.-step*radius/np.linalg.norm(q))
    np.testing.assert_allclose(ball.dual_prox(v, step).cpu(), expected, rtol=0, atol=1e-12)
    torch.testing.assert_close(ball.dual_prox(v, step), v-step*ball.projection(v/step), rtol=0, atol=1e-12)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_ct_constant_optimum(device):
    op = ParallelBeamRadon2D(8, angles=torch.arange(7, device=device)*np.pi/7, device=device)
    truth = torch.full((1,1,8,8), .1, device=device)
    y = op.forward(truth)
    result = ConstrainedTVSolver(epsilon=.1, epsilon_source='explicit', num_iterations=5000,
                                 tolerance=1e-5).solve_detailed(y, op)
    print(device, result.status, result.actual_iterations, result.metadata, flush=True)
    assert result.status == 'converged'
    assert result.metadata['constraint_diagnostics']['constraint_violation'] <= 1e-5
    # A constant feasible image has TV=0, a global lower-bound certificate.
    assert result.final_objective <= 1e-4


def test_invalid_ball():
    y = torch.ones(1,1,2,2)
    for radius in [-1., float('nan'), float('inf')]:
        with pytest.raises(ValueError):
            ResidualBall(y, radius, source='explicit')
    with pytest.raises(ValueError):
        ResidualBall(y, 1., source='explicit', mask=torch.ones(2,2))


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('limited', [False, True])
def test_asd_adaptation(device, limited):
    angles = torch.arange(7, device=device)*(2. if limited else np.pi)/7
    op = ParallelBeamRadon2D(8, angles=angles, device=device)
    truth = torch.zeros(1,1,8,8,device=device)
    truth[...,2:6,2:6] = .3
    result = ConstrainedTVSolver(epsilon=.2, epsilon_source='explicit test', algorithm='asd_pocs_tv',
                                 num_iterations=8, tv_updates=3).solve_detailed(op.forward(truth), op)
    history = result.metadata['adaptation_history']
    assert len(history) == 8
    assert history[-1]['relaxation'] < history[0]['relaxation']
    assert sum(row['reductions'] for row in history) > 0
    assert sum(row['accepted_tv_updates'] for row in history) > 0
    for row in history:
        assert row['tv_change'] <= .95*row['data_change']+1e-6
    assert result.metadata['variant_id'] == 'sart2d_feasibility_backtracking_v1'
    assert result.metadata['nonnegative']
    assert result.status == 'max_iterations'

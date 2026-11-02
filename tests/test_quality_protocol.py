"""P5 protocol seams: data isolation, coefficient convention and hard caps."""
import time
from types import SimpleNamespace
import torch
import pytest
from examples.quality_comparison import make_case, parameters, evaluate, jobs
from inv_framework.ct.backends.radon_torch import ParallelBeamRadon2D
from inv_framework.api.execution import build_solver
from inv_framework.solvers.algebraic import SARTSolver
from inv_framework.solvers.base import SolveControl


SETTINGS = dict(sigma=.05, I0={'C4':10000., 'C5':20., 'C6':20.}, sigma_e=3.)


def test_independent_pilot_and_shared_formal_data():
    pilot, _, pt, _, _ = make_case('C2', True, SETTINGS)
    formal, _, ft, _, _ = make_case('C2', False, SETTINGS)
    repeated, _, _, _, _ = make_case('C2', False, SETTINGS)
    assert not torch.equal(pt,ft)
    assert pilot.metadata['measurement']['seed'] != formal.metadata['measurement']['seed']
    assert torch.equal(formal.measurement,repeated.measurement)
    assert not hasattr(formal,'truth')


def test_mean_loss_coefficient_reaches_solver_without_changing_legacy_strength():
    settings = dict(alpha_reg=.01/1536,budget={'iterations':2000})
    job = jobs()[0]
    case, _, _, _, _ = make_case('C2',False,SETTINGS)
    params = parameters(job,settings)
    solver = build_solver('pdhg',params,case)
    assert solver.parameters['reg_strength'] == pytest.approx(.01)
    assert 1024*settings['alpha_reg'] == pytest.approx(.01*2/3)




def test_mixed_raw_readout_and_observation_only_weights():
    case, _, _, _, _ = make_case('C6',True,SETTINGS)
    raw=case.public_measurement['c_obs']
    assert (raw<0).any() and 'counts' not in case.public_measurement
    expected=raw.clamp_min(1).square()/(raw.clamp_min(1)+9)
    torch.testing.assert_close(case.public_measurement['weights'],expected)


def test_fixed_range_metrics_do_not_rescale_output():
    truth=torch.ones(1,1,32,32)*.5
    case=SimpleNamespace(measurement=torch.ones(1,1,2,32),public_measurement={})
    a=evaluate(truth+.1,truth,case.measurement,case)
    b=evaluate(truth+.2,truth,case.measurement,case)
    assert a['rmse']==pytest.approx(.1)
    assert a['psnr']-b['psnr']==pytest.approx(6.0206,abs=1e-4)
    assert b['ssim']<a['ssim']

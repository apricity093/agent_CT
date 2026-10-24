"""P5 protocol seams: data isolation, coefficient convention and hard caps."""
import time
from types import SimpleNamespace
import torch
import pytest
from examples.p5_quality import TimedOperator, make_case, parameters, evaluate, jobs
from inv_framework.instrumentation import OperatorBudgetExceeded
from inv_framework.operators.ct.radon_torch import ParallelBeamRadon2D
from inv_framework.ct_runtime import build_solver
from inv_framework.solvers.subset import SARTSolver
from inv_framework.solvers.base import SolveControl
from inv_framework.instrumentation import CountingLinearOperator


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
    manifest = dict(alpha_reg=.01/1536,budget={'iterations':2000})
    job = jobs()[0]
    case, _, _, _, _ = make_case('C2',False,SETTINGS)
    params = parameters(job,manifest)
    solver = build_solver('pdhg',params,case)
    assert solver.parameters['reg_strength'] == pytest.approx(.01)
    assert 1024*manifest['alpha_reg'] == pytest.approx(.01*2/3)


def test_hard_call_cap_reserves_endpoint_and_wall_cap():
    op = ParallelBeamRadon2D(4,angles=torch.tensor([0.]))
    counted=TimedOperator(op,dict(forward=2,adjoint=1,wall_seconds=600))
    x=torch.ones(1,1,4,4)
    counted.forward(x)
    with pytest.raises(OperatorBudgetExceeded): counted.forward(x)
    counted.max_forward_calls=2
    counted.forward(x)
    assert counted.stats()['forward_calls']==2
    counted.deadline=time.perf_counter()-1
    with pytest.raises(OperatorBudgetExceeded,match='wall'): counted.adjoint(torch.ones(1,1,1,4))
    assert counted.stats()['adjoint_calls']==0


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


@pytest.mark.parametrize('cap',[0,1,8,11])
def test_sart_budget_returns_last_complete_sweep(cap):
    op=ParallelBeamRadon2D(4,angles=torch.arange(4)*torch.pi/4)
    truth=torch.zeros(1,1,4,4);truth[...,1:3,1:3]=.4
    y=op.forward(truth)
    counted=CountingLinearOperator(op,max_forward_calls=cap,max_adjoint_calls=100)
    solver=SARTSolver(num_iterations=5,block_size=1,relaxation=.1)
    result=solver.solve_detailed(y,counted,control=SolveControl(max_iterations=5,stop_on_convergence=False))
    assert result.status=='resource_exhausted'
    assert result.final_objective is None and result.predicted_measurement is None
    assert result.actual_iterations==(1 if cap==11 else 0)
    reference=SARTSolver(num_iterations=1,block_size=1,relaxation=.1).solve_detailed(y,op)
    expected=reference.reconstruction if cap==11 else torch.zeros_like(truth)
    torch.testing.assert_close(result.reconstruction,expected)
    assert counted.stats()['forward_calls']==cap


def test_sart_clean_full_block_endpoint():
    from examples.p5_quality import phantom

    op=ParallelBeamRadon2D(32,angles=torch.arange(180)*torch.pi/180)
    y=op.forward(phantom(False))
    counted=TimedOperator(op,dict(forward=300,adjoint=300,wall_seconds=600))
    result=SARTSolver(num_iterations=1,block_size=180,relaxation=.1).solve_detailed(
        y,counted,control=SolveControl(max_iterations=1,stop_on_convergence=False))
    assert result.actual_iterations==1 and result.status=='max_iterations'
    assert result.reconstruction.shape==(1,1,32,32)
    assert torch.isfinite(result.reconstruction).all()
    counted.max_forward_calls=300
    prediction=counted.forward(result.reconstruction)
    torch.testing.assert_close(prediction,result.predicted_measurement)
    assert counted.stats()['forward_calls']==4
    assert counted.stats()['adjoint_calls']==2

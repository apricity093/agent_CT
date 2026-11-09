"""Public proximal budgets and discrepancy stopping regressions."""
import json
import numpy as np
import pytest
import torch
from inv_framework.core.operator import LinearOperator
from inv_framework.solvers.base import SolveControl
from inv_framework.solvers.composition import CompositeSolver
from inv_framework.runner import run_experiment
import test_composition_adaptation as adaptation


class Identity(LinearOperator):
    domain_shape = range_shape = (1, 2, 2)
    def forward(self, x): return x
    def adjoint(self, x): return x
    def squared_norm_upper_bound(self, x): return 1.


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_public_ct_cholesky_keeps_the_cg_target_and_updates(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    config,_,_,_=adaptation.fixture('ct')
    config['inputs'].pop('public_measurement')
    config['inputs']['observation']=config['inputs']['observation'].astype(np.float64)
    config.update(algorithm='admm',fidelity='l2',device=device,
        params=dict(regularizer='tv_haar',level=1,strength=.01,haar_strength=.01,
            num_iterations=12,cg_tolerance=1e-10,cg_max_iterations=300))
    cg=run_experiment(config)
    direct=run_experiment(config,params=dict(subproblem_solver='cholesky'))
    assert cg.status==direct.status=='success',direct.error
    assert cg.iterations==direct.iterations==12
    np.testing.assert_allclose(direct.outputs['reconstruction'],cg.outputs['reconstruction'],atol=1e-10,rtol=1e-10)
    meta=direct.diagnostics['convergence']['metadata']
    assert meta['subproblem_path']=='dense_cholesky'
    assert all(r['converged'] and r['residual']<=r['target'] for r in meta['cg_subproblems'])


@pytest.mark.parametrize('limit',[1,500])
def test_default_fista_keeps_native_status_with_sampled_generic_trajectory(monkeypatch,limit):
    from dataclasses import replace
    from inv_framework.api import execution
    from inv_framework.solvers.convergence import ConvergenceStatus
    original=execution.post_run_validation
    def sampled(*args,**kwargs):
        report=original(*args,**kwargs)
        return replace(report,status=ConvergenceStatus.NUMERICAL_ERROR,
            stopping_reason='trajectory_ended_before_convergence',failure_reason='trajectory_ended_before_convergence')
    monkeypatch.setattr(execution,'post_run_validation',sampled)
    config,_,_,_=adaptation.fixture('ct')
    config['inputs']['observation']=config['inputs']['observation'].astype(np.float64)
    config.update(algorithm='fista',params=dict(regularizer='tikhonov',strength=.1,
        num_iterations=limit,tolerance=1e-5,weight_source='provided'))
    result=run_experiment(config)
    assert result.status=='success',result.error
    assert result.converged==(limit==500)
    assert result.diagnostics['convergence']['endpoint_validation']['iterations']==result.iterations
    if limit==1:assert result.stopping_reason=='maximum_iterations_reached'
    else:
        confirmation=result.diagnostics['endpoint_confirmation']
        assert confirmation['passed'] and confirmation['status']=='native_endpoint_confirmed'


@pytest.mark.parametrize('domain', ['ct', 'deconvolution'])
def test_public_haar_proximal_controls(domain, monkeypatch, tmp_path):
    import inv_framework.core.regularizer as module
    original = module.joint_prox
    calls = []
    def capture(reg, value, weight, **kwargs):
        entry = dict(tolerance=kwargs['tolerance'], budget=kwargs['max_iterations'],
                     expected_target=kwargs['tolerance']*max(1., float(value.norm()))/kwargs['iteration']**2.1)
        calls.append(entry)
        try:
            return original(reg, value, weight, **kwargs)
        except module.ProximalAccuracyError as error:
            entry['reason'] = str(error)
            raise
        finally:
            entry.update(target=reg.last_prox_target, iterations=reg.last_prox_iterations)
    monkeypatch.setattr(module, 'joint_prox', capture)
    if domain == 'deconvolution':
        config = dict(modality=domain, inputs=dict(observation=np.ones((2,2)), psf=np.ones((1,1))),
                      model=dict(axes=['y','x'], boundary='periodic', psf_origin=[0,0]))
    else:
        config, _, _, _ = adaptation.fixture('ct')
        config['inputs'].pop('public_measurement')
        config.pop('fidelity')
    config.update(algorithm='fista', params=dict(regularizer='haar_l1', level=1, strength=.2,
        step_size=1., num_iterations=1, min_value=0., max_value=2., tv_tolerance=1e-12, tv_num_iterations=1))
    result = run_experiment(config, tmp_path/'boxed')
    assert result.status == 'success', result.error
    assert calls and all(c['budget']==1 and c['tolerance']==1e-12 and c['iterations']<=1 for c in calls)
    assert all(c['target']==pytest.approx(c['expected_target'], rel=1e-12, abs=0.) for c in calls)
    boxed_calls=list(calls)
    if domain=='deconvolution':
        assert calls[0]['target']==pytest.approx(2e-12, rel=1e-12, abs=0.)
        assert calls[0]['iterations']==1
        assert result.stopping_reason=='joint_prox_inner_max_iterations' and not result.converged
        config['inputs']['observation']=np.array([[0.,1.],[2.,4.]])
        calls.clear()
        limited=run_experiment(config,tmp_path/'budget')
        assert calls[0]['iterations']==1 and limited.stopping_reason=='maximum_iterations_reached'
        config['inputs']['observation']=np.ones((2,2))
        config['params'].update(min_value=None,max_value=None)
        calls.clear()
        closed=run_experiment(config,tmp_path/'closed')
        assert closed.status=='success' and not calls
        np.testing.assert_allclose(closed.outputs['reconstruction'],np.ones((2,2)),atol=1e-14)
    else:
        assert calls[0]['reason']=='joint_prox_returned_precision'
        assert result.stopping_reason=='joint_prox_returned_precision' and not result.converged
        assert result.diagnostics['convergence']['status']=='stalled'
    print(domain, json.dumps(dict(calls=boxed_calls, reason=result.stopping_reason)))


def test_admm_discrepancy_on_returned_point(tmp_path):
    y=torch.ones(1,1,2,2,dtype=torch.float64)
    solver=CompositeSolver('admm', regularizer='tikhonov',strength=.1,num_iterations=80,tolerance=1e-5)
    blocked=solver.solve_detailed(y,Identity(),control=SolveControl(max_iterations=80,tolerance=1e-5,patience=1,discrepancy_target=.01))
    normal=solver.solve_detailed(y,Identity(),control=SolveControl(max_iterations=80,tolerance=1e-5,patience=1))
    relative=float((blocked.reconstruction-y).norm()/y.norm())
    assert blocked.status=='max_iterations' and blocked.actual_iterations==80
    assert relative>.01 and relative==pytest.approx(1/11,abs=1e-5)
    assert all(not row.criteria['discrepancy'] for row in blocked.trajectory)
    assert blocked.trajectory[-1].criteria==dict(primal=True,dual=True,discrepancy=False)
    assert normal.status=='converged' and normal.actual_iterations<80
    evidence=dict(relative_residual=relative,target=.01,status=blocked.status,iterations=blocked.actual_iterations,
                  reason=blocked.stopping_reason,criteria=blocked.trajectory[-1].criteria,
                  normal_iterations=normal.actual_iterations,normal_status=normal.status)
    (tmp_path/'discrepancy.json').write_text(json.dumps(evidence,indent=2))
    print(json.dumps(evidence))


def test_public_ct_discrepancy_reaches_monitor(monkeypatch,tmp_path):
    config,_,_,_=adaptation.fixture('ct')
    config['inputs']['public_measurement']={'counts':np.full_like(config['inputs']['observation'],1e8)}
    config.update(fidelity='l2', algorithm='admm', params=dict(regularizer='tikhonov',strength=.1,num_iterations=80,tolerance=1e-5,cg_tolerance=.2),
        stopping_policy=dict(patience=1,min_iterations=1,check_every=1,stalled=dict(enabled=False),divergence=dict(enabled=False),discrepancy=dict(enabled=True,tau=1.)))
    seen={}
    original=CompositeSolver.solve_detailed
    def capture(self,measurement,operator,*args,**kwargs):
        seen['target']=kwargs['control'].discrepancy_target
        result=original(self,measurement,operator,*args,**kwargs)
        seen['native_status']=result.status
        seen['criteria']=dict(result.trajectory[-1].criteria)
        return result
    monkeypatch.setattr(CompositeSolver,'solve_detailed',capture)
    result=run_experiment(config,tmp_path/'public_ct')
    assert result.status=='success',result.error
    assert seen['target']>0 and seen['criteria']['discrepancy'] is False
    y=config['inputs']['observation']
    from inv_framework.ct.backends import ParallelBeamRadon2D
    op=ParallelBeamRadon2D(2,angles=torch.tensor(config['inputs']['geometry']['angles_rad']))
    relative=float((op.forward(torch.tensor(result.outputs['reconstruction']))-torch.tensor(y)).norm()/torch.tensor(y).norm())
    assert relative>seen['target'] and not result.converged and seen['native_status']!='converged'
    print(json.dumps(dict(seen,relative_residual=relative,iterations=result.iterations,reason=result.stopping_reason)))


@pytest.mark.parametrize('domain,prior',[(d,p) for d in ('ct','mri','deconvolution') for p in ('tikhonov','tv','haar_l1','h1','gradient_huber','tv_haar')])
def test_affected_admm_adaptation(domain,prior,monkeypatch,tmp_path):
    monkeypatch.setattr(adaptation,'ROOT',tmp_path)
    adaptation.test_public_adaptation(domain,'admm',prior)

"""Cooperative wall interruption shares the existing last-complete-point contract."""
import pytest
import torch
from inv_framework.operators.ct.radon_torch import ParallelBeamRadon2D
from inv_framework.instrumentation import CountingLinearOperator
from inv_framework.solvers import ConstrainedTVSolver, SolveControl


@pytest.mark.parametrize('algorithm', ['sirt', 'cgls', 'fista'])
def test_iteration_budget_does_not_extend_configured_run(tmp_path, algorithm):
    from pathlib import Path
    from inv_framework.ct_runtime import run_case

    root = Path(__file__).resolve().parents[1]
    result = run_case(algorithm, 'parallel_2d/disk_analytic_32',
        root / f'configs/algorithms/{algorithm}.yaml', tmp_path / 'run',
        data_root=root / 'test/data', parameter_overrides={'num_iterations': 1},
        max_iterations=4)
    assert result['status'] == 'success'
    assert result['convergence_status'] == 'max_iterations'
    assert result['diagnostics']['iterations_completed'] == 1
    assert result['diagnostics']['convergence']['endpoint_validation']['max_iterations'] == 1


@pytest.mark.parametrize('algorithm',['pdhg','asd_pocs_tv'])
def test_wall_returns_last_complete_point(algorithm):
    op=ParallelBeamRadon2D(8,angles=torch.arange(7)*torch.pi/7)
    truth=torch.zeros(1,1,8,8); truth[...,2:6,2:6]=.3
    y=op.forward(truth)
    solver=ConstrainedTVSolver(epsilon=.2,epsilon_source='explicit fixture',algorithm=algorithm,num_iterations=3)
    one=solver.solve_detailed(y,op,control=SolveControl(max_iterations=1,stop_on_convergence=False))
    counted=CountingLinearOperator(op)
    def expire(record):
        counted.counters.wall_deadline=0.
        return True
    interrupted=solver.solve_detailed(y,counted,control=SolveControl(max_iterations=3,stop_on_convergence=False),callback=expire)
    assert interrupted.status=='resource_exhausted'
    assert interrupted.stopping_reason=='wall_time_budget_exhausted'
    assert interrupted.actual_iterations==1
    assert torch.equal(one.reconstruction,interrupted.reconstruction)
    assert interrupted.final_objective is None and interrupted.final_residual is None
    assert interrupted.metadata['endpoint_diagnostics_complete'] is False
    assert interrupted.metadata['constraint_diagnostics'] is None
    assert not interrupted.metadata['patience_window']


@pytest.mark.parametrize('confirm', [True, False, 'corrupt_trajectory'])
def test_early_native_stop_requires_independent_endpoint_confirmation(tmp_path, monkeypatch, confirm):
    from pathlib import Path
    import inv_framework.ct_runtime as runtime
    root = Path(__file__).resolve().parents[1]
    original = runtime.confirm_endpoint
    def endpoint(**kwargs):
        if confirm == 'corrupt_trajectory':
            kwargs['trajectory'][-1]['native_criterion_value'] += 1.
        result = original(**kwargs)
        if not confirm:
            result = {**result, 'passed': False, 'reasons': ['fixture_endpoint_mismatch']}
        return result
    monkeypatch.setattr(runtime, 'confirm_endpoint', endpoint)
    result = runtime.run_case('sirt', 'parallel_2d/disk_analytic_32',
        root / 'configs/algorithms/sirt.yaml', tmp_path / 'run', data_root=root / 'test/data',
        parameter_overrides={'num_iterations': 100},
        stopping_policy={'min_iterations': 2, 'check_every': 1, 'patience': 2, 'stop_on_convergence': True,
            'relative_iterate_tolerance': .01, 'normalized_normal_residual_tolerance': 1e-4,
            'relative_objective_tolerance': 1e-5, 'prox_gradient_mapping_tolerance': 1e-4,
            'discrepancy': {'enabled': False}})
    if confirm is True:
        assert result['status'] == 'success'
        assert result['convergence_status'] == 'converged'
        assert result['diagnostics']['iterations_completed'] < 100
        assert result['diagnostics']['convergence']['endpoint_confirmation']['passed']
    else:
        assert result['convergence_status'] == 'numerical_error'


@pytest.mark.parametrize('algorithm',['constrained_pdhg','asd_pocs_tv'])
@pytest.mark.parametrize('phase',['parameter_estimation','endpoint_confirmation'])
def test_runtime_wall_includes_preparation_and_endpoint(tmp_path,monkeypatch,algorithm,phase):
    import json
    from pathlib import Path
    from dataclasses import replace
    from inv_framework.benchmarks.ct_cases import load_ct_case,write_ct_case
    from inv_framework.instrumentation import ResourceAccounting
    from inv_framework.ct_runtime import run_case
    root=Path(__file__).resolve().parents[1]
    op=ParallelBeamRadon2D(8,angles=torch.arange(7)*torch.pi/7)
    truth=torch.zeros(1,1,8,8); truth[...,2:6,2:6]=.3
    template=load_ct_case('parallel_2d/disk_analytic_32',data_root=root/'test/data')
    case=replace(template,case_id='wall',truth=truth,measurement=op.forward(truth),
        measurement_clean=op.forward(truth),roi_mask=None,valid_measurement_mask=None,
        geometry={**template.geometry,'domain_shape':list(op.domain_shape),'range_shape':list(op.range_shape),
                  'angles_rad':op.angles.tolist()})
    record=write_ct_case(case,tmp_path/'case'); record['path']='case'
    (tmp_path/'catalog.json').write_text(json.dumps({'schema_version':'1.0','cases':[record]}))
    start=ResourceAccounting.start_phase
    def expire(self,name):
        start(self,name)
        if name==phase:self.counters.wall_deadline=0.
    monkeypatch.setattr(ResourceAccounting,'start_phase',expire)
    result=run_case(algorithm,'wall',root/f'configs/algorithms/{algorithm}.yaml',tmp_path/'run',data_root=tmp_path,
        parameter_overrides={'num_iterations':1},max_wall_seconds=100.)
    assert result['status']=='resource_exhausted'
    assert result['stopping_reason']=='wall_time_budget_exhausted'
    d=result['diagnostics']
    assert d['resources']['budget']['wall_time_budget_exhausted']
    assert d['convergence'].get('final_objective') is None
    if phase=='endpoint_confirmation':assert (tmp_path/'run/partial_reconstruction.pt').is_file()


@pytest.mark.parametrize('algorithm', ['admm', 'pdhg', 'fista'])
def test_composable_wall_reason_keeps_last_complete_point(algorithm):
    from inv_framework.solvers import ADMMSolver, PDHGSolver, FISTASolver
    op = ParallelBeamRadon2D(8, angles=torch.arange(7) * torch.pi / 7)
    truth = torch.zeros(1, 1, 8, 8); truth[..., 2:6, 2:6] = .3
    y = op.forward(truth)
    cls = {'admm': ADMMSolver, 'pdhg': PDHGSolver, 'fista': FISTASolver}[algorithm]
    solver = cls(num_iterations=3)
    one = solver.solve_detailed(y, op, control=SolveControl(max_iterations=1, stop_on_convergence=False))
    counted = CountingLinearOperator(op)
    def expire(record):
        counted.counters.wall_deadline = 0.
        return True
    interrupted = solver.solve_detailed(y, counted,
        control=SolveControl(max_iterations=3, stop_on_convergence=False, callback=expire))
    assert interrupted.status == 'resource_exhausted'
    assert interrupted.stopping_reason == 'wall_time_budget_exhausted'
    assert interrupted.actual_iterations == 1
    assert torch.equal(one.reconstruction, interrupted.reconstruction)
    if algorithm == 'fista':
        assert interrupted.metadata['endpoint_mapping_complete']
        assert interrupted.final_objective == one.final_objective
        assert interrupted.final_residual == one.final_residual
    else:
        assert interrupted.final_objective is None and interrupted.final_residual is None

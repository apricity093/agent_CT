"""Independent CG numerical completion and task stopping-policy synchronization."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from inv_framework.deconvolution.api import DeconvolutionTaskRequest, run_deconvolution_task
from inv_framework.core.imaging_operators import KnownConvolution
from inv_framework.solvers.base import SolveControl
from inv_framework.solvers.quadratic import QuadraticSolver


ROOT = Path(__file__).resolve().parents[1] / 'outputs/cg_stopping_sync'


@pytest.mark.parametrize('exact_initial', [False, True])
def test_public_initial_solution(exact_initial):
    y = np.ones((2, 2)) if exact_initial else np.zeros((2, 2))
    inputs = dict(observation=y, psf=np.ones((1, 1)))
    if exact_initial:
        inputs['initial_estimate'] = np.full((2, 2), .5)
    result = run_deconvolution_task(DeconvolutionTaskRequest(
        'cg', inputs, dict(axes=['y', 'x'], boundary='periodic', psf_origin=[0, 0]),
        params=dict(regularizer='tikhonov', strength=1., num_iterations=20, tolerance=1e-8),
        fidelity='l2'))
    assert result.status == 'success' and result.converged
    assert result.stopping_reason == 'normalized_normal_residual'
    assert result.iterations == 0
    np.testing.assert_array_equal(result.outputs['reconstruction'], y/2)
    assert result.diagnostics['normalized_normal_residual'] == 0
    assert result.diagnostics['cg_numerical']['converged']
    rows = result.diagnostics['history']
    assert len(rows) == 1 and rows[0]['iteration'] == 0
    assert rows[0]['algorithm_residual'] == 0 and rows[0]['stopping_candidate']
    evidence = dict(converged=result.converged, iterations=result.iterations,
                    stopping_reason=result.stopping_reason,
                    normalized_normal_residual=result.diagnostics['normalized_normal_residual'],
                    data_residual=result.diagnostics['data_residual'], history=rows)
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / ('exact_initial.json' if exact_initial else 'zero_observation.json')).write_text(
        json.dumps(evidence, indent=2), encoding='utf-8')
    print(evidence)


def identity_problem():
    op = KnownConvolution((2, 2), torch.ones(1, 1, dtype=torch.float64), [0, 0], 'periodic')
    solver = QuadraticSolver('cg', regularizer='tikhonov', strength=1.,
                             num_iterations=20, tolerance=1e-8)
    return solver, op, torch.ones(1, 2, 2, dtype=torch.float64)


def test_numerical_completion_does_not_manufacture_patience():
    solver, op, y = identity_problem()
    result = solver.solve_detailed(y, op, control=SolveControl(max_iterations=20, patience=2))
    assert result.status == 'stalled'
    assert result.stopping_reason == 'cg_numerical_target_reached_policy_unmet'
    assert result.actual_iterations == 1
    assert result.metadata['normalized_normal_residual'] == 0
    assert result.metadata['cg_numerical']['converged']
    torch.testing.assert_close(result.reconstruction, y/2, atol=0, rtol=0)
    assert len(result.trajectory) == 1
    row = result.trajectory[0]
    assert row.iteration == 1 and row.consecutive_criteria_count == 1
    assert not row.stopping_candidate
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / 'patience.json').write_text(json.dumps(result.to_dict(), indent=2), encoding='utf-8')
    print(result.status, result.stopping_reason, result.actual_iterations,
          result.metadata['normalized_normal_residual'])


def test_normal_nonzero_initial_and_budget():
    from test_composition_adaptation import fixture
    from inv_framework.runner import run_experiment
    config, a, y, q = fixture('deconvolution')
    config['inputs']['initial_estimate'] = np.full((2, 2), .1)
    config.update(algorithm='cg', params=dict(regularizer='tikhonov', strength=.08,
                                            num_iterations=50, tolerance=1e-9))
    result = run_experiment(config, ROOT / 'ordinary')
    assert result.converged and result.iterations > 0
    reference = np.linalg.solve(a.T @ (q[:, None]*a) + .08*np.eye(4), a.T @ (q*y))
    np.testing.assert_allclose(result.outputs['reconstruction'].ravel(), reference, atol=1e-8)
    limited = run_experiment(config, ROOT / 'budget', params=dict(num_iterations=1))
    assert limited.status == 'success' and not limited.converged
    assert limited.iterations == 1 and limited.stopping_reason == 'maximum_iterations_reached'
    assert limited.diagnostics['normalized_normal_residual'] > 1e-9


def test_callback_cancellation_wins_over_numerical_completion():
    solver, op, y = identity_problem()
    rows = []
    def cancel(row):
        rows.append(row)
        return False
    result = solver.solve_detailed(y, op, callback=cancel)
    assert result.status == 'cancelled' and result.stopping_reason == 'callback_cancelled'
    assert result.actual_iterations == 1 and len(rows) == 1
    assert result.metadata['normalized_normal_residual'] == 0
    assert result.metadata['cg_numerical']['converged']

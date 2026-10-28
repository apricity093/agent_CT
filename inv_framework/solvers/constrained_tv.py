"""Two-dimensional nonnegative residual-ball TV reconstruction.

PDHG consumes the existing split TV contract. ASD uses normalized angle-block
SART and normalized TV subgradients with feasibility-preserving backtracking.
"""
import math
from dataclasses import replace
import torch
from ..residual_ball import ResidualBall
from ..regularizers.tv import TVRegularizer
from ..instrumentation import OperatorBudgetExceeded
from .base import InverseProblemSolver, IterationRecorder, ConsecutiveStoppingMonitor, resolve_control
from .splitting import Penalty, norm_bound
from ._utils import (prepare_initial_image, validate_measurement_shape,
                     make_angle_subsets, make_subset_operator, select_measurement_subset)
from .subset import _safe_reciprocal


class ConstrainedTVSolver(InverseProblemSolver):
    """Explicit-radius constrained TV solver; no automatic radius."""

    def __init__(self, *, epsilon, epsilon_source, algorithm='pdhg', tv_mode='isotropic',
                 num_iterations=2000, tolerance=1e-5, feasibility_tolerance=1e-5,
                 relaxation=1., relaxation_decay=.99, tv_step=.2,
                 tv_step_decay=.95, tv_updates=20, update_ratio=.95, block_size=1,
                 residual_metric='raw', whitening=None):
        if algorithm not in ('pdhg', 'asd_pocs_tv'):
            raise ValueError('unsupported constrained TV algorithm')
        for name, value in dict(tolerance=tolerance, feasibility_tolerance=feasibility_tolerance,
                                relaxation=relaxation, tv_step=tv_step, update_ratio=update_ratio).items():
            if not math.isfinite(float(value)) or value <= 0:
                raise ValueError(name+' must be positive finite')
        for value in (relaxation_decay, tv_step_decay):
            if not math.isfinite(float(value)) or not 0 < value < 1:
                raise ValueError('decay must lie in (0,1)')
        for value in (num_iterations, tv_updates, block_size):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError('iteration and subset sizes must be positive integers')
        if isinstance(epsilon, bool) or not math.isfinite(float(epsilon)) or epsilon < 0:
            raise ValueError('epsilon must be nonnegative finite')
        if not isinstance(epsilon_source, str) or not epsilon_source.strip():
            raise ValueError('epsilon source must be explicit')
        if relaxation > 1:
            raise ValueError('relaxation must be <= 1')
        self.parameters = dict(epsilon=epsilon, epsilon_source=epsilon_source,
            algorithm=algorithm, tv_mode=tv_mode, num_iterations=num_iterations,
            tolerance=tolerance, feasibility_tolerance=feasibility_tolerance,
            relaxation=relaxation, relaxation_decay=relaxation_decay,
            tv_step=tv_step, tv_step_decay=tv_step_decay, tv_updates=tv_updates,
            update_ratio=update_ratio, block_size=block_size)
        self.regularizer = TVRegularizer(tv_mode)
        if residual_metric not in ('raw','gaussian_whitened'):
            raise ValueError('unknown residual metric')
        self.parameters['residual_metric'] = residual_metric
        self.whitening = whitening

    def solve(self, measurement, operator, x_init=None, **kwargs):
        return self.solve_detailed(measurement, operator, x_init=x_init, **kwargs).reconstruction

    def solve_detailed(self, measurement, operator, x_init=None, *, mask=None, control=None, callback=None):
        cfg = self.parameters
        if cfg['residual_metric'] == 'gaussian_whitened':
            from ..operators.weighted import WeightedLinearOperator
            if self.whitening is None:
                raise ValueError('gaussian_whitened residual requires known sigma')
            operator = WeightedLinearOperator(operator, self.whitening.square())
            measurement = operator.whiten(measurement)
        validate_measurement_shape(measurement, operator, cfg['algorithm'])
        if len(operator.domain_shape) != 3 or operator.domain_shape[0] != 1 or measurement.shape[0] != 1:
            raise ValueError('constrained TV requires a single two-dimensional one-channel case')
        ball = ResidualBall(measurement, cfg['epsilon'], source=cfg['epsilon_source'], mask=mask)
        ball.accounting_operator = operator
        control = resolve_control(control, default_iterations=cfg['num_iterations'],
                                  default_tolerance=cfg['tolerance'], callback=callback)
        if control.discrepancy_target is not None:
            raise ValueError('constrained TV uses epsilon; disable discrepancy stopping')
        x = prepare_initial_image(measurement, operator, x_init=x_init, initial_value=0.).clamp_min(0)
        recorder = IterationRecorder(control, measurement, operator, algorithm=cfg['algorithm'])
        recorder.set_initial(x)
        penalty = Penalty(self.regularizer, operator.domain_shape)
        penalty.accounting_operator = operator
        status, reason = 'max_iterations', 'maximum_iterations_reached'
        completed = 0
        endpoint = False
        prediction = tv = diagnostic = metric = None
        history = []
        projection_calls = 1
        monitor = ConsecutiveStoppingMonitor(control)
        tolerance = float(control.tolerance)
        beta, step = cfg['relaxation'], None
        try:
            if not bool(torch.isfinite(x).all()):
                raise FloatingPointError('non_finite_initialization')
            if cfg['algorithm'] == 'pdhg':
                bound = norm_bound(operator, x)+penalty.bound
                if not math.isfinite(bound) or bound <= 0:
                    raise FloatingPointError('non_finite_operator_norm_bound')
                tau = sigma = .99/math.sqrt(bound)
                p = torch.zeros_like(measurement)
                d = torch.zeros_like(penalty.apply(x))
                bar = x.clone()
            else:
                subsets = []
                for indices in make_angle_subsets(measurement.shape[-2], cfg['block_size'], device=x.device):
                    sub = make_subset_operator(operator, indices)
                    active = select_measurement_subset(ball.mask, indices)
                    row = _safe_reciprocal(sub.forward(torch.ones_like(x)), 1e-8)*active
                    column = _safe_reciprocal(sub.adjoint(active.to(x.dtype)), 1e-8)
                    if not bool(torch.isfinite(row).all() and torch.isfinite(column).all()):
                        raise FloatingPointError('non_finite_sart_normalization')
                    subsets.append((sub, select_measurement_subset(measurement, indices), row, column))
            for iteration in range(1, min(cfg['num_iterations'], control.max_iterations)+1):
                endpoint = False
                old = x
                if cfg['algorithm'] == 'pdhg':
                    np = ball.dual_prox(p+sigma*operator.forward(bar), sigma)
                    nd = penalty.dual_prox(d+sigma*penalty.apply(bar), sigma, 1.)
                    candidate = (x-tau*(operator.adjoint(np)+penalty.apply(nd, True))).clamp_min(0)
                    projection_calls += 1
                else:
                    candidate = x.clone()
                    for sub, y, row, column in subsets:
                        candidate = (candidate-beta*column*sub.adjoint(row*(sub.forward(candidate)-y))).clamp_min(0)
                        projection_calls += 1
                    data_change = float((candidate-x).norm())
                    data_residual = float(ball.residual(operator.forward(candidate)).norm())
                    if not math.isfinite(data_change+data_residual):
                        raise FloatingPointError('non_finite_data_update')
                    if step is None:
                        # A feasible nonconstant initial point still needs a TV descent direction.
                        step = cfg['tv_step']*max(data_change, float(candidate.norm())*cfg['tolerance'])
                    start_tv = candidate
                    shrinks = accepted = 0
                    for _ in range(cfg['tv_updates']):
                        z = penalty.apply(candidate)
                        subgradient = self.regularizer.coefficient_subgradient(z)
                        gradient = penalty.apply(subgradient, True)
                        length = float(gradient.norm())
                        if not math.isfinite(length):
                            raise FloatingPointError('non_finite_tv_gradient')
                        if length == 0 or step == 0:
                            break
                        trial_step = step
                        for _ in range(40):
                            trial = (candidate-trial_step*gradient/length).clamp_min(0)
                            projection_calls += 1
                            trial_tv = float(penalty.value(penalty.apply(trial)))
                            trial_residual = float(ball.residual(operator.forward(trial)).norm())
                            if not math.isfinite(trial_tv+trial_residual):
                                raise FloatingPointError('non_finite_tv_trial')
                            ratio_ok = float((trial-start_tv).norm()) <= cfg['update_ratio']*max(data_change, cfg['tolerance']*float(start_tv.norm()))
                            feasible_ok = (data_residual > cfg['epsilon']+cfg['feasibility_tolerance'] or
                                           trial_residual <= cfg['epsilon']+cfg['feasibility_tolerance'])
                            if ratio_ok and feasible_ok and trial_tv <= float(penalty.value(z)):
                                candidate = trial
                                accepted += 1
                                break
                            trial_step *= cfg['tv_step_decay']
                            shrinks += 1
                        else:
                            break
                        step = trial_step
                    history.append(dict(sweep=iteration, relaxation=beta, tv_step=step,
                                        data_change=data_change, tv_change=float((candidate-start_tv).norm()),
                                        reductions=shrinks, accepted_tv_updates=accepted))
                    beta *= cfg['relaxation_decay']
                if not bool(torch.isfinite(candidate).all()):
                    raise FloatingPointError('non_finite_candidate')
                # Commit only a complete outer update. Diagnostics may still exhaust the budget.
                if cfg['algorithm'] == 'pdhg':
                    bar, p, d = 2*candidate-x, np, nd
                x = candidate
                completed = iteration
                prediction = operator.forward(x)
                tx = penalty.apply(x)
                tv = float(penalty.value(tx))
                diagnostic = ball.diagnostics(prediction)
                feasible = diagnostic['constraint_violation'] <= cfg['feasibility_tolerance']
                if cfg['algorithm'] == 'pdhg':
                    rx = (x-(x-tau*(operator.adjoint(p)+penalty.apply(d, True))).clamp_min(0))/tau
                    projection_calls += 1
                    rp = (p-ball.dual_prox(p+sigma*prediction, sigma))/sigma
                    rd = (d-penalty.dual_prox(d+sigma*tx, sigma, 1.))/sigma
                    metric = float((rx.square().sum()+rp.square().sum()+rd.square().sum()).sqrt())/max(1.,float(x.norm()),float(p.norm()),float(d.norm()))
                    optimal = metric <= tolerance
                else:
                    metric = float((x-old).norm())/max(1.,float(old.norm()))
                    # Heuristic stationarity is not a global optimality certificate.
                    optimal = metric <= tolerance and beta > .005
                if not math.isfinite(tv+metric+diagnostic['raw_residual_norm']):
                    raise FloatingPointError('non_finite_diagnostics')
                endpoint = True
                decision = monitor.observe(iteration, criteria={'feasible': feasible, 'stationary': optimal},
                    relative_change=float((x-old).norm())/max(1., float(old.norm())), monitor_value=metric)
                if not recorder.record(iteration, x, residual=ball.residual(prediction), objective=tv,
                                       algorithm_residual=metric, criteria=decision.criteria if decision.checked else {},
                                       stopping_candidate=decision.checked,
                                       consecutive_criteria_count=decision.consecutive,
                                       native_criterion_name='primal_dual_fixed_point' if cfg['algorithm']=='pdhg' else 'relative_iterate_change',
                                       native_criterion_value=metric, native_criterion_threshold=tolerance,
                                       metadata={'checked': decision.checked}):
                    status, reason = 'cancelled', 'callback_cancelled'
                    break
                if decision.converged and control.stop_on_convergence:
                    status, reason = ('converged', 'primal_dual_fixed_point') if cfg['algorithm']=='pdhg' else ('stalled', 'feasible_iterate_stagnation')
                    break
        except OperatorBudgetExceeded as error:
            status, reason = 'resource_exhausted', 'wall_time_budget_exhausted' if error.kind == 'wall' else 'operator_call_budget_exhausted'
        except FloatingPointError as error:
            status, reason = 'numerical_error', str(error)
        result = recorder.finish(x, actual_iterations=completed, status=status, stopping_reason=reason,
            final_objective=tv if endpoint else None,
            final_residual=diagnostic['raw_residual_norm'] if endpoint else None,
            predicted_measurement=prediction if endpoint else None,
            resources={**penalty.calls, 'nonnegative_projection_calls':projection_calls,
                       'ball_prox_calls':ball.prox_calls, 'ball_prox_seconds':ball.prox_seconds,
                       'mask_calls':ball.mask_calls},
            metadata=dict(objective_type='residual_ball_constrained_tv', variant_id='sart2d_feasibility_backtracking_v1' if cfg['algorithm']=='asd_pocs_tv' else 'residual_ball_pdhg_v1',
                          endpoint_diagnostics_complete=endpoint, parameters={**cfg, 'tolerance': tolerance}, adaptation_history=history,
                          constraint_diagnostics=diagnostic if endpoint else None,
                          feasibility_status=('feasible' if diagnostic['constraint_violation']<=cfg['feasibility_tolerance'] else 'violated') if endpoint else 'unconfirmed',
                          optimization_residual=metric if endpoint else None,
                          residual_metric=cfg['residual_metric'],
                          nonnegative=bool((x>=0).all()), native_termination_managed=True))

        if not endpoint:
            # Earlier trajectory checkpoints remain valid history, but cannot
            # serve as terminal evidence for an unconfirmed returned point.
            result = replace(result, relative_iterate_change=None,
                metadata={**result.metadata, 'terminal_record': None, 'patience_window': []})
        return result


def constrained_pdhg(measurement, operator, x_init=None, *, mask=None, **parameters):
    """Return nonnegative residual-ball TV reconstruction using PDHG."""
    return ConstrainedTVSolver(algorithm='pdhg', **parameters).solve(
        measurement, operator, x_init=x_init, mask=mask)


def asd_pocs_tv(measurement, operator, x_init=None, *, mask=None, **parameters):
    """Return the SART2D feasibility-backtracking TV variant reconstruction."""
    return ConstrainedTVSolver(algorithm='asd_pocs_tv', **parameters).solve(
        measurement, operator, x_init=x_init, mask=mask)

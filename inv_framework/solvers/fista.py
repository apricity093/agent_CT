"""Composite FISTA with checked quadratic/Poisson majorization and joint prox."""

import math

import torch

from ..instrumentation import OperatorBudgetExceeded
from ..regularizers import TikhonovRegularizer
from .base import InverseProblemSolver, IterationRecorder, ConsecutiveStoppingMonitor, resolve_control
from ._utils import prepare_initial_image, require_linear_operator, validate_measurement_shape, apply_box_constraints


def proximal(regularizer, value, weight, operator, min_value=None, max_value=None,
             *, iteration=1, certified=False):
    from .detailed import _proximal_with_accounting
    if weight == 0:
        return apply_box_constraints(value, min_value=min_value, max_value=max_value)
    if certified and (type(regularizer).__name__ == 'TVRegularizer' or
                      type(regularizer).__name__ == 'HaarL1Regularizer' and (min_value is not None or max_value is not None)):
        from ..regularizers.joint_prox import joint_prox
        import time
        started = time.perf_counter()
        try:
            return joint_prox(regularizer, value, weight, min_value=min_value,
                max_value=max_value, iteration=iteration,
                tolerance=getattr(regularizer, 'tolerance', 1e-5),
                max_iterations=getattr(regularizer, 'num_iterations', 1000),
                check_budget=getattr(operator, 'check_wall_budget', None))
        finally:
            elapsed = time.perf_counter()-started
            regularizer.prox_iterations = getattr(regularizer, 'prox_iterations', 0)+regularizer.last_prox_iterations
            regularizer.prox_calls = getattr(regularizer, 'prox_calls', 0)+1
            regularizer.prox_seconds = getattr(regularizer, 'prox_seconds', 0.)+elapsed
            record = getattr(operator, 'record_prox', None)
            if callable(record): record(iterations=regularizer.last_prox_iterations, elapsed_seconds=elapsed)
    result = _proximal_with_accounting(regularizer, value, weight, operator)
    return apply_box_constraints(result, min_value=min_value, max_value=max_value)


def fista_update(operator, y, momentum, gradient, residual, step, regularizer, strength,
                 min_value=None, max_value=None, work=None, fidelity=None,
                 smooth_regularizer=False, iteration=1, certified=False):
    """Check f(z) <= f(v)+<grad f(v),z-v>+||z-v||²/(2t), per batch.

    The smooth term includes data fidelity and any H1/Huber penalty. Generic
    TV/Haar denoising uses a separate gap certificate; the legacy TV entry
    retains its finite FGP convention. No exact-prox rate is claimed.
    """
    rejected = 0
    for _ in range(60):
        if work is not None:
            work["backtracking_trials"] += 1
        next_x = proximal(regularizer, momentum - step * gradient,
                           0. if smooth_regularizer else step * strength,
                           operator, min_value, max_value, iteration=iteration, certified=certified)
        next_residual = operator.forward(next_x) - y
        delta = next_x - momentum
        axes = tuple(range(1, y.ndim))
        f_next = fidelity.value_per_batch(next_residual+y) if fidelity is not None else 0.5 * next_residual.square().sum(dim=axes)
        f_current = fidelity.value_per_batch(residual+y) if fidelity is not None else 0.5 * residual.square().sum(dim=axes)
        if smooth_regularizer and strength:
            f_next = f_next+strength*regularizer.value(next_x)
            f_current = f_current+strength*regularizer.value(momentum)
        flat = lambda x: x.reshape(x.shape[0], -1).sum(1)
        bound = f_current + flat(gradient * delta) + flat(delta.square()) / (2 * step)
        if not bool(torch.isfinite(next_x).all() and torch.isfinite(f_next).all() and torch.isfinite(bound).all()):
            raise FloatingPointError("non_finite_fista_update")
        slack = 32 * torch.finfo(y.dtype).eps * torch.maximum(f_current.abs(), bound.abs()).clamp_min(1)
        if bool(torch.all(f_next <= bound + slack)):
            return next_x, next_residual, step, rejected
        rejected += 1
        if work is not None:
            work["backtracking_rejections"] += 1
        step *= 0.5
    raise FloatingPointError("smooth_majorizer_backtracking_failed")


def solve_fista_detailed(operator, measurement, *, regularizer=None, strength=None,
                         reg_strength=None, num_iterations=50, step_size=None, x_init=None,
                         tolerance=1e-5, power_iterations=12, min_value=None, max_value=None,
                         control=None, callback=None, legacy=False, fidelity=None):
    from .detailed import (_common_parameter_errors, _parameter_failure_result, _finite_nonnegative,
                           _finite_positive, _numerical_result, _policy_active, _policy_status,
                           _control_threshold)
    from .regularized import _estimate_lipschitz

    from ..fidelity import TransmissionPoissonFidelity
    from ..regularizers import ZeroRegularizer
    from ..regularizers.joint_prox import ProximalAccuracyError
    poisson = isinstance(fidelity, TransmissionPoissonFidelity)
    algorithm = "tv_fista" if legacy else "fista"
    require_linear_operator(operator, algorithm)
    validate_measurement_shape(measurement, operator, algorithm)
    reg = regularizer if regularizer is not None else TikhonovRegularizer()
    errors = _common_parameter_errors(num_iterations, min_value, max_value)
    if strength is not None and reg_strength is not None and strength != reg_strength:
        errors.append("strength and reg_strength conflict")
    weight = strength if strength is not None else reg_strength if reg_strength is not None else (1e-3 if legacy else 1e-2)
    if weight == 0: reg = ZeroRegularizer()
    smooth_regularizer = (type(reg).__name__ == 'GradientHuberRegularizer' or
                          isinstance(reg, TikhonovRegularizer) and reg.operator is not None)
    if not _finite_nonnegative(weight):
        errors.append("strength must be finite and nonnegative")
    if not _finite_nonnegative(tolerance):
        errors.append("tolerance must be finite and nonnegative")
    if step_size is not None and not _finite_positive(step_size):
        errors.append("step_size must be finite and positive")
    if isinstance(power_iterations, bool) or not isinstance(power_iterations, int) or power_iterations <= 0:
        errors.append("power_iterations must be a positive integer")
    if not smooth_regularizer and (not getattr(reg, 'has_full_proximal', False) or not callable(getattr(reg,'proximal',None))):
        errors.append('FISTA requires a full proximal; generalized Tikhonov/H1 is unsupported')
    if not callable(getattr(reg,'value',None)):
        errors.append('unsupported FISTA regularizer')
    try:
        reg.validate_shape(operator.domain_shape)
    except ValueError as error:
        errors.append(str(error))
    if errors:
        return _parameter_failure_result(algorithm, measurement, operator, x_init=x_init,
            errors=errors, max_iterations=num_iterations if isinstance(num_iterations, int) else None, parameters={})
    control = resolve_control(control, default_iterations=num_iterations, default_tolerance=tolerance, callback=callback)
    limit = min(num_iterations, control.max_iterations)
    x = prepare_initial_image(measurement, operator, x_init=x_init, initial_value=0.0)
    x = apply_box_constraints(x, min_value=min_value, max_value=max_value)
    if not bool(torch.isfinite(measurement).all() and torch.isfinite(x).all()):
        return _numerical_result(algorithm, measurement, operator, x_init=x_init, max_iterations=limit,
                                 reason="non_finite_input", parameters={})
    recorder = IterationRecorder(control, measurement, operator, algorithm=algorithm)
    recorder.set_initial(x)
    monitor = ConsecutiveStoppingMonitor(control)
    momentum, acceleration = x.clone(), 1.0
    actual = 0
    work = {"backtracking_rejections": 0, "backtracking_trials": 0}
    status, reason = "max_iterations", "maximum_iterations_reached"
    prediction = residual = objective = mapping = None
    endpoint_complete = False
    budget_interruption = None
    estimate = None
    previous_objective = None
    plateau_run = 0
    norm = lambda v: float(v.norm())
    measurement_norm = max(norm(measurement), 1e-12)
    legacy_trace = legacy and not _policy_active(control)
    fidelity_before={k:getattr(fidelity,k,0) for k in ("weight_multiplications","prox_calls","prox_seconds")}
    work_before = {name: getattr(reg, name, 0) for name in ("prox_calls", "prox_iterations", "prox_seconds", "gradient_calls", "gradient_adjoint_calls")}
    auxiliary=getattr(reg,'operator',None)
    auxiliary_before=auxiliary.stats() if callable(getattr(auxiliary,'stats',None)) else {}
    haar_before = (getattr(reg, "analysis_calls", 0), getattr(reg, "synthesis_calls", 0), getattr(reg, "transform_seconds", 0))

    def endpoint(point, point_residual, current_step):
        gradient = operator.adjoint(fidelity.gradient(point_residual+measurement) if fidelity is not None else point_residual)
        if smooth_regularizer and weight: gradient = gradient+weight*reg.gradient(point)
        prox = proximal(reg, point - current_step * gradient, 0. if smooth_regularizer else current_step * weight,
                        operator, min_value, max_value, iteration=max(actual, 1), certified=not legacy)
        if not bool(torch.isfinite(gradient).all() and torch.isfinite(prox).all()):
            raise FloatingPointError("non_finite_endpoint_mapping")
        # Scale by ||x||/t, as used by the existing CT endpoint contract.
        value = norm((point - prox) / current_step) / max(norm(point) / current_step, 1e-12)
        if not math.isfinite(value):
            raise FloatingPointError("non_finite_endpoint_mapping")
        return value

    try:
        if step_size is None:
            if poisson:
                from .splitting import norm_bound
                estimate = norm_bound(operator, x)*float((fidelity.I0*fidelity.mask).max())*fidelity.scale**2
            else:
                estimate = _estimate_lipschitz(fidelity.whiten_operator(operator) if fidelity is not None else operator, measurement, power_iterations)
            if smooth_regularizer and weight:
                estimate += 8*weight/getattr(reg, 'delta', 1.)
        else: estimate = None
        step = float(step_size) if step_size is not None else (0.99 / estimate if estimate > 1e-12 else 1.0)
        if not math.isfinite(step):
            raise FloatingPointError("non_finite_initial_step")
        for iteration in range(1, limit + 1):
            # Nonnegative accepted images with a nonnegative CT operator keep
            # the Poisson exponential in its physical domain. Restart an
            # extrapolation that leaves it, rather than clipping the objective.
            if poisson and bool((momentum < 0).any()):
                momentum, acceleration = x.clone(), 1.
            r = operator.forward(momentum) - measurement
            g = operator.adjoint(fidelity.gradient(r+measurement) if fidelity is not None else r)
            if smooth_regularizer and weight: g = g+weight*reg.gradient(momentum)
            if previous_objective is None:
                previous_objective = (float(fidelity.value(r+measurement)) if fidelity is not None else 0.5*norm(r)**2) + weight*float(reg.value(x).sum())
            next_x, next_residual, step, _ = fista_update(
                operator, measurement, momentum, g, r, step, reg, weight, min_value, max_value, work, fidelity,
                smooth_regularizer, iteration, not legacy)
            # Commit a complete accepted iterate before optional diagnostics, so
            # a budget stop returns this point rather than an extrapolated state.
            old_x, x = x, next_x
            actual = iteration
            residual, prediction = next_residual, next_residual + measurement
            objective = (float(fidelity.value(prediction)) if fidelity is not None else 0.5 * norm(residual)**2) + weight * float(reg.value(x).sum())
            if not math.isfinite(objective):
                raise FloatingPointError("non_finite_composite_objective")
            endpoint_complete = False
            change = norm(x - old_x) / max(norm(old_x), 1e-12)
            mapping = endpoint(x, residual, step) if not legacy_trace else None
            endpoint_complete = mapping is not None
            objective_change = math.inf if previous_objective is None else abs(objective - previous_objective) / max(abs(previous_objective), 1e-12)
            mapping_tol = _control_threshold(control, "prox_gradient_mapping_tolerance", float(control.tolerance or 0))
            objective_tol = _control_threshold(control, "relative_objective_tolerance", float(control.tolerance or 0))
            criteria = {"prox_gradient_mapping": mapping is not None and mapping <= mapping_tol,
                        "relative_composite_objective_change": objective_change <= objective_tol}
            if control.discrepancy_target is not None:
                criteria["discrepancy"] = norm(residual)/measurement_norm <= control.discrepancy_target
            decision = monitor.observe(iteration, criteria=criteria, relative_change=change, monitor_value=objective)
            plateau = (control.stall_enabled and decision.checked and criteria.get("discrepancy") is False
                       and criteria["prox_gradient_mapping"] and criteria["relative_composite_objective_change"]
                       and change <= control.stall_relative_iterate_tolerance)
            if decision.checked:
                plateau_run = plateau_run + 1 if plateau else 0
            record_x = momentum if legacy_trace else x
            record_r = r if legacy_trace else residual
            record_obj = (float(fidelity.value(r+measurement)) if fidelity is not None else 0.5*norm(r)**2) + weight*float(reg.value(momentum).sum()) if legacy_trace else objective
            keep_going = recorder.record(iteration, record_x, residual=record_r, objective=record_obj,
                algorithm_residual=mapping, step_size=step, stopping_candidate=decision.converged,
                consecutive_criteria_count=decision.consecutive, criteria=criteria,
                native_criterion_name="normalized_prox_gradient_mapping", native_criterion_value=mapping,
                native_criterion_threshold=mapping_tol,
                metadata={"objective_state": "momentum" if legacy_trace else "returned", "mapping_state": "returned",
                          "complete_objective": True, "normalized_prox_gradient_mapping": mapping,
                          "relative_composite_objective_change": None if not math.isfinite(objective_change) else objective_change,
                          "native_plateau": bool(control.stall_enabled and criteria.get("discrepancy") is False and criteria["prox_gradient_mapping"] and criteria["relative_composite_objective_change"]),
                          "checked": decision.checked, "native_plateau_consecutive": plateau_run})
            previous_objective = objective
            if not keep_going:
                status, reason = "cancelled", "callback_cancelled"
                break
            terminal, terminal_reason = _policy_status(decision)
            if terminal or plateau_run >= control.stall_patience:
                status, reason = terminal or "stalled", terminal_reason or "stalled_before_discrepancy"
                break
            if decision.converged and control.stop_on_convergence:
                status, reason = "converged", "discrepancy_prox_gradient_and_objective_patience"
                break
            if legacy_trace and float(control.tolerance or 0) > 0 and bool(torch.all((x-old_x).flatten(1).norm(dim=1) <= control.tolerance*old_x.flatten(1).norm(dim=1).clamp_min(1.0))):
                status, reason = "stalled", "legacy_iterate_tolerance"
                break
            next_acceleration = 0.5*(1+math.sqrt(1+4*acceleration**2))
            momentum = x + ((acceleration-1)/next_acceleration)*(x-old_x)
            acceleration = next_acceleration
        if not endpoint_complete:
            if residual is None:
                prediction = operator.forward(x)
                residual = prediction - measurement
            mapping = endpoint(x, residual, step)
            endpoint_complete = True
    except OperatorBudgetExceeded as error:
        budget_interruption = {"kind": error.kind, "limit": error.limit, "attempted": error.attempted}
        status, reason = "resource_exhausted", "wall_time_budget_exhausted" if error.kind == "wall" else "operator_call_budget_exhausted"
    except ProximalAccuracyError as error:
        status, reason = 'max_iterations', str(error)
    except FloatingPointError as error:
        status, reason = "numerical_error", str(error)
        recorder.mark_numerical_error(reason)
    resources = {**{"fidelity_"+k:getattr(fidelity,k,0)-v for k,v in fidelity_before.items()}, **{name: getattr(reg, name, 0)-value for name, value in work_before.items()},
                 **work,
                 "haar_analysis_calls": getattr(reg, "analysis_calls", 0)-haar_before[0],
                 "haar_synthesis_calls": getattr(reg, "synthesis_calls", 0)-haar_before[1],
                 "haar_transform_seconds": getattr(reg, "transform_seconds", 0)-haar_before[2]}
    if auxiliary_before:
        resources.update({k:v-auxiliary_before[k] for k,v in auxiliary.stats().items()})
    return recorder.finish(x, actual_iterations=actual, status=status, stopping_reason=reason,
        final_residual=None if residual is None else norm(residual)/measurement_norm,
        final_objective=objective, predicted_measurement=prediction, resources=resources,
        metadata={"step_size": locals().get("step"), "operator_norm_squared": estimate,
                  "norm_estimate_is_upper_bound": False, "smooth_majorizer_checked": actual > 0,
                  "endpoint_mapping_complete": endpoint_complete,
                  "budget_interruption": budget_interruption,
                  "normalized_prox_gradient_mapping": mapping if endpoint_complete else None,
                  "trajectory_state": "momentum" if legacy_trace else "returned",
                  "proximal_accuracy": reg.proximal_accuracy,
                  "joint_prox_gap": getattr(reg, 'last_prox_gap', None),
                  "joint_prox_error_bound": getattr(reg, 'last_prox_error_bound', None),
                  "joint_prox_target": getattr(reg, 'last_prox_target', None),
                  "legacy_box_projection": legacy and (min_value is not None or max_value is not None),
                  "strength": weight, "regularizer": type(reg).__name__,
                  "level": getattr(reg, "level", None),
                  "tv_num_iterations": getattr(reg, "num_iterations", None),
                  "tv_tolerance": getattr(reg, "tolerance", None),
                  "native_plateau_stall": {"enabled": control.stall_enabled,
                      "consecutive": plateau_run, "patience": control.stall_patience,
                      "relative_iterate_tolerance": control.stall_relative_iterate_tolerance},
                  "native_termination_managed": True})


def fista(operator, y, **kwargs):
    result = solve_fista_detailed(operator, y, **kwargs)
    if result.status == "invalid_parameters":
        raise ValueError(result.stopping_reason)
    if result.status == "numerical_error":
        raise FloatingPointError(result.stopping_reason)
    if result.status == "resource_exhausted":
        raise OperatorBudgetExceeded(result.stopping_reason, **result.metadata["budget_interruption"])
    return result.reconstruction


class FISTASolver(InverseProblemSolver):
    def __init__(self, regularizer=None, strength=None, reg_strength=None, num_iterations=50,
                 step_size=None, tolerance=1e-5, power_iterations=12, min_value=None, max_value=None, fidelity=None):
        self.parameters = dict(regularizer=regularizer, strength=strength, reg_strength=reg_strength,
            num_iterations=num_iterations, step_size=step_size, tolerance=tolerance,
            power_iterations=power_iterations, min_value=min_value, max_value=max_value, fidelity=fidelity)

    def solve(self, measurement, operator, x_init=None, **kwargs):
        return fista(operator, measurement, x_init=x_init, **{**self.parameters, **kwargs})

    def solve_detailed(self, measurement, operator, x_init=None, *, control=None, callback=None, **kwargs):
        return solve_fista_detailed(operator, measurement, x_init=x_init, control=control, callback=callback,
                                    **{**self.parameters, **kwargs})

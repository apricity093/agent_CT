"""Expectation maximization for emission-count measurements."""
from __future__ import annotations
from typing import Iterable, Optional, Sequence, Any
from inv_framework.core.operator import ForwardOperator, LinearOperator
from inv_framework.solvers.base import InverseProblemSolver, ConsecutiveStoppingMonitor, IterationRecorder, SolveControl, SolveResult, resolve_control
from inv_framework.solvers._utils import apply_box_constraints, prepare_initial_image, require_linear_operator, validate_measurement_shape
from inv_framework.solvers.subset import make_angle_subsets, make_subset_operator, select_measurement_subset
import torch


def _require_nonnegative_count_data(measurement: torch.Tensor, solver: str) -> None:
    """Reject signed/log-domain data instead of silently clipping it."""

    if not torch.isfinite(measurement).all():
        raise ValueError(f"{solver} requires finite observations")
    if bool(torch.any(measurement < 0.0)):
        raise ValueError(
            f"{solver} requires nonnegative emission/count observations; "
            "log-domain or signed line-integral data are incompatible"
        )


def _positive_initial(
    measurement: torch.Tensor,
    operator: LinearOperator,
    x_init: Optional[torch.Tensor],
    initial_value: float,
    min_value: Optional[float],
) -> torch.Tensor:
    x = prepare_initial_image(
        measurement,
        operator,
        x_init=x_init,
        initial_value=float(initial_value),
    )
    lower = 0.0 if min_value is None else float(min_value)
    return x.clamp_min(lower)




def mlem(operator: LinearOperator,
         y: torch.Tensor,
         num_iterations: int = 50,
         x_init: torch.Tensor = None,
         initial_value: float = 1e-6,
         min_value: float = 0.0,
         max_value: float = None,
         eps: float = 1e-8) -> torch.Tensor:
    """MLEM for nonnegative linear projection data."""
    require_linear_operator(operator, "mlem")
    validate_measurement_shape(y, operator, "mlem")
    _require_nonnegative_count_data(y, "mlem")

    x = _positive_initial(y, operator, x_init, initial_value, min_value)
    y_nonnegative = y
    sensitivity = operator.adjoint(torch.ones_like(y_nonnegative)).clamp_min(float(eps))

    for _ in range(int(num_iterations)):
        prediction = operator.forward(x).clamp_min(float(eps))
        ratio = y_nonnegative / prediction
        correction = operator.adjoint(ratio) / sensitivity
        x = x * correction.clamp_min(0.0)
        x = apply_box_constraints(x, min_value=min_value, max_value=max_value)
    return x


class MLEMSolver(InverseProblemSolver):
    def __init__(self,
                 num_iterations: int = 50,
                 initial_value: float = 1e-6,
                 min_value: float = 0.0,
                 max_value: float = None,
                 eps: float = 1e-8):
        self.num_iterations = num_iterations
        self.initial_value = initial_value
        self.min_value = min_value
        self.max_value = max_value
        self.eps = eps

    def solve(self, measurement, operator: ForwardOperator, x_init=None, **kwargs):
        return mlem(
            operator,
            measurement,
            num_iterations=self.num_iterations,
            x_init=x_init,
            initial_value=kwargs.pop("initial_value", self.initial_value),
            min_value=kwargs.pop("min_value", self.min_value),
            max_value=kwargs.pop("max_value", self.max_value),
            eps=kwargs.pop("eps", self.eps),
        )

    def solve_detailed(self, measurement, operator: ForwardOperator, x_init=None, *, control=None, callback=None, **kwargs):
        pass

        return solve_mlem_detailed(
            operator, measurement,
            num_iterations=kwargs.pop("num_iterations", self.num_iterations),
            x_init=x_init,
            initial_value=kwargs.pop("initial_value", self.initial_value),
            min_value=kwargs.pop("min_value", self.min_value),
            max_value=kwargs.pop("max_value", self.max_value),
            eps=kwargs.pop("eps", self.eps),
            control=control, callback=callback,
        )


def _poisson_deviance(prediction: torch.Tensor, measurement: torch.Tensor, eps: float) -> float:
    pred = prediction.clamp_min(float(eps))
    observed = measurement.clamp_min(0.0)
    value = pred - observed + torch.where(
        observed > 0.0,
        observed * torch.log((observed + float(eps)) / pred),
        torch.zeros_like(observed),
    )
    return float((2.0 * value).sum().item())


def _normalized_poisson_deviance(
    prediction: torch.Tensor,
    measurement: torch.Tensor,
    eps: float,
) -> float:
    """Return dimensionless Poisson deviance normalized by ``2*sum(y)``."""

    raw = _poisson_deviance(prediction, measurement, eps)
    normalizer = max(2.0 * float(measurement.detach().sum().item()), float(eps))
    return raw / normalizer


def _validate_count_data(measurement: torch.Tensor, solver: str) -> None:
    if not torch.isfinite(measurement).all():
        raise ValueError(f"{solver} requires finite observations")
    if bool(torch.any(measurement < 0.0)):
        raise ValueError(
            f"{solver} requires nonnegative emission/count observations; "
            "log-domain or signed line-integral data are incompatible"
        )


def _statistical_parameter_errors(
    num_iterations: Any,
    initial_value: Any,
    min_value: Any,
    max_value: Any,
    eps: Any,
    *,
    block_size: Any = None,
    order_strategy: Any = "ordered",
    seed: Any = None,
) -> list[str]:
    from inv_framework.solvers._iteration import _common_parameter_errors, _finite_positive, _finite_scalar
    errors = _common_parameter_errors(num_iterations, min_value, max_value)
    if not _finite_positive(initial_value):
        errors.append("initial_value must be a finite positive number")
    if min_value is not None and _finite_scalar(min_value) and float(min_value) < 0.0:
        errors.append("min_value must be nonnegative")
    if max_value is not None and _finite_scalar(max_value) and float(max_value) < 0.0:
        errors.append("max_value must be nonnegative")
    if not _finite_positive(eps):
        errors.append("eps must be a finite positive number")
    if block_size is not None and (
        isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0
    ):
        errors.append("block_size must be a positive integer or None")
    if not isinstance(order_strategy, str) or order_strategy not in {"ordered", "random"}:
        errors.append("order_strategy must be 'ordered' or 'random'")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        errors.append("seed must be an integer or None")
    return errors


def _is_valid_em_prediction(
    value: Any,
    expected_shape: tuple[int, ...],
) -> bool:
    from inv_framework.solvers._iteration import _finite_tensor
    return (
        isinstance(value, torch.Tensor)
        and tuple(value.shape) == expected_shape
        and _finite_tensor(value)
        and not bool(torch.any(value < 0.0))
    )


def _statistical_tolerance(control: SolveControl, name: str) -> float:
    explicit = getattr(control, name, None)
    if explicit is not None:
        return float(explicit)
    if control.tolerance is not None:
        return float(control.tolerance)
    return 1e-5


def _solve_em_detailed(
    algorithm: str,
    operator: LinearOperator,
    measurement: torch.Tensor,
    *,
    num_iterations: int = 50,
    block_size: int | None = None,
    subset_indices: Optional[Iterable[Sequence[int]]] = None,
    order_strategy: str = "ordered",
    seed: int | None = None,
    x_init: torch.Tensor | None = None,
    initial_value: float = 1e-6,
    min_value: float = 0.0,
    max_value: float | None = None,
    eps: float = 1e-8,
    control: SolveControl | None = None,
    callback: Callback | None = None,
) -> SolveResult:
    """Shared MLEM/OSEM loop with complete-epoch evidence only."""
    from inv_framework.solvers._iteration import _finish_prediction, _finish_status, _finite_scalar, _finite_tensor, _global_norm, _normalized, _numerical_result, _parameter_failure_result, _policy_active, _relative_change, _safe_record

    parameters: dict[str, Any] = {
        "num_iterations": num_iterations,
        "initial_value": initial_value,
        "min_value": min_value,
        "max_value": max_value,
        "eps": eps,
    }
    if algorithm == "osem":
        parameters.update({
            "block_size": block_size,
            "order_strategy": order_strategy,
            "seed": seed,
        })
    try:
        require_linear_operator(operator, algorithm)
        validate_measurement_shape(measurement, operator, algorithm)
    except (TypeError, ValueError) as error:
        return _parameter_failure_result(
            algorithm, measurement, operator, x_init=x_init, errors=[str(error)],
            max_iterations=None, parameters=parameters,
        )

    subset_values: list[Sequence[int]] | None = None
    if algorithm == "osem" and subset_indices is not None:
        try:
            subset_values = list(subset_indices)
        except (TypeError, ValueError) as error:
            return _parameter_failure_result(
                algorithm, measurement, operator, x_init=x_init, errors=[str(error)],
                max_iterations=None, parameters=parameters,
            )
        try:
            parameters["subset_indices"] = [
                tuple(int(value) for value in item) for item in subset_values
            ]
        except (TypeError, ValueError, OverflowError) as error:
            return _parameter_failure_result(
                algorithm, measurement, operator, x_init=x_init, errors=[str(error)],
                max_iterations=None, parameters=parameters,
            )
    parameter_errors = _statistical_parameter_errors(
        num_iterations, initial_value, min_value, max_value, eps,
        block_size=block_size, order_strategy=order_strategy, seed=seed,
    )
    if parameter_errors:
        return _parameter_failure_result(
            algorithm, measurement, operator, x_init=x_init, errors=parameter_errors,
            max_iterations=None, parameters=parameters,
        )
    if not _finite_tensor(measurement):
        return _numerical_result(
            algorithm, measurement, operator, x_init=x_init, max_iterations=None,
            reason="non_finite_measurement", parameters=parameters,
        )
    if bool(torch.any(measurement < 0.0)):
        return _parameter_failure_result(
            algorithm, measurement, operator, x_init=x_init,
            errors=[
                f"{algorithm} requires nonnegative emission/count observations; "
                "log-domain or signed line-integral data are incompatible"
            ],
            max_iterations=None, parameters=parameters,
        )
    try:
        control = resolve_control(
            control,
            default_iterations=int(num_iterations),
            default_tolerance=1e-5,
            callback=callback,
        )
    except (TypeError, ValueError) as error:
        return _parameter_failure_result(
            algorithm, measurement, operator, x_init=x_init, errors=[str(error)],
            max_iterations=None, parameters=parameters,
        )
    limit = min(int(num_iterations), int(control.max_iterations or num_iterations))
    parameters["num_iterations"] = int(num_iterations)
    parameters["max_iterations"] = limit
    try:
        x = prepare_initial_image(
            measurement, operator, x_init=x_init, initial_value=float(initial_value)
        )
        x = apply_box_constraints(x, min_value=min_value, max_value=max_value)
    except (TypeError, ValueError) as error:
        return _parameter_failure_result(
            algorithm, measurement, operator, x_init=x_init, errors=[str(error)],
            max_iterations=limit, parameters=parameters,
        )
    if not _finite_tensor(x) or bool(torch.any(x < 0.0)):
        return _numerical_result(
            algorithm, measurement, operator, x_init=x_init, max_iterations=limit,
            reason="non_finite_initial_reconstruction", parameters=parameters,
        )

    recorder = IterationRecorder(control, measurement, operator, algorithm=algorithm)
    recorder.set_initial(x)
    actual = 0

    def numerical_result(reason: str, predicted: torch.Tensor | None = None) -> SolveResult:
        from inv_framework.solvers._iteration import _finite_tensor
        recorder.mark_numerical_error(reason)
        return recorder.finish(
            x,
            actual_iterations=actual,
            status="numerical_error",
            stopping_reason=reason,
            predicted_measurement=predicted if _finite_tensor(predicted) else None,
            metadata={
                "criterion": "normalized_poisson_deviance_and_complete_epoch_iterate_change",
                "iteration_unit": "epochs",
                "native_termination_managed": True,
                **parameters,
            },
        )

    subsets: list[torch.Tensor] = []
    subset_operators: list[LinearOperator] = []
    sensitivities: list[torch.Tensor] = []
    sensitivity_masks: list[torch.Tensor] = []
    if algorithm == "osem":
        if block_size is None and subset_values is None:
            block_size = max(int(operator.range_shape[-2]) // 10, 1)
            parameters["block_size"] = block_size
        try:
            subsets = make_angle_subsets(
                num_angles=int(operator.range_shape[-2]),
                block_size=block_size,
                subset_indices=subset_values,
                order_strategy=order_strategy,
                seed=seed,
                device=measurement.device,
            )
            subset_operators = [make_subset_operator(operator, indices) for indices in subsets]
        except (NotImplementedError, TypeError, ValueError) as error:
            return _parameter_failure_result(
                algorithm, measurement, operator, x_init=x_init, errors=[str(error)],
                max_iterations=limit, parameters=parameters,
            )
        parameters["subset_sizes"] = [int(indices.numel()) for indices in subsets]
        for sub_operator, indices in zip(subset_operators, subsets):
            y_sub = select_measurement_subset(measurement, indices)
            try:
                sensitivity = sub_operator.adjoint(torch.ones_like(y_sub))
            except Exception as error:
                return numerical_result("subset_sensitivity_evaluation_failed")
            expected_domain = (measurement.shape[0], *tuple(operator.domain_shape))
            if (
                not isinstance(sensitivity, torch.Tensor)
                or tuple(sensitivity.shape) != expected_domain
                or not _finite_tensor(sensitivity)
                or bool(torch.any(sensitivity < 0.0))
                or bool(torch.all(sensitivity <= 0.0))
            ):
                return numerical_result("non_positive_subset_sensitivity")
            sensitivities.append(sensitivity.clamp_min(float(eps)))
            sensitivity_masks.append(sensitivity > float(eps))
    else:
        try:
            sensitivity = operator.adjoint(torch.ones_like(measurement))
        except Exception as error:
            return numerical_result("sensitivity_evaluation_failed")
        expected_domain = (measurement.shape[0], *tuple(operator.domain_shape))
        if (
            not isinstance(sensitivity, torch.Tensor)
            or tuple(sensitivity.shape) != expected_domain
            or not _finite_tensor(sensitivity)
            or bool(torch.any(sensitivity < 0.0))
            or bool(torch.all(sensitivity <= 0.0))
        ):
            return numerical_result("non_positive_sensitivity")
        sensitivities = [sensitivity.clamp_min(float(eps))]
        sensitivity_masks = [sensitivity > float(eps)]

    try:
        prediction = _finish_prediction(operator, x)
    except Exception as error:
        return numerical_result("initial_prediction_failed")
    expected_range = (measurement.shape[0], *tuple(operator.range_shape))
    if not _is_valid_em_prediction(prediction, expected_range):
        return numerical_result("invalid_initial_prediction", prediction)
    previous_deviance = _normalized_poisson_deviance(prediction, measurement, eps)
    if not _finite_scalar(previous_deviance):
        return numerical_result("non_finite_initial_poisson_deviance", prediction)

    measurement_norm = _global_norm(measurement)
    monitor = ConsecutiveStoppingMonitor(control)
    actual = 0
    converged = False
    cancelled = False
    terminal_status: str | None = None
    terminal_reason: str | None = None
    plateau_run = 0
    cycle_run = 0
    cycle_amplitude = 0.0
    cycle_patience = int((control.metadata or {}).get("osem_cycle_patience", control.stall_patience))
    cycle_tolerance = float(
        (control.metadata or {}).get(
            "osem_cycle_tolerance",
            max(control.stall_relative_iterate_tolerance, 1e-8),
        )
    )
    epoch_states: list[torch.Tensor] = []
    objective_tolerance = _statistical_tolerance(control, "relative_objective_tolerance")
    iterate_tolerance = _statistical_tolerance(control, "relative_iterate_tolerance")

    for epoch in range(1, limit + 1):
        previous = x
        subset_amplitude = 0.0
        try:
            if algorithm == "mlem":
                ratio = measurement / prediction.clamp_min(float(eps))
                correction = operator.adjoint(ratio)
                if (
                    tuple(correction.shape) != tuple(x.shape)
                    or not _finite_tensor(ratio)
                    or not _finite_tensor(correction)
                    or bool(torch.any(correction < 0.0))
                ):
                    return numerical_result("invalid_multiplicative_update", prediction)
                update_factor = correction / sensitivities[0]
                x = x * torch.where(
                    sensitivity_masks[0], update_factor, torch.ones_like(update_factor)
                )
                x = apply_box_constraints(x, min_value=min_value, max_value=max_value)
            else:
                for subset_index, (indices, sub_operator, sensitivity, sensitivity_mask) in enumerate(
                    zip(subsets, subset_operators, sensitivities, sensitivity_masks)
                ):
                    y_sub = select_measurement_subset(measurement, indices)
                    prediction_sub = sub_operator.forward(x)
                    expected_subset = (measurement.shape[0], *tuple(sub_operator.range_shape))
                    if not _is_valid_em_prediction(prediction_sub, expected_subset):
                        return numerical_result("invalid_subset_prediction", prediction)
                    ratio = y_sub / prediction_sub.clamp_min(float(eps))
                    correction = sub_operator.adjoint(ratio)
                    if (
                        tuple(correction.shape) != tuple(x.shape)
                        or not _finite_tensor(ratio)
                        or not _finite_tensor(correction)
                        or bool(torch.any(correction < 0.0))
                    ):
                        return numerical_result("invalid_subset_multiplicative_update", prediction)
                    before_subset = x
                    update_factor = correction / sensitivity
                    x = x * torch.where(
                        sensitivity_mask, update_factor, torch.ones_like(update_factor)
                    )
                    x = apply_box_constraints(x, min_value=min_value, max_value=max_value)
                    if not _finite_tensor(x) or bool(torch.any(x < 0.0)):
                        return numerical_result("non_finite_multiplicative_update", prediction)
                    subset_amplitude = max(subset_amplitude, _relative_change(before_subset, x))
        except Exception as error:
            return numerical_result("em_update_failed", prediction)
        if not _finite_tensor(x) or bool(torch.any(x < 0.0)):
            return numerical_result("non_finite_multiplicative_update", prediction)
        try:
            prediction = _finish_prediction(operator, x)
        except Exception as error:
            return numerical_result("prediction_evaluation_failed")
        if not _is_valid_em_prediction(prediction, expected_range):
            return numerical_result("invalid_prediction", prediction)
        change = _relative_change(previous, x)
        raw_deviance = _poisson_deviance(prediction, measurement, eps)
        normalized_deviance = _normalized_poisson_deviance(prediction, measurement, eps)
        deviance_change = abs(normalized_deviance - previous_deviance) / max(
            abs(previous_deviance), 1e-12
        )
        residual = prediction - measurement
        if not all(
            _finite_scalar(value)
            for value in (change, raw_deviance, normalized_deviance, deviance_change)
        ) or not _finite_tensor(residual):
            return numerical_result("non_finite_statistical_diagnostic", prediction)

        cycle_detected = False
        period_two_distance = None
        adjacent_distance = None
        if algorithm == "osem":
            if len(epoch_states) >= 2:
                period_two_distance = _relative_change(epoch_states[-2], x)
                adjacent_distance = _relative_change(epoch_states[-1], x)
                cycle_detected = (
                    period_two_distance <= cycle_tolerance
                    and adjacent_distance > max(5.0 * cycle_tolerance, 2.0 * iterate_tolerance, 1e-12)
                )
            fixed_point_with_subset_motion = (
                change <= cycle_tolerance
                and subset_amplitude > max(10.0 * cycle_tolerance, 10.0 * iterate_tolerance, 1e-12)
            )
            cycle_detected = cycle_detected or fixed_point_with_subset_motion
            cycle_run = cycle_run + 1 if cycle_detected else 0
            cycle_amplitude = max(subset_amplitude, float(adjacent_distance or 0.0))
            epoch_states.append(x.detach().clone())
            if len(epoch_states) > 2:
                epoch_states.pop(0)

        discrepancy_ok = (
            control.discrepancy_target is None
            or normalized_deviance <= float(control.discrepancy_target)
        )
        native_ok = (
            deviance_change <= objective_tolerance
            and change <= iterate_tolerance
        )
        criteria: dict[str, bool] = {
            "normalized_poisson_deviance_change": deviance_change <= objective_tolerance,
            "relative_iterate_change": change <= iterate_tolerance,
        }
        if control.discrepancy_target is not None:
            criteria = {"discrepancy": discrepancy_ok, **criteria}
        if algorithm == "osem" and _policy_active(control):
            criteria["osem_cycle_free"] = not cycle_detected
        decision = monitor.observe(
            epoch,
            criteria=criteria,
            relative_change=change,
            monitor_value=normalized_deviance,
        )
        converged = decision.converged
        if _policy_active(control):
            if decision.diverged:
                terminal_status = "diverged"
                terminal_reason = "persistent_poisson_deviance_increase"
            elif (
                decision.checked
                and control.stall_enabled
                and control.discrepancy_target is not None
                and not discrepancy_ok
                and native_ok
            ):
                plateau_run += 1
                if plateau_run >= control.stall_patience:
                    terminal_status = "stalled"
                    terminal_reason = "poisson_deviance_plateau_before_discrepancy"
            else:
                plateau_run = 0
            if (
                terminal_status is None
                and algorithm == "osem"
                and control.stall_enabled
                and decision.checked
                and cycle_run >= cycle_patience
            ):
                terminal_status = "stalled"
                terminal_reason = "osem_subset_cycle_detected"
        actual = epoch
        metadata = {
            "checked": decision.checked,
            "complete_epoch": True,
            "epoch_boundary": True,
            "poisson_deviance": raw_deviance,
            "normalized_poisson_deviance": normalized_deviance,
            "previous_normalized_poisson_deviance": previous_deviance,
            "normalized_poisson_deviance_change": deviance_change,
            "poisson_deviance_normalization": "2*sum_observed",
            "relative_epoch_change": change,
            "normalized_poisson_deviance_tolerance": objective_tolerance,
            "relative_iterate_tolerance": iterate_tolerance,
            "discrepancy_metric": "normalized_poisson_deviance",
            "discrepancy_satisfied": discrepancy_ok,
            "plateau_run": plateau_run,
            "cycle_detected": cycle_detected,
            "cycle_free": not cycle_detected,
            "cycle_run": cycle_run,
            "cycle_amplitude": cycle_amplitude,
            "subset_amplitude": subset_amplitude,
            "native_termination_managed": True,
        }
        if not _safe_record(
            recorder,
            epoch,
            x,
            residual=residual,
            objective=raw_deviance,
            algorithm_residual=normalized_deviance,
            stopping_candidate=bool(decision.checked or converged or terminal_status),
            consecutive_criteria_count=decision.consecutive,
            criteria=criteria,
            native_criterion_name="normalized_poisson_deviance_change",
            native_criterion_value=deviance_change,
            native_criterion_threshold=objective_tolerance,
            epoch=epoch,
            subset_count=len(subsets) if algorithm == "osem" else None,
            metadata=metadata,
        ):
            cancelled = True
            break
        previous_deviance = normalized_deviance
        if terminal_status or (converged and control.stop_on_convergence):
            break

    final_residual_tensor = prediction - measurement
    final_residual = _normalized(_global_norm(final_residual_tensor), measurement_norm)
    status = terminal_status or (
        "cancelled" if cancelled else _finish_status(
            actual=actual,
            limit=limit,
            converged=converged,
            cancelled=cancelled,
            numerical_failure=recorder.numerical_failure,
        )
    )
    converged_reason = (
        "poisson_deviance_and_relative_epoch_change_patience"
        if algorithm == "osem" else
        "poisson_deviance_and_relative_iterate_change_patience"
    )
    stopping_reason = (
        terminal_reason
        if terminal_reason else
        converged_reason if converged and actual < limit else
        "callback_cancelled" if cancelled else
        "maximum_epochs_reached"
    )
    return recorder.finish(
        x,
        actual_iterations=actual,
        status=status,
        stopping_reason=stopping_reason,
        final_residual=final_residual,
        final_objective=_poisson_deviance(prediction, measurement, eps),
        predicted_measurement=prediction,
        metadata={
            "likelihood": "poisson_emission_style",
            "criterion": "normalized_poisson_deviance_and_complete_epoch_iterate_change",
            "poisson_deviance_normalization": "2*sum_observed",
            "complete_epoch_count": actual,
            "subset_count": len(subsets) if algorithm == "osem" else None,
            "normalized_poisson_deviance": previous_deviance,
            "normalized_poisson_deviance_tolerance": objective_tolerance,
            "relative_iterate_tolerance": iterate_tolerance,
            "cycle_detected": bool(algorithm == "osem" and cycle_run >= cycle_patience),
            "cycle_run": cycle_run,
            "cycle_amplitude": cycle_amplitude,
            "native_termination_managed": True,
            "max_iterations": limit,
            **parameters,
        },
    )


def solve_mlem_detailed(
    operator: LinearOperator,
    measurement: torch.Tensor,
    *,
    num_iterations: int = 50,
    x_init: torch.Tensor | None = None,
    initial_value: float = 1e-6,
    min_value: float = 0.0,
    max_value: float | None = None,
    eps: float = 1e-8,
    control: SolveControl | None = None,
    callback: Callback | None = None,
) -> SolveResult:
    return _solve_em_detailed(
        "mlem", operator, measurement,
        num_iterations=num_iterations,
        x_init=x_init,
        initial_value=initial_value,
        min_value=min_value,
        max_value=max_value,
        eps=eps,
        control=control,
        callback=callback,
    )

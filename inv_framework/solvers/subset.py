"""Ordered-subset reconstruction and projection-angle scheduling."""
from __future__ import annotations
from typing import List, Iterable, Optional, Sequence, Any
from inv_framework.core.operator import ForwardOperator, LinearOperator
from inv_framework.solvers.base import InverseProblemSolver, ConsecutiveStoppingMonitor, IterationRecorder, SolveControl, SolveResult, resolve_control
from inv_framework.solvers._utils import apply_box_constraints, prepare_initial_image, require_linear_operator, validate_measurement_shape
import torch


def osem(operator: LinearOperator,
         y: torch.Tensor,
         num_iterations: int = 50,
         block_size: int = None,
         subset_indices: Optional[Iterable[Sequence[int]]] = None,
         order_strategy: str = "ordered",
         seed: Optional[int] = None,
         x_init: torch.Tensor = None,
         initial_value: float = 1e-6,
         min_value: float = 0.0,
         max_value: float = None,
         eps: float = 1e-8) -> torch.Tensor:
    """Ordered-subsets EM for nonnegative linear projection data."""
    from inv_framework.solvers.statistical import _positive_initial, _require_nonnegative_count_data
    require_linear_operator(operator, "osem")
    validate_measurement_shape(y, operator, "osem")
    _require_nonnegative_count_data(y, "osem")

    if block_size is None:
        block_size = max(int(operator.range_shape[-2]) // 10, 1)
    x = _positive_initial(y, operator, x_init, initial_value, min_value)
    y_nonnegative = y
    subsets = make_angle_subsets(
        num_angles=operator.range_shape[-2],
        block_size=block_size,
        subset_indices=subset_indices,
        order_strategy=order_strategy,
        seed=seed,
        device=y.device,
    )

    for _ in range(int(num_iterations)):
        for indices in subsets:
            sub_operator = make_subset_operator(operator, indices)
            y_sub = select_measurement_subset(y_nonnegative, indices)
            sensitivity = sub_operator.adjoint(torch.ones_like(y_sub)).clamp_min(float(eps))
            prediction = sub_operator.forward(x).clamp_min(float(eps))
            ratio = y_sub / prediction
            correction = sub_operator.adjoint(ratio) / sensitivity
            x = x * correction.clamp_min(0.0)
            x = apply_box_constraints(x, min_value=min_value, max_value=max_value)
    return x


class OSEMSolver(InverseProblemSolver):
    def __init__(self,
                 num_iterations: int = 50,
                 block_size: int = None,
                 subset_indices: Optional[Iterable[Sequence[int]]] = None,
                 order_strategy: str = "ordered",
                 seed: Optional[int] = None,
                 initial_value: float = 1e-6,
                 min_value: float = 0.0,
                 max_value: float = None,
                 eps: float = 1e-8):
        self.num_iterations = num_iterations
        self.block_size = block_size
        self.subset_indices = _store_subset_indices(subset_indices)
        self.order_strategy = order_strategy
        self.seed = seed
        self.initial_value = initial_value
        self.min_value = min_value
        self.max_value = max_value
        self.eps = eps

    def solve(self, measurement, operator: ForwardOperator, x_init=None, **kwargs):
        return osem(
            operator,
            measurement,
            num_iterations=self.num_iterations,
            block_size=kwargs.pop("block_size", self.block_size),
            subset_indices=kwargs.pop("subset_indices", self.subset_indices),
            order_strategy=kwargs.pop("order_strategy", self.order_strategy),
            seed=kwargs.pop("seed", self.seed),
            x_init=x_init,
            initial_value=kwargs.pop("initial_value", self.initial_value),
            min_value=kwargs.pop("min_value", self.min_value),
            max_value=kwargs.pop("max_value", self.max_value),
            eps=kwargs.pop("eps", self.eps),
        )

    def solve_detailed(self, measurement, operator: ForwardOperator, x_init=None, *, control=None, callback=None, **kwargs):
        pass

        return solve_osem_detailed(
            operator, measurement,
            num_iterations=kwargs.pop("num_iterations", self.num_iterations),
            block_size=kwargs.pop("block_size", self.block_size),
            subset_indices=kwargs.pop("subset_indices", self.subset_indices),
            order_strategy=kwargs.pop("order_strategy", self.order_strategy),
            seed=kwargs.pop("seed", self.seed),
            x_init=x_init,
            initial_value=kwargs.pop("initial_value", self.initial_value),
            min_value=kwargs.pop("min_value", self.min_value),
            max_value=kwargs.pop("max_value", self.max_value),
            eps=kwargs.pop("eps", self.eps),
            control=control, callback=callback,
        )


def _safe_reciprocal(z: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.where(z.abs() < eps, torch.full_like(z, float("inf")), z).reciprocal()


def _store_subset_indices(subset_indices: Optional[Iterable[Sequence[int]]]):
    if subset_indices is None:
        return None
    return [tuple(int(v) for v in indices) for indices in subset_indices]


def ossart(operator: LinearOperator,
           y: torch.Tensor,
           num_iterations: int = 100,
           block_size: int = None,
           subset_indices: Optional[Iterable[Sequence[int]]] = None,
           order_strategy: str = "ordered",
           seed: Optional[int] = None,
           relaxation: float = 1.0,
           min_value: float = None,
           max_value: float = None,
           x_init: torch.Tensor = None,
           eps: float = 1e-8) -> torch.Tensor:
    """Ordered-subsets SART."""
    from inv_framework.solvers.algebraic import sart
    if block_size is None:
        block_size = max(int(operator.range_shape[-2]) // 10, 1)
    return sart(
        operator,
        y,
        num_iterations=num_iterations,
        block_size=block_size,
        subset_indices=subset_indices,
        order_strategy=order_strategy,
        seed=seed,
        relaxation=relaxation,
        min_value=min_value,
        max_value=max_value,
        x_init=x_init,
        eps=eps,
    )


class OSSARTSolver(InverseProblemSolver):
    def __init__(self,
                 num_iterations: int = 100,
                 block_size: int = None,
                 subset_indices: Optional[Iterable[Sequence[int]]] = None,
                 order_strategy: str = "ordered",
                 seed: Optional[int] = None,
                 relaxation: float = 1.0,
                 min_value: float = None,
                 max_value: float = None,
                 eps: float = 1e-8):
        self.num_iterations = num_iterations
        self.block_size = block_size
        self.subset_indices = _store_subset_indices(subset_indices)
        self.order_strategy = order_strategy
        self.seed = seed
        self.relaxation = relaxation
        self.min_value = min_value
        self.max_value = max_value
        self.eps = eps

    def solve(self, measurement, operator: ForwardOperator, x_init=None, **kwargs):
        return ossart(
            operator,
            measurement,
            num_iterations=self.num_iterations,
            block_size=kwargs.pop("block_size", self.block_size),
            subset_indices=kwargs.pop("subset_indices", self.subset_indices),
            order_strategy=kwargs.pop("order_strategy", self.order_strategy),
            seed=kwargs.pop("seed", self.seed),
            relaxation=kwargs.pop("relaxation", self.relaxation),
            min_value=kwargs.pop("min_value", self.min_value),
            max_value=kwargs.pop("max_value", self.max_value),
            x_init=x_init,
            eps=kwargs.pop("eps", self.eps),
        )

    def solve_detailed(self, measurement, operator: ForwardOperator, x_init=None, *, control=None, callback=None, **kwargs):
        pass

        return solve_os_sart_detailed(
            operator, measurement,
            num_iterations=kwargs.pop("num_iterations", self.num_iterations),
            block_size=kwargs.pop("block_size", self.block_size),
            subset_indices=kwargs.pop("subset_indices", self.subset_indices),
            order_strategy=kwargs.pop("order_strategy", self.order_strategy),
            seed=kwargs.pop("seed", self.seed),
            relaxation=kwargs.pop("relaxation", self.relaxation),
            min_value=kwargs.pop("min_value", self.min_value),
            max_value=kwargs.pop("max_value", self.max_value),
            x_init=x_init,
            eps=kwargs.pop("eps", self.eps),
            control=control, callback=callback,
        )


def _subset_solver_detailed(
    algorithm: str,
    operator: LinearOperator,
    measurement: torch.Tensor,
    *,
    num_iterations: int,
    block_size: int | None,
    subset_indices: Optional[Iterable[Sequence[int]]],
    order_strategy: str,
    seed: int | None,
    relaxation: float,
    min_value: float | None,
    max_value: float | None,
    x_init: torch.Tensor | None,
    eps: float,
    control: SolveControl | None,
    callback: Callback | None,
) -> SolveResult:
    from inv_framework.solvers._iteration import _common_parameter_errors, _finish_prediction, _finish_status, _finite_scalar, _finite_tensor, _global_norm, _native_tolerance, _normalized, _numerical_result, _objective, _parameter_failure_result, _policy_active, _policy_status, _relative_change, _row_action_criteria, _safe_record, _tolerance_reached
    require_linear_operator(operator, algorithm)
    batch = validate_measurement_shape(measurement, operator, algorithm)
    parameter_errors = _common_parameter_errors(num_iterations, min_value, max_value)
    if (
        isinstance(relaxation, bool)
        or not isinstance(relaxation, (int, float))
        or (not _finite_scalar(relaxation))
    ):
        parameter_errors.append("relaxation must be a finite number")
    elif not 0.0 < float(relaxation) <= 1.0:
        parameter_errors.append("relaxation must satisfy 0 < relaxation <= 1")
    if (
        isinstance(eps, bool)
        or not isinstance(eps, (int, float))
        or (not _finite_scalar(eps))
        or (float(eps) <= 0.0)
    ):
        parameter_errors.append("eps must be a finite positive number")
    if block_size is not None and (
        isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size <= 0
    ):
        parameter_errors.append("block_size must be a positive integer or None")
    elif block_size is not None and int(block_size) > int(operator.range_shape[-2]):
        parameter_errors.append(
            f"block_size must be no greater than the number of views ({int(operator.range_shape[-2])})"
        )
    subset_spec: list[tuple[int, ...]] | None = None
    if subset_indices is not None:
        try:
            subset_spec = [
                tuple((int(value) for value in indices)) for indices in subset_indices
            ]
        except (TypeError, ValueError):
            parameter_errors.append(
                "subset_indices must be an iterable of integer index groups"
            )
    if parameter_errors:
        return _parameter_failure_result(
            algorithm,
            measurement,
            operator,
            x_init=x_init,
            errors=parameter_errors,
            max_iterations=None,
            parameters={
                "num_iterations": num_iterations,
                "block_size": block_size,
                "subset_indices": subset_spec,
                "relaxation": relaxation,
                "eps": eps,
                "min_value": min_value,
                "max_value": max_value,
            },
        )
    control = resolve_control(
        control,
        default_iterations=int(num_iterations),
        default_tolerance=1e-05,
        callback=callback,
    )
    limit = min(int(num_iterations), int(control.max_iterations or num_iterations))
    parameters = {
        "num_iterations": int(num_iterations),
        "block_size": block_size,
        "subset_indices": subset_spec,
        "order_strategy": order_strategy,
        "seed": seed,
        "relaxation": float(relaxation),
        "eps": float(eps),
        "min_value": min_value,
        "max_value": max_value,
    }
    if not _finite_tensor(measurement):
        return _numerical_result(
            algorithm,
            measurement,
            operator,
            x_init=x_init,
            max_iterations=limit,
            reason="non_finite_measurement",
            parameters=parameters,
        )
    x = prepare_initial_image(measurement, operator, x_init=x_init, initial_value=0.0)
    if not _finite_tensor(x):
        return _numerical_result(
            algorithm,
            measurement,
            operator,
            x_init=x_init,
            max_iterations=limit,
            reason="non_finite_initial_reconstruction",
            parameters=parameters,
        )
    recorder = IterationRecorder(control, measurement, operator, algorithm=algorithm)
    monitor = ConsecutiveStoppingMonitor(control)
    recorder.set_initial(x)
    try:
        subsets = make_angle_subsets(
            num_angles=int(operator.range_shape[-2]),
            block_size=block_size,
            subset_indices=subset_spec,
            order_strategy=order_strategy,
            seed=seed,
            device=measurement.device,
        )
    except (TypeError, ValueError) as error:
        return _parameter_failure_result(
            algorithm,
            measurement,
            operator,
            x_init=x_init,
            errors=[str(error)],
            max_iterations=limit,
            parameters=parameters,
        )
    domain = (batch, *tuple(operator.domain_shape))
    measurement_norm = _global_norm(measurement)

    actual = 0
    converged = False
    cancelled = False
    terminal_status = None
    terminal_reason = None
    prediction = torch.zeros_like(measurement)
    native_tolerance = _native_tolerance(control)
    partial_epoch: int | None = None
    subset_updates = 0
    completed_x = x
    for epoch in range(1, limit + 1):
        previous = x
        epoch_complete = True
        for index, indices in enumerate(subsets):
            sub_operator = make_subset_operator(operator, indices)
            y_sub = select_measurement_subset(measurement, indices)
            ones_range = torch.ones(
                (batch, *tuple(sub_operator.range_shape)),
                device=measurement.device,
                dtype=measurement.dtype,
            )
            row_weight = sub_operator.forward(
                torch.ones(domain, device=measurement.device, dtype=measurement.dtype)
            )
            row_weight = torch.where(
                row_weight.abs() < float(eps),
                torch.full_like(row_weight, float("inf")),
                row_weight,
            ).reciprocal()
            if not _finite_tensor(row_weight):
                recorder.mark_numerical_error("non_finite_row_normalization")
                epoch_complete = False
                break
            column_weight = sub_operator.adjoint(ones_range)
            column_weight = torch.where(
                column_weight.abs() < float(eps),
                torch.full_like(column_weight, float("inf")),
                column_weight,
            ).reciprocal()
            if not _finite_tensor(column_weight):
                recorder.mark_numerical_error("non_finite_column_normalization")
                epoch_complete = False
                break
            residual = sub_operator.forward(x) - y_sub
            if not _finite_tensor(residual):
                recorder.mark_numerical_error("non_finite_residual")
                epoch_complete = False
                break
            x = x - float(relaxation) * column_weight * sub_operator.adjoint(
                row_weight * residual
            )
            x = apply_box_constraints(x, min_value=min_value, max_value=max_value)
            subset_updates += 1
            if not _finite_tensor(x):
                recorder.mark_numerical_error("non_finite_reconstruction")
                epoch_complete = False
                break
        if not epoch_complete:
            partial_epoch = epoch
            break
        prediction = _finish_prediction(operator, x)
        residual = prediction - measurement
        if not _finite_tensor(prediction) or not _finite_tensor(residual):
            recorder.mark_numerical_error("non_finite_epoch_prediction")
            partial_epoch = epoch
            break
        change = _relative_change(previous, x)
        residual_value = _normalized(_global_norm(residual), measurement_norm)
        if not _finite_scalar(change):
            recorder.mark_numerical_error("non_finite_relative_epoch_change")
            partial_epoch = epoch
            break
        if not _finite_scalar(residual_value):
            recorder.mark_numerical_error("non_finite_discrepancy")
            partial_epoch = epoch
            break
        criteria, native_tolerance = _row_action_criteria(
            control, residual_value, change, epoch=True
        )
        decision = monitor.observe(
            epoch,
            criteria=criteria,
            relative_change=change,
            monitor_value=residual_value,
        )
        converged = (
            decision.converged
            if _policy_active(control)
            else _tolerance_reached(
                control, residual=residual_value, change=change, require_both=True
            )
        )
        terminal_status, terminal_reason = (
            _policy_status(decision) if _policy_active(control) else (None, None)
        )
        actual = epoch
        completed_x = x
        objective = _objective(residual)
        if not _finite_scalar(objective):
            recorder.mark_numerical_error("non_finite_objective")
            partial_epoch = epoch
            break
        if not _safe_record(
            recorder,
            epoch,
            x,
            residual=residual,
            objective=objective,
            algorithm_residual=residual_value,
            relaxation=float(relaxation),
            stopping_candidate=converged,
            consecutive_criteria_count=decision.consecutive,
            criteria=criteria,
            native_criterion_name="relative_epoch_change",
            native_criterion_value=change,
            native_criterion_threshold=native_tolerance,
            epoch=epoch,
            subset_count=len(subsets),
            metadata={
                "complete_sweep": True,
                "epoch_boundary": True,
                "checked": decision.checked,
                "unit": "epochs",
                "subset_sizes": [int(item.numel()) for item in subsets],
            },
        ):
            cancelled = True
            break
        if terminal_status or (converged and control.stop_on_convergence):
            break
    if recorder.numerical_failure:
        return recorder.finish(
            x,
            actual_iterations=actual,
            status="numerical_error",
            stopping_reason=recorder.numerical_error_reason
            or "non_finite_solver_state",
            predicted_measurement=prediction,
            metadata={
                "criterion": "complete_epoch_discrepancy_and_relative_epoch_change",
                "iteration_unit": "epochs",
                "subset_count": len(subsets),
                "internal_subset_updates": subset_updates,
                "partial_epoch": partial_epoch,
                **parameters,
            },
        )

    endpoint_prediction = prediction

    endpoint_confirmation: dict[str, Any] = {"requested": False}
    if _policy_active(control) and (not cancelled):

        endpoint_prediction = _finish_prediction(operator, x)

        if not _finite_tensor(endpoint_prediction):
            recorder.mark_numerical_error("non_finite_endpoint_prediction")
        else:
            endpoint_residual = endpoint_prediction - measurement
            endpoint_value = _normalized(
                _global_norm(endpoint_residual), measurement_norm
            )
            trajectory_value = _normalized(
                _global_norm(prediction - measurement), measurement_norm
            )
            endpoint_confirmation = {
                "requested": True,
                "independent": True,
                "complete_epoch": True,
                "trajectory_normalized_data_residual": trajectory_value,
                "endpoint_normalized_data_residual": endpoint_value,
                "trajectory_endpoint_consistent": abs(trajectory_value - endpoint_value)
                <= 1e-07 + 0.0001 * max(abs(trajectory_value), abs(endpoint_value)),
            }
    if recorder.numerical_failure:
        return recorder.finish(
            x,
            actual_iterations=actual,
            status="numerical_error",
            stopping_reason=recorder.numerical_error_reason
            or "non_finite_solver_state",
            predicted_measurement=endpoint_prediction,
            resources={}
           ,
            metadata={
                "criterion": "complete_epoch_discrepancy_and_relative_epoch_change",
                "endpoint_confirmation": endpoint_confirmation,
                "iteration_unit": "epochs",
                "subset_count": len(subsets),
                "internal_subset_updates": subset_updates,
                "partial_epoch": partial_epoch,
                **parameters,
            },
        )
    final_residual_tensor = endpoint_prediction - measurement
    final_residual = _normalized(_global_norm(final_residual_tensor), measurement_norm)
    final_objective = _objective(final_residual_tensor)
    status = terminal_status or _finish_status(
        actual=actual,
        limit=limit,
        converged=converged,
        cancelled=cancelled,
        numerical_failure=recorder.numerical_failure,
    )
    return recorder.finish(
        x,
        actual_iterations=actual,
        status=status,
        stopping_reason=(
            terminal_reason
            if terminal_reason
            else (
                "discrepancy_and_relative_epoch_change_patience"
                if converged and _policy_active(control)
                else (
                    "complete_sweep_residual_and_iterate_tolerance"
                    if converged
                    else "callback_cancelled" if cancelled else "maximum_epochs_reached"
                )
            )
        ),
        final_residual=final_residual,
        final_objective=final_objective,
        predicted_measurement=endpoint_prediction,
        resources={}
       ,
        metadata={
            "complete_sweep_count": actual,
            "subset_count": len(subsets),
            "iteration_unit": "epochs",
            "internal_subset_updates": subset_updates,
            "partial_epoch": partial_epoch,
            "endpoint_confirmation": endpoint_confirmation,
            "criterion": "complete_epoch_discrepancy_and_relative_epoch_change",
            "max_iterations": limit,
            **parameters,
        },
    )


def solve_os_sart_detailed(
    operator: LinearOperator,
    measurement: torch.Tensor,
    *,
    num_iterations: int = 100,
    block_size: int | None = None,
    subset_indices: Optional[Iterable[Sequence[int]]] = None,
    order_strategy: str = "ordered",
    seed: int | None = None,
    relaxation: float = 1.0,
    min_value: float | None = None,
    max_value: float | None = None,
    x_init: torch.Tensor | None = None,
    eps: float = 1e-8,
    control: SolveControl | None = None,
    callback: Callback | None = None,
) -> SolveResult:
    if block_size is None and subset_indices is None:
        block_size = max(int(operator.range_shape[-2]) // 10, 1)
    return _subset_solver_detailed(
        "os_sart", operator, measurement, num_iterations=num_iterations, block_size=block_size,
        subset_indices=subset_indices, order_strategy=order_strategy, seed=seed,
        relaxation=relaxation, min_value=min_value, max_value=max_value, x_init=x_init,
        eps=eps, control=control, callback=callback,
    )


def solve_osem_detailed(
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
    from inv_framework.solvers.statistical import _solve_em_detailed
    return _solve_em_detailed(
        "osem", operator, measurement,
        num_iterations=num_iterations,
        block_size=block_size,
        subset_indices=subset_indices,
        order_strategy=order_strategy,
        seed=seed,
        x_init=x_init,
        initial_value=initial_value,
        min_value=min_value,
        max_value=max_value,
        eps=eps,
        control=control,
        callback=callback,
    )



def _indices_to_tensor(
    indices: Sequence[int],
    num_angles: int,
    device: Optional[torch.device],
) -> torch.Tensor:
    idx = torch.as_tensor(indices, dtype=torch.long, device=device)
    if idx.ndim != 1 or idx.numel() == 0:
        raise ValueError("Each subset index group must be a non-empty 1D sequence.")
    if torch.any(idx < 0) or torch.any(idx >= int(num_angles)):
        raise ValueError(
            f"Subset indices must be in [0, {int(num_angles) - 1}]; "
            f"got {idx.detach().cpu().tolist()}."
        )
    return idx


def make_angle_subsets(
    num_angles: int,
    block_size: Optional[int] = None,
    subset_indices: Optional[Iterable[Sequence[int]]] = None,
    order_strategy: str = "ordered",
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> List[torch.Tensor]:
    """Build angle subset index tensors without mutating global RNG state."""
    num_angles = int(num_angles)
    if num_angles <= 0:
        raise ValueError("num_angles must be positive.")

    if subset_indices is not None:
        subsets = [
            _indices_to_tensor(indices, num_angles=num_angles, device=device)
            for indices in subset_indices
        ]
        if not subsets:
            raise ValueError("subset_indices must contain at least one subset.")
        return subsets

    if block_size is None:
        block_size = num_angles
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    indices = torch.arange(num_angles, dtype=torch.long, device=device)
    if order_strategy == "ordered":
        pass
    elif order_strategy == "random":
        generator = torch.Generator(device=indices.device)
        if seed is not None:
            generator.manual_seed(int(seed))
        indices = indices[torch.randperm(num_angles, device=indices.device, generator=generator)]
    else:
        raise ValueError(
            f"Unknown order_strategy {order_strategy!r}; expected 'ordered' or 'random'."
        )
    return [indices[i : i + block_size] for i in range(0, num_angles, block_size)]


def select_measurement_subset(measurement: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Select projection-angle rows from a batched sinogram-like tensor."""
    return measurement.index_select(dim=-2, index=indices.to(device=measurement.device))


def make_subset_operator(operator: LinearOperator, indices: torch.Tensor) -> LinearOperator:
    """Return a projection-angle subset operator when the backend supports it."""
    subset = getattr(operator, "subset", None)
    if callable(subset):
        sub_operator = subset(indices)
        if not isinstance(sub_operator, LinearOperator):
            raise TypeError(
                "operator.subset(indices) must return a LinearOperator; "
                f"got {type(sub_operator).__name__}."
            )
        return sub_operator

    from inv_framework.ct.backends.radon_torch import ParallelBeamRadon2D

    if isinstance(operator, ParallelBeamRadon2D):
        idx = indices.to(device=operator.angles.device, dtype=torch.long)
        angles = operator.angles.index_select(0, idx)
        return ParallelBeamRadon2D(
            image_size=int(operator.image_size),
            angles=angles,
            device=operator.angles.device,
            in_channels=int(operator.domain_shape[0]),
        )

    raise NotImplementedError(
        f"{type(operator).__name__} does not provide subset(indices); "
        "ordered-subset solvers require a projection-angle subset backend."
    )

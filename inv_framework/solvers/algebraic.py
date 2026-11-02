"""Row-action and Krylov reconstruction methods."""
from __future__ import annotations
from inv_framework.core.operator import ForwardOperator, LinearOperator
from inv_framework.solvers.base import InverseProblemSolver, SolveControl, SolveResult, ConsecutiveStoppingMonitor, IterationRecorder, resolve_control
from typing import Iterable, Optional, Sequence, Any
from inv_framework.solvers._utils import apply_box_constraints, prepare_initial_image, require_linear_operator, validate_measurement_shape
from inv_framework.solvers.subset import make_angle_subsets, make_subset_operator, select_measurement_subset
import torch


def sirt(operator: LinearOperator,
         y: torch.Tensor,
         num_iterations: int = 100,
         min_value: float = None,
         max_value: float = None,
         x_init: torch.Tensor = None) -> torch.Tensor:
    """SIRT with row/column weighting."""
    require_linear_operator(operator, "sirt")
    device = y.device
    B = y.shape[0]
    domain = (B, *operator.domain_shape)
    range_ = (B, *operator.range_shape)

    R = operator.forward(torch.ones(domain, device=device))
    R = torch.where(R < 1e-8, torch.full_like(R, float('inf')), R).reciprocal()
    C = operator.adjoint(torch.ones(range_, device=device))
    C = torch.where(C < 1e-8, torch.full_like(C, float('inf')), C).reciprocal()

    x = (torch.zeros(domain, device=device)
         if x_init is None else x_init.clone())
    for _ in range(num_iterations):
        residual = operator.forward(x) - y
        x = x - C * operator.adjoint(R * residual)
        if min_value is not None or max_value is not None:
            x = x.clamp(min_value, max_value)
    return x


def landweber(operator: LinearOperator,
              y: torch.Tensor,
              num_iterations: int = 100,
              step_size: float = None,
              x_init: torch.Tensor = None,
              min_value: float = None,
              max_value: float = None) -> torch.Tensor:
    """Landweber: x <- x - alpha A^T (A x - y)."""
    require_linear_operator(operator, "landweber")
    device = y.device
    B = y.shape[0]
    domain = (B, *operator.domain_shape)

    if step_size is None:
        step_size = 1e-3

    x = (torch.zeros(domain, device=device)
         if x_init is None else x_init.clone())
    for _ in range(num_iterations):
        residual = operator.forward(x) - y
        x = x - step_size * operator.adjoint(residual)
        if min_value is not None or max_value is not None:
            x = x.clamp(min_value, max_value)
    return x


class SIRTSolver(InverseProblemSolver):
    def __init__(self,
                 num_iterations: int = 100,
                 min_value: float = None,
                 max_value: float = None):
        self.num_iterations = num_iterations
        self.min_value = min_value
        self.max_value = max_value

    def solve(self, measurement, operator, x_init=None, **kwargs):
        return sirt(operator, measurement,
                    num_iterations=self.num_iterations,
                    min_value=self.min_value,
                    max_value=self.max_value,
                    x_init=x_init)

    def solve_detailed(self, measurement, operator, x_init=None, *, control=None, callback=None, **kwargs):
        pass

        return solve_sirt_detailed(
            operator, measurement,
            num_iterations=kwargs.pop("num_iterations", self.num_iterations),
            min_value=kwargs.pop("min_value", self.min_value),
            max_value=kwargs.pop("max_value", self.max_value),
            x_init=x_init, control=control, callback=callback,
        )


class LandweberSolver(InverseProblemSolver):
    def __init__(self,
                 num_iterations: int = 100,
                 step_size: float = None,
                 min_value: float = None,
                 max_value: float = None):
        self.num_iterations = num_iterations
        self.step_size = step_size
        self.min_value = min_value
        self.max_value = max_value

    def solve(self, measurement, operator, x_init=None, **kwargs):
        return landweber(operator, measurement,
                         num_iterations=self.num_iterations,
                         step_size=self.step_size,
                         x_init=x_init,
                         min_value=self.min_value,
                         max_value=self.max_value)

    def solve_detailed(self, measurement, operator, x_init=None, *, control=None, callback=None, **kwargs):
        pass

        return solve_landweber_detailed(
            operator, measurement,
            num_iterations=kwargs.pop("num_iterations", self.num_iterations),
            step_size=kwargs.pop("step_size", self.step_size),
            x_init=x_init,
            min_value=kwargs.pop("min_value", self.min_value),
            max_value=kwargs.pop("max_value", self.max_value),
            control=control, callback=callback,
        )


def _ct_batch_sqnorm(z: torch.Tensor) -> torch.Tensor:
    return (z.reshape(z.shape[0], -1) ** 2).sum(dim=1)


def _ct_batch_norm(z: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(_ct_batch_sqnorm(z).clamp_min(0.0))


def _ct_batch_view(coeff: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return coeff.reshape((coeff.shape[0],) + (1,) * (target.ndim - 1))


def cgls(operator: LinearOperator,
         y: torch.Tensor,
         num_iterations: int = 10,
         tol: float = 1e-6,
         x_init: torch.Tensor = None,
         min_value: float = None,
         max_value: float = None,
         eps: float = 1e-12) -> torch.Tensor:
    """Conjugate-gradient least squares for min_x ||A x - y||_2^2."""

    require_linear_operator(operator, "cgls")
    validate_measurement_shape(y, operator, "cgls")

    x = prepare_initial_image(y, operator, x_init=x_init, initial_value=0.0)
    r = y - operator.forward(x)
    s = operator.adjoint(r)
    p = s.clone()
    gamma = _ct_batch_sqnorm(s)

    for _ in range(int(num_iterations)):
        if torch.all(torch.sqrt(gamma.clamp_min(0.0)) <= float(tol)):
            break
        q = operator.forward(p)
        delta = _ct_batch_sqnorm(q).clamp_min(float(eps))
        alpha = gamma / delta
        x = x + _ct_batch_view(alpha, x) * p
        x = apply_box_constraints(x, min_value=min_value, max_value=max_value)
        r = y - operator.forward(x)
        s_new = operator.adjoint(r)
        gamma_new = _ct_batch_sqnorm(s_new)
        beta = gamma_new / gamma.clamp_min(float(eps))
        p = s_new + _ct_batch_view(beta, p) * p
        gamma = gamma_new
    return x


def lsqr(operator: LinearOperator,
         y: torch.Tensor,
         num_iterations: int = 10,
         damping: float = 0.0,
         atol: float = 1e-6,
         btol: float = 1e-6,
         x_init: torch.Tensor = None,
         min_value: float = None,
         max_value: float = None,
         eps: float = 1e-12) -> torch.Tensor:
    """LSQR using Golub-Kahan bidiagonalization."""

    require_linear_operator(operator, "lsqr")
    validate_measurement_shape(y, operator, "lsqr")

    eps = float(eps)
    damping = float(damping)
    x = prepare_initial_image(y, operator, x_init=x_init, initial_value=0.0)
    rhs_norm = _ct_batch_norm(y)

    u = y - operator.forward(x)
    beta = _ct_batch_norm(u)
    u = u / _ct_batch_view(beta.clamp_min(eps), u)

    v = operator.adjoint(u)
    alpha = _ct_batch_norm(v)
    v = v / _ct_batch_view(alpha.clamp_min(eps), v)

    w = v.clone()
    phi_bar = beta.clone()
    rho_bar = alpha.clone()

    for _ in range(int(num_iterations)):
        residual_norm = _ct_batch_norm(y - operator.forward(x))
        stop_tol = float(atol) * rhs_norm + float(btol)
        if torch.all(residual_norm <= stop_tol):
            break

        u = operator.forward(v) - _ct_batch_view(alpha, u) * u
        beta = _ct_batch_norm(u)
        u = u / _ct_batch_view(beta.clamp_min(eps), u)

        v = operator.adjoint(u) - _ct_batch_view(beta, v) * v
        alpha = _ct_batch_norm(v)
        v = v / _ct_batch_view(alpha.clamp_min(eps), v)

        if damping > 0.0:
            rho_damped = torch.sqrt(rho_bar * rho_bar + damping * damping).clamp_min(eps)
            phi_bar = (rho_bar / rho_damped) * phi_bar
            rho_bar = rho_damped

        rho = torch.sqrt(rho_bar * rho_bar + beta * beta).clamp_min(eps)
        c = rho_bar / rho
        s = beta / rho
        theta = s * alpha
        rho_bar = -c * alpha
        phi = c * phi_bar
        phi_bar = s * phi_bar

        x = x + _ct_batch_view(phi / rho, x) * w
        x = apply_box_constraints(x, min_value=min_value, max_value=max_value)
        w = v - _ct_batch_view(theta / rho, w) * w
    return x


class CGLSSolver(InverseProblemSolver):
    def __init__(self,
                 num_iterations: int = 10,
                 tol: float = 1e-6,
                 min_value: float = None,
                 max_value: float = None,
                 regularizer='none', strength=0., fidelity=None):
        self.num_iterations = num_iterations
        self.tol = tol
        self.min_value = min_value
        self.max_value = max_value
        self.regularizer, self.strength, self.fidelity = regularizer, strength, fidelity

    def solve(self, measurement, operator, x_init=None, **kwargs):
        if self.fidelity is not None or self.strength:
            return self.solve_detailed(measurement, operator, x_init=x_init, **kwargs).reconstruction
        return cgls(operator, measurement,
                    num_iterations=self.num_iterations,
                    tol=self.tol,
                    x_init=x_init,
                    min_value=self.min_value,
                    max_value=self.max_value)

    def solve_detailed(self, measurement, operator, x_init=None, *, control=None, callback=None, **kwargs):
        pass
        if self.fidelity is not None or self.strength or kwargs.get('fidelity') is not None:
            pass
            return solve_krylov_composite('cgls', measurement, operator, regularizer=self.regularizer,
                strength=self.strength, fidelity=kwargs.pop('fidelity', self.fidelity),
                num_iterations=kwargs.pop('num_iterations', self.num_iterations), tol=kwargs.pop('tol', self.tol),
                min_value=self.min_value, max_value=self.max_value, x_init=x_init, control=control, callback=callback)
        return solve_cgls_detailed(
            operator, measurement,
            num_iterations=kwargs.pop("num_iterations", self.num_iterations),
            tol=kwargs.pop("tol", self.tol),
            x_init=x_init,
            min_value=kwargs.pop("min_value", self.min_value),
            max_value=kwargs.pop("max_value", self.max_value),
            operator_norm_estimate=kwargs.pop("operator_norm_estimate", None),
            operator_norm_squared=kwargs.pop("operator_norm_squared", None),
            control=control, callback=callback,
        )


class LSQRSolver(InverseProblemSolver):
    def __init__(self,
                 num_iterations: int = 10,
                 damping: float = 0.0,
                 atol: float = 1e-6,
                 btol: float = 1e-6,
                 min_value: float = None,
                 max_value: float = None,
                 regularizer='none', strength=0., fidelity=None):
        self.num_iterations = num_iterations
        self.damping = damping
        self.atol = atol
        self.btol = btol
        self.min_value = min_value
        self.max_value = max_value
        self.regularizer, self.strength, self.fidelity = regularizer, strength, fidelity

    def solve(self, measurement, operator, x_init=None, **kwargs):
        if self.fidelity is not None or self.strength:
            return self.solve_detailed(measurement, operator, x_init=x_init, **kwargs).reconstruction
        return lsqr(operator, measurement,
                    num_iterations=self.num_iterations,
                    damping=self.damping,
                    atol=self.atol,
                    btol=self.btol,
                    x_init=x_init,
                    min_value=self.min_value,
                    max_value=self.max_value)

    def solve_detailed(self, measurement, operator, x_init=None, *, control=None, callback=None, **kwargs):
        pass
        if self.fidelity is not None or self.strength or kwargs.get('fidelity') is not None:
            pass
            return solve_krylov_composite('lsqr', measurement, operator, regularizer=self.regularizer,
                strength=self.strength, fidelity=kwargs.pop('fidelity', self.fidelity), damping=self.damping,
                num_iterations=kwargs.pop('num_iterations', self.num_iterations), atol=self.atol, btol=self.btol,
                min_value=self.min_value, max_value=self.max_value, x_init=x_init, control=control, callback=callback)
        return solve_lsqr_detailed(
            operator, measurement,
            num_iterations=kwargs.pop("num_iterations", self.num_iterations),
            damping=kwargs.pop("damping", self.damping),
            atol=kwargs.pop("atol", self.atol),
            btol=kwargs.pop("btol", self.btol),
            x_init=x_init,
            min_value=kwargs.pop("min_value", self.min_value),
            max_value=kwargs.pop("max_value", self.max_value),
            operator_norm_estimate=kwargs.pop("operator_norm_estimate", None),
            operator_norm_squared=kwargs.pop("operator_norm_squared", None),
            control=control, callback=callback,
        )


def sart(operator: LinearOperator,
         y: torch.Tensor,
         num_iterations: int = 100,
         block_size: int = 1,
         subset_indices: Optional[Iterable[Sequence[int]]] = None,
         order_strategy: str = "ordered",
         seed: Optional[int] = None,
         relaxation: float = 1.0,
         min_value: float = None,
         max_value: float = None,
         x_init: torch.Tensor = None,
         eps: float = 1e-8) -> torch.Tensor:
    """SART update using projection-angle subset operators."""
    from inv_framework.solvers.subset import _safe_reciprocal
    require_linear_operator(operator, "sart")
    batch = validate_measurement_shape(y, operator, "sart")
    x = prepare_initial_image(y, operator, x_init=x_init, initial_value=0.0)
    domain_shape = (batch, *tuple(operator.domain_shape))
    subsets = make_angle_subsets(
        num_angles=operator.range_shape[-2],
        block_size=block_size,
        subset_indices=subset_indices,
        order_strategy=order_strategy,
        seed=seed,
        device=y.device,
    )

    ones_domain = torch.ones(domain_shape, device=y.device, dtype=y.dtype)
    for _ in range(int(num_iterations)):
        for indices in subsets:
            sub_operator = make_subset_operator(operator, indices)
            y_sub = select_measurement_subset(y, indices)
            ones_range = torch.ones(
                (batch, *tuple(sub_operator.range_shape)),
                device=y.device,
                dtype=y.dtype,
            )
            row_weight = _safe_reciprocal(sub_operator.forward(ones_domain), float(eps))
            col_weight = _safe_reciprocal(sub_operator.adjoint(ones_range), float(eps))
            residual = sub_operator.forward(x) - y_sub
            x = x - float(relaxation) * col_weight * sub_operator.adjoint(row_weight * residual)
            x = apply_box_constraints(x, min_value=min_value, max_value=max_value)
    return x


class SARTSolver(InverseProblemSolver):
    def __init__(self,
                 num_iterations: int = 100,
                 block_size: int = 1,
                 subset_indices: Optional[Iterable[Sequence[int]]] = None,
                 order_strategy: str = "ordered",
                 seed: Optional[int] = None,
                 relaxation: float = 1.0,
                 min_value: float = None,
                 max_value: float = None,
                 eps: float = 1e-8):
        from inv_framework.solvers.subset import _store_subset_indices
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
        return sart(
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

        return solve_sart_detailed(
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


def solve_sirt_detailed(
    operator: LinearOperator,
    measurement: torch.Tensor,
    *,
    num_iterations: int = 100,
    min_value: float | None = None,
    max_value: float | None = None,
    x_init: torch.Tensor | None = None,
    control: SolveControl | None = None,
    callback: Callback | None = None,
) -> SolveResult:
    from inv_framework.solvers._iteration import _common_parameter_errors, _finish_prediction, _finish_status, _finite_scalar, _finite_tensor, _global_norm, _native_tolerance, _normalized, _numerical_result, _objective, _parameter_failure_result, _policy_active, _policy_status, _relative_change, _row_action_criteria, _safe_record, _tolerance_reached
    require_linear_operator(operator, "sirt")
    batch = validate_measurement_shape(measurement, operator, "sirt")
    parameter_errors = _common_parameter_errors(num_iterations, min_value, max_value)
    if parameter_errors:
        return _parameter_failure_result(
            "sirt", measurement, operator, x_init=x_init, errors=parameter_errors,
            max_iterations=None,
            parameters={
                "num_iterations": num_iterations,
                "min_value": min_value,
                "max_value": max_value,
            },
        )
    control = resolve_control(
        control, default_iterations=int(num_iterations), default_tolerance=1e-5, callback=callback
    )
    limit = min(int(num_iterations), int(control.max_iterations or num_iterations))
    parameters = {
        "num_iterations": int(num_iterations),
        "min_value": min_value,
        "max_value": max_value,
    }
    if not _finite_tensor(measurement):
        return _numerical_result(
            "sirt", measurement, operator, x_init=x_init, max_iterations=limit,
            reason="non_finite_measurement", parameters=parameters,
        )
    x = prepare_initial_image(measurement, operator, x_init=x_init, initial_value=0.0)
    if not _finite_tensor(x):
        return _numerical_result(
            "sirt", measurement, operator, x_init=x_init, max_iterations=limit,
            reason="non_finite_initial_reconstruction", parameters=parameters,
        )
    recorder = IterationRecorder(control, measurement, operator, algorithm="sirt")
    monitor = ConsecutiveStoppingMonitor(control)
    recorder.set_initial(x)
    domain = (batch, *tuple(operator.domain_shape))
    range_ = (batch, *tuple(operator.range_shape))

    row_weight = operator.forward(torch.ones(domain, device=measurement.device, dtype=measurement.dtype))
    row_weight = torch.where(row_weight < 1e-8, torch.full_like(row_weight, float("inf")), row_weight).reciprocal()
    if not _finite_tensor(row_weight):
        recorder.mark_numerical_error("non_finite_row_normalization")
        return recorder.finish(
            x, actual_iterations=0, status="numerical_error",
            stopping_reason="non_finite_row_normalization",

            metadata={"criterion": "normalized_data_residual_and_relative_iterate_change", **parameters},
        )
    column_weight = operator.adjoint(torch.ones(range_, device=measurement.device, dtype=measurement.dtype))
    column_weight = torch.where(column_weight < 1e-8, torch.full_like(column_weight, float("inf")), column_weight).reciprocal()
    if not _finite_tensor(column_weight):
        recorder.mark_numerical_error("non_finite_column_normalization")
        return recorder.finish(
            x, actual_iterations=0, status="numerical_error",
            stopping_reason="non_finite_column_normalization",

            metadata={"criterion": "normalized_data_residual_and_relative_iterate_change", **parameters},
        )
    measurement_norm = _global_norm(measurement)
    actual = 0
    converged = False
    cancelled = False
    terminal_status = None
    terminal_reason = None
    prediction = operator.forward(x)
    residual = prediction - measurement
    if not _finite_tensor(prediction) or not _finite_tensor(residual):
        recorder.mark_numerical_error("non_finite_initial_residual")
        return recorder.finish(
            x, actual_iterations=0, status="numerical_error",
            stopping_reason="non_finite_initial_residual",
            predicted_measurement=prediction,

            metadata={"criterion": "normalized_data_residual_and_relative_iterate_change", **parameters},
        )
    native_tolerance = _native_tolerance(control)
    for iteration in range(1, limit + 1):
        previous = x
        x = x - column_weight * operator.adjoint(row_weight * residual)
        x = apply_box_constraints(x, min_value=min_value, max_value=max_value)
        actual = iteration
        if not _finite_tensor(x):
            recorder.mark_numerical_error("non_finite_reconstruction")
            recorder.record(
                iteration, x, metadata={"state": "post_update", "checked": False, "non_finite_field": "reconstruction"}
            )
            break
        prediction = operator.forward(x)
        residual = prediction - measurement
        if not _finite_tensor(prediction):
            recorder.mark_numerical_error("non_finite_prediction")
            recorder.record(
                iteration, x, residual=residual,
                metadata={"state": "post_update", "checked": False, "non_finite_field": "prediction"}
            )
            break
        if not _finite_tensor(residual):
            recorder.mark_numerical_error("non_finite_residual")
            recorder.record(
                iteration, x, residual=residual,
                metadata={"state": "post_update", "checked": False, "non_finite_field": "residual"}
            )
            break
        change = _relative_change(previous, x)
        residual_value = _normalized(_global_norm(residual), measurement_norm)
        if not _finite_scalar(change):
            recorder.mark_numerical_error("non_finite_relative_iterate_change")
            recorder.record(
                iteration, x, residual=residual,
                metadata={"state": "post_update", "checked": False, "non_finite_field": "relative_iterate_change"}
            )
            break
        if not _finite_scalar(residual_value):
            recorder.mark_numerical_error("non_finite_discrepancy")
            recorder.record(
                iteration, x, residual=residual,
                metadata={"state": "post_update", "checked": False, "non_finite_field": "discrepancy"}
            )
            break
        criteria, native_tolerance = _row_action_criteria(
            control, residual_value, change, epoch=False
        )
        decision = monitor.observe(iteration, criteria=criteria, relative_change=change, monitor_value=residual_value)
        converged = decision.converged if _policy_active(control) else _tolerance_reached(control, residual=residual_value, change=change)
        terminal_status, terminal_reason = _policy_status(decision) if _policy_active(control) else (None, None)
        objective = _objective(residual)
        if not _finite_scalar(objective):
            recorder.mark_numerical_error("non_finite_objective")
            recorder.record(
                iteration, x, residual=residual,
                metadata={"state": "post_update", "checked": decision.checked, "non_finite_field": "objective"}
            )
            break
        if not _safe_record(
            recorder,
            iteration,
            x,
            residual=residual,
            objective=objective,
            algorithm_residual=residual_value,
            stopping_candidate=converged,
            consecutive_criteria_count=decision.consecutive,
            criteria=criteria,
            native_criterion_name="relative_iterate_change",
            native_criterion_value=change,
            native_criterion_threshold=native_tolerance,
            metadata={"state": "post_update", "checked": decision.checked, "unit": "iterations"},
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
            stopping_reason=recorder.numerical_error_reason or "non_finite_solver_state",
            predicted_measurement=prediction,
            metadata={
                "criterion": "normalized_data_residual_and_relative_iterate_change",
                "iteration_unit": "iterations",
                **parameters,
            },
        )

    endpoint_prediction = prediction

    endpoint_confirmation: dict[str, Any] = {"requested": False}
    if _policy_active(control) and not cancelled:

        endpoint_prediction = _finish_prediction(operator, x)

        if not _finite_tensor(endpoint_prediction):
            recorder.mark_numerical_error("non_finite_endpoint_prediction")
        else:
            endpoint_residual = endpoint_prediction - measurement
            endpoint_value = _normalized(_global_norm(endpoint_residual), measurement_norm)
            trajectory_value = _normalized(_global_norm(prediction - measurement), measurement_norm)
            endpoint_confirmation = {
                "requested": True,
                "independent": True,
                "trajectory_normalized_data_residual": trajectory_value,
                "endpoint_normalized_data_residual": endpoint_value,
                "trajectory_endpoint_consistent": (
                    abs(trajectory_value - endpoint_value)
                    <= 1e-7 + 1e-4 * max(abs(trajectory_value), abs(endpoint_value))
                ),
            }
    if recorder.numerical_failure:
        return recorder.finish(
            x,
            actual_iterations=actual,
            status="numerical_error",
            stopping_reason=recorder.numerical_error_reason or "non_finite_solver_state",
            predicted_measurement=endpoint_prediction,
            resources={},
            metadata={
                "criterion": "normalized_data_residual_and_relative_iterate_change",
                "endpoint_confirmation": endpoint_confirmation,
                "iteration_unit": "iterations",
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
    if actual == 0 and not cancelled:
        status = "partial"
    return recorder.finish(
        x,
        actual_iterations=actual,
        status=status,
        stopping_reason=(
            terminal_reason if terminal_reason else
            "discrepancy_and_relative_iterate_change_patience" if converged and _policy_active(control) else
            "relative_residual_and_iterate_tolerance" if converged else
            "callback_cancelled" if cancelled else
            "maximum_iterations_reached"
        ),
        final_residual=final_residual,
        final_objective=final_objective,
        predicted_measurement=endpoint_prediction,
        resources={},
        metadata={
            "criterion": "normalized_data_residual_and_relative_iterate_change",
            "iteration_unit": "iterations",
            "endpoint_confirmation": endpoint_confirmation,
            "max_iterations": limit,
            **parameters,
        },
    )


def solve_landweber_detailed(
    operator: LinearOperator,
    measurement: torch.Tensor,
    *,
    num_iterations: int = 100,
    step_size: float | None = None,
    x_init: torch.Tensor | None = None,
    min_value: float | None = None,
    max_value: float | None = None,
    control: SolveControl | None = None,
    callback: Callback | None = None,
) -> SolveResult:
    from inv_framework.solvers._iteration import _common_parameter_errors, _finish_prediction, _finish_status, _finite_scalar, _finite_tensor, _global_norm, _landweber_step_preflight, _native_tolerance, _normalized, _numerical_result, _objective, _parameter_failure_result, _policy_active, _policy_status, _relative_change, _row_action_criteria, _safe_record, _tolerance_reached
    require_linear_operator(operator, "landweber")
    validate_measurement_shape(measurement, operator, "landweber")
    parameter_errors = _common_parameter_errors(num_iterations, min_value, max_value)
    if parameter_errors:
        return _parameter_failure_result(
            "landweber", measurement, operator, x_init=x_init, errors=parameter_errors,
            max_iterations=None,
            parameters={
                "num_iterations": num_iterations,
                "step_size": step_size,
                "min_value": min_value,
                "max_value": max_value,
            },
        )
    control = resolve_control(
        control, default_iterations=int(num_iterations), default_tolerance=1e-5, callback=callback
    )
    limit = min(int(num_iterations), int(control.max_iterations or num_iterations))
    parameters = {
        "num_iterations": int(num_iterations),
        "step_size": 1e-3 if step_size is None else step_size,
        "min_value": min_value,
        "max_value": max_value,
    }
    if not _finite_tensor(measurement):
        return _numerical_result(
            "landweber", measurement, operator, x_init=x_init, max_iterations=limit,
            reason="non_finite_measurement", parameters=parameters,
        )
    x = prepare_initial_image(measurement, operator, x_init=x_init, initial_value=0.0)
    if not _finite_tensor(x):
        return _numerical_result(
            "landweber", measurement, operator, x_init=x_init, max_iterations=limit,
            reason="non_finite_initial_reconstruction", parameters=parameters,
        )
    step, norm_squared, step_errors, norm_estimates = _landweber_step_preflight(
        operator, measurement, control, parameters["step_size"]
    )
    if step_errors:
        return _parameter_failure_result(
            "landweber", measurement, operator, x_init=x_init, errors=step_errors,
            max_iterations=limit,
            parameters={**parameters, "step_size": step},
            metadata={"operator_norm_squared": norm_squared, **norm_estimates},
        )
    parameters["step_size"] = step
    x = prepare_initial_image(measurement, operator, x_init=x_init, initial_value=0.0)
    recorder = IterationRecorder(control, measurement, operator, algorithm="landweber")
    monitor = ConsecutiveStoppingMonitor(control)
    recorder.set_initial(x)

    measurement_norm = _global_norm(measurement)
    actual = 0
    converged = False
    cancelled = False
    terminal_status = None
    terminal_reason = None
    prediction = operator.forward(x)
    residual = prediction - measurement
    if not _finite_tensor(prediction) or not _finite_tensor(residual):
        recorder.mark_numerical_error("non_finite_initial_residual")
        return recorder.finish(
            x, actual_iterations=0, status="numerical_error",
            stopping_reason="non_finite_initial_residual",
            predicted_measurement=prediction,
            resources={"parameter_estimation": norm_estimates},
            metadata={"criterion": "normalized_data_residual_and_relative_iterate_change", **parameters, **norm_estimates},
        )
    native_tolerance = _native_tolerance(control)
    for iteration in range(1, limit + 1):
        previous = x
        x = x - step * operator.adjoint(residual)
        x = apply_box_constraints(x, min_value=min_value, max_value=max_value)
        actual = iteration
        if not _finite_tensor(x):
            recorder.mark_numerical_error("non_finite_reconstruction")
            recorder.record(
                iteration, x, metadata={"state": "post_update", "checked": False, "non_finite_field": "reconstruction"}
            )
            break
        prediction = operator.forward(x)
        residual = prediction - measurement
        if not _finite_tensor(prediction):
            recorder.mark_numerical_error("non_finite_prediction")
            recorder.record(
                iteration, x, residual=residual,
                metadata={"state": "post_update", "checked": False, "non_finite_field": "prediction"}
            )
            break
        if not _finite_tensor(residual):
            recorder.mark_numerical_error("non_finite_residual")
            recorder.record(
                iteration, x, residual=residual,
                metadata={"state": "post_update", "checked": False, "non_finite_field": "residual"}
            )
            break
        change = _relative_change(previous, x)
        residual_value = _normalized(_global_norm(residual), measurement_norm)
        if not _finite_scalar(change):
            recorder.mark_numerical_error("non_finite_relative_iterate_change")
            recorder.record(
                iteration, x, residual=residual,
                metadata={"state": "post_update", "checked": False, "non_finite_field": "relative_iterate_change"}
            )
            break
        if not _finite_scalar(residual_value):
            recorder.mark_numerical_error("non_finite_discrepancy")
            recorder.record(
                iteration, x, residual=residual,
                metadata={"state": "post_update", "checked": False, "non_finite_field": "discrepancy"}
            )
            break
        criteria, native_tolerance = _row_action_criteria(
            control, residual_value, change, epoch=False
        )
        decision = monitor.observe(iteration, criteria=criteria, relative_change=change, monitor_value=residual_value)
        converged = decision.converged if _policy_active(control) else _tolerance_reached(control, residual=residual_value, change=change)
        terminal_status, terminal_reason = _policy_status(decision) if _policy_active(control) else (None, None)
        objective = _objective(residual)
        if not _finite_scalar(objective):
            recorder.mark_numerical_error("non_finite_objective")
            recorder.record(
                iteration, x, residual=residual,
                metadata={"state": "post_update", "checked": decision.checked, "non_finite_field": "objective"}
            )
            break
        if not _safe_record(
            recorder,
            iteration,
            x,
            residual=residual,
            objective=objective,
            algorithm_residual=residual_value,
            step_size=step,
            stopping_candidate=converged,
            consecutive_criteria_count=decision.consecutive,
            criteria=criteria,
            native_criterion_name="relative_iterate_change",
            native_criterion_value=change,
            native_criterion_threshold=native_tolerance,
            metadata={"state": "post_update", "checked": decision.checked, "unit": "iterations"},
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
            stopping_reason=recorder.numerical_error_reason or "non_finite_solver_state",
            predicted_measurement=prediction,
            metadata={"criterion": "normalized_data_residual_and_relative_iterate_change", **parameters, **norm_estimates},
        )

    endpoint_prediction = prediction

    endpoint_confirmation: dict[str, Any] = {"requested": False}
    if _policy_active(control) and not cancelled:

        endpoint_prediction = _finish_prediction(operator, x)

        if not _finite_tensor(endpoint_prediction):
            recorder.mark_numerical_error("non_finite_endpoint_prediction")
        else:
            endpoint_residual = endpoint_prediction - measurement
            endpoint_value = _normalized(_global_norm(endpoint_residual), measurement_norm)
            trajectory_value = _normalized(_global_norm(prediction - measurement), measurement_norm)
            endpoint_confirmation = {
                "requested": True,
                "independent": True,
                "trajectory_normalized_data_residual": trajectory_value,
                "endpoint_normalized_data_residual": endpoint_value,
                "trajectory_endpoint_consistent": (
                    abs(trajectory_value - endpoint_value)
                    <= 1e-7 + 1e-4 * max(abs(trajectory_value), abs(endpoint_value))
                ),
            }
    if recorder.numerical_failure:
        return recorder.finish(
            x,
            actual_iterations=actual,
            status="numerical_error",
            stopping_reason=recorder.numerical_error_reason or "non_finite_solver_state",
            predicted_measurement=endpoint_prediction,
            resources={},
            metadata={"criterion": "normalized_data_residual_and_relative_iterate_change", "endpoint_confirmation": endpoint_confirmation, **parameters, **norm_estimates},
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
            terminal_reason if terminal_reason else
            "discrepancy_and_relative_iterate_change_patience" if converged and _policy_active(control) else
            "relative_residual_and_iterate_tolerance" if converged else
            "callback_cancelled" if cancelled else
            "maximum_iterations_reached"
        ),
        final_residual=final_residual,
        final_objective=final_objective,
        predicted_measurement=endpoint_prediction,
        resources={},
        metadata={
            "criterion": "normalized_data_residual_and_relative_iterate_change",
            "endpoint_confirmation": endpoint_confirmation,
            "max_iterations": limit,
            **parameters,
            **norm_estimates,
        },
    )


def solve_krylov_composite(
    algorithm,
    measurement,
    operator,
    *,
    regularizer="none",
    strength=0.0,
    fidelity=None,
    damping=0.0,
    **kwargs,
):
    """Run the original Krylov recurrence on a weighted augmented system."""
    from inv_framework.solvers._iteration import _finite_nonnegative
    from dataclasses import replace
    from inv_framework.core.fidelity import QuadraticFidelity
    from inv_framework.core.operator import AugmentedLeastSquaresOperator

    if regularizer not in {"none", "tikhonov", "h1"} or not _finite_nonnegative(
        strength
    ):
        raise ValueError(
            "Krylov requires none, identity or H1 and nonnegative strength"
        )
    if kwargs.get("min_value") is not None or kwargs.get("max_value") is not None:
        raise ValueError("augmented Krylov requires unconstrained reconstruction")
    if strength and damping:
        raise ValueError("explicit augmentation and LSQR damping cannot both be used")
    if damping:
        strength, regularizer = (damping**2, "tikhonov")
    if regularizer == "none" and strength:
        raise ValueError("nonzero strength requires an explicit regularizer")
    fidelity = fidelity if fidelity is not None else QuadraticFidelity(measurement)
    if not getattr(fidelity, "is_quadratic", False):
        raise ValueError("Krylov requires quadratic fidelity")
    augmented = AugmentedLeastSquaresOperator(operator, fidelity, strength, regularizer)
    function = solve_cgls_detailed if algorithm == "cgls" else solve_lsqr_detailed
    result = function(augmented, augmented.rhs(measurement), **kwargs)
    prediction = None
    objective = residual = None
    prediction = operator.forward(result.reconstruction)
    transformed = (
        augmented.transform.forward(result.reconstruction)
        if strength and augmented.transform
        else result.reconstruction
    )
    objective = float(fidelity.value(prediction)) + (
        strength / 2 * float(transformed.square().sum()) if strength else 0.0
    )
    residual = float(
        (prediction - measurement).norm() / measurement.norm().clamp_min(1e-12)
    )
    return replace(
        result,
        predicted_measurement=prediction,
        final_objective=objective,
        final_residual=residual,
        resources={
            **result.resources,
        },
        metadata={
            **result.metadata,
            "augmented_least_squares": True,
            "regularizer": regularizer,
            "strength": strength,
            "endpoint_diagnostics_complete": True,
            "uniqueness": "not_assumed; requires ker(B) intersect ker(L) = {0}",
        },
    )


def solve_cgls_detailed(
    operator: LinearOperator,
    measurement: torch.Tensor,
    *,
    num_iterations: int = 10,
    tol: float = 1e-6,
    x_init: torch.Tensor | None = None,
    min_value: float | None = None,
    max_value: float | None = None,
    eps: float = 1e-12,
    control: SolveControl | None = None,
    callback: Callback | None = None,
    operator_norm_estimate: float | None = None,
    operator_norm_squared: float | None = None,
) -> SolveResult:
    from inv_framework.solvers._iteration import _common_parameter_errors, _control_threshold, _finish_prediction, _finish_status, _finite_nonnegative, _finite_positive, _finite_tensor, _global_norm, _normalized, _numerical_result, _objective, _parameter_failure_result, _policy_active, _policy_status, _relative_change, _resolve_operator_norm_estimate, _safe_record, _sqnorm
    require_linear_operator(operator, "cgls")
    validate_measurement_shape(measurement, operator, "cgls")
    parameter_errors = _common_parameter_errors(num_iterations, min_value, max_value)
    if not _finite_nonnegative(tol):
        parameter_errors.append("tol must be a finite nonnegative number")
    if not _finite_positive(eps):
        parameter_errors.append("eps must be a finite positive number")
    if operator_norm_estimate is not None and not _finite_nonnegative(operator_norm_estimate):
        parameter_errors.append("operator_norm_estimate must be finite and nonnegative")
    if operator_norm_squared is not None and not _finite_nonnegative(operator_norm_squared):
        parameter_errors.append("operator_norm_squared must be finite and nonnegative")
    if parameter_errors:
        return _parameter_failure_result(
            "cgls", measurement, operator, x_init=x_init,
            errors=parameter_errors,
            max_iterations=num_iterations if isinstance(num_iterations, int) and not isinstance(num_iterations, bool) else None,
            parameters={
                "num_iterations": num_iterations,
                "tol": tol,
                "min_value": min_value,
                "max_value": max_value,
                "eps": eps,
                "operator_norm_estimate": operator_norm_estimate,
                "operator_norm_squared": operator_norm_squared,
            },
        )
    control = resolve_control(control, default_iterations=int(num_iterations), default_tolerance=None, callback=callback)
    limit = min(int(num_iterations), int(control.max_iterations or num_iterations))
    parameters = {
        "num_iterations": int(num_iterations),
        "tol": float(tol),
        "min_value": min_value,
        "max_value": max_value,
        "eps": float(eps),
    }
    if not _finite_tensor(measurement):
        return _numerical_result(
            "cgls", measurement, operator, x_init=x_init, max_iterations=limit,
            reason="non_finite_solver_state", parameters=parameters,
        )
    x = prepare_initial_image(measurement, operator, x_init=x_init, initial_value=0.0)
    if not _finite_tensor(x):
        return _numerical_result(
            "cgls", measurement, operator, x_init=x_init, max_iterations=limit,
            reason="non_finite_initial_reconstruction", parameters=parameters,
        )
    if _policy_active(control) or operator_norm_estimate is not None or operator_norm_squared is not None:
        try:
            operator_norm, norm_metadata = _resolve_operator_norm_estimate(
                operator,
                measurement,
                control,
                explicit=operator_norm_estimate,
                explicit_squared=operator_norm_squared,
            )
        except FloatingPointError:
            return _numerical_result(
                "cgls", measurement, operator, x_init=x_init, max_iterations=limit,
                reason="non_finite_operator_norm_estimate", parameters=parameters,
            )
        except ValueError as error:
            return _parameter_failure_result(
                "cgls", measurement, operator, x_init=x_init, errors=[str(error)],
                max_iterations=limit, parameters=parameters,
            )
    else:
        operator_norm = 1.0
        norm_metadata = {
            "operator_norm_estimate": None,
            "operator_norm_squared": None,
            "operator_norm_estimator": "not_required_for_legacy_stop",
        }
    recorder = IterationRecorder(control, measurement, operator, algorithm="cgls")
    monitor = ConsecutiveStoppingMonitor(control)
    recorder.set_initial(x)
    prediction = operator.forward(x)
    residual = prediction - measurement
    adjoint_residual = operator.adjoint(-residual)
    gamma = _sqnorm(adjoint_residual)
    initial_normal = max(_global_norm(adjoint_residual), float(eps))
    normal_tolerance = float(tol)
    if control.tolerance is not None:
        normal_tolerance = float(control.tolerance)
    if not _finite_tensor(prediction) or not _finite_tensor(residual) or not _finite_tensor(adjoint_residual):
        recorder.mark_numerical_error("non_finite_solver_state")
    actual = 0
    converged = False
    cancelled = False
    terminal_status = None
    terminal_reason = None
    if not recorder.numerical_failure and not _policy_active(control) and _global_norm(adjoint_residual) <= normal_tolerance:
        recorder.record(
            0,
            x,
            residual=residual,
            objective=_objective(residual),
            algorithm_residual=_global_norm(adjoint_residual) / max(
                operator_norm * _global_norm(residual), float(eps)
            ),
            stopping_candidate=True,
            metadata={**norm_metadata, "criterion": "absolute_normal_residual"},
        )
        converged = True
    direction = adjoint_residual.clone()
    for iteration in range(1, limit + 1):
        if recorder.numerical_failure:
            break
        if converged and control.stop_on_convergence:
            break
        q = operator.forward(direction)
        denominator = _sqnorm(q)
        if not _finite_tensor(q) or not bool(torch.isfinite(denominator).all()):
            recorder.mark_numerical_error("non_finite_krylov_direction")
            break
        if bool(torch.any(denominator <= float(eps))):
            # A zero Krylov direction at a stationary point is benign (for
            # example, a zero operator or an exact initial solution).  It is
            # not evidence of data consistency; leave policy runs as a
            # structured stall and reserve numerical_error for a material
            # normal-equation breakdown.
            normal_value = _global_norm(adjoint_residual)
            breakdown_scale = max(_global_norm(measurement), 1.0)
            if normal_value <= max(float(eps), 1e-7 * breakdown_scale):
                if not _policy_active(control) and normal_value <= normal_tolerance:
                    converged = True
                    break
                if _policy_active(control):
                    residual = prediction - measurement
                    residual_value = _normalized(_global_norm(residual), _global_norm(measurement))
                    normalized_normal = normal_value / max(
                        operator_norm * _global_norm(residual), float(eps)
                    )
                    normal_tolerance_policy = _control_threshold(
                        control, "normalized_normal_residual_tolerance"
                    )
                    iterate_tolerance_policy = _control_threshold(
                        control, "relative_iterate_tolerance"
                    )
                    criteria = {
                        "krylov_native": (
                            normalized_normal <= normal_tolerance_policy
                            or 0.0 <= iterate_tolerance_policy
                        ),
                    }
                    if control.discrepancy_target is not None:
                        criteria = {
                            "discrepancy": residual_value <= float(control.discrepancy_target),
                            **criteria,
                        }
                    decision = monitor.observe(
                        iteration,
                        criteria=criteria,
                        relative_change=0.0,
                        monitor_value=normalized_normal,
                    )
                    converged = decision.converged
                    terminal_status, terminal_reason = _policy_status(decision)
                    actual = iteration
                    if not _safe_record(
                        recorder,
                        iteration,
                        x,
                        residual=residual,
                        objective=_objective(residual),
                        algorithm_residual=normalized_normal,
                        stopping_candidate=converged,
                        consecutive_criteria_count=decision.consecutive,
                        criteria=criteria,
                        native_criterion_name="normalized_normal_residual",
                        native_criterion_value=normalized_normal,
                        native_criterion_threshold=normal_tolerance_policy,
                        metadata={
                            **norm_metadata,
                            "criterion": "scale_aware_normal_residual",
                            "normal_residual_absolute": normal_value,
                            "normal_residual_denominator": operator_norm * _global_norm(residual),
                            "breakdown": "stationary_krylov_direction",
                        },
                    ):
                        cancelled = True
                        break
                    if terminal_status or (converged and control.stop_on_convergence):
                        break
                    if control.discrepancy_target is not None and not criteria["discrepancy"]:
                        terminal_status = "stalled"
                        terminal_reason = "stalled_before_discrepancy"
                        break
                    continue
                terminal_status = "max_iterations"
                terminal_reason = "maximum_iterations_reached"
                break
            recorder.mark_numerical_error("normal_equation_breakdown")
            break
        alpha = gamma / denominator.clamp_min(float(eps))
        if not bool(torch.isfinite(alpha).all()):
            recorder.mark_numerical_error("non_finite_krylov_step")
            break
        previous = x
        x = x + alpha.reshape((alpha.shape[0],) + (1,) * (x.ndim - 1)) * direction
        x = apply_box_constraints(x, min_value=min_value, max_value=max_value)
        if not _finite_tensor(x):
            recorder.mark_numerical_error("non_finite_reconstruction")
            break
        prediction = operator.forward(x)
        residual = prediction - measurement
        next_adjoint = operator.adjoint(-residual)
        next_gamma = _sqnorm(next_adjoint)
        if not _finite_tensor(prediction) or not _finite_tensor(residual) or not _finite_tensor(next_adjoint):
            recorder.mark_numerical_error("non_finite_solver_state")
            break
        change = _relative_change(previous, x)
        normal_value = _global_norm(next_adjoint)
        residual_value = _normalized(_global_norm(residual), _global_norm(measurement))
        normalized_normal = normal_value / max(operator_norm * _global_norm(residual), float(eps))
        normal_tolerance_policy = _control_threshold(
            control, "normalized_normal_residual_tolerance"
        )
        iterate_tolerance_policy = _control_threshold(
            control, "relative_iterate_tolerance"
        )
        criteria = {
            "krylov_native": (
                normalized_normal <= normal_tolerance_policy
                or change <= iterate_tolerance_policy
            ),
        }
        if control.discrepancy_target is not None:
            criteria = {
                "discrepancy": residual_value <= float(control.discrepancy_target),
                **criteria,
            }
        decision = monitor.observe(iteration, criteria=criteria, relative_change=change, monitor_value=normalized_normal)
        converged = decision.converged if _policy_active(control) else normal_value <= normal_tolerance
        terminal_status, terminal_reason = _policy_status(decision) if _policy_active(control) else (None, None)
        actual = iteration
        if not _safe_record(
            recorder,
            actual,
            x,
            residual=residual,
            objective=_objective(residual),
            algorithm_residual=normalized_normal if _policy_active(control) else _normalized(normal_value, initial_normal),
            stopping_candidate=converged,
            consecutive_criteria_count=decision.consecutive,
            criteria=criteria,
            native_criterion_name="normalized_normal_residual",
            native_criterion_value=normalized_normal,
            native_criterion_threshold=(
                normal_tolerance_policy
                if _policy_active(control)
                else control.normalized_normal_residual_tolerance
            ),
            metadata={
                **norm_metadata,
                "criterion": "scale_aware_normal_residual",
                "normal_residual_absolute": normal_value,
                "normal_residual_denominator": operator_norm * _global_norm(residual),
            },
        ):
            cancelled = True
            break
        if terminal_status or (converged and control.stop_on_convergence):
            break
        beta = next_gamma / gamma.clamp_min(float(eps))
        direction = next_adjoint + beta.reshape((beta.shape[0],) + (1,) * (direction.ndim - 1)) * direction
        adjoint_residual, gamma = next_adjoint, next_gamma
    # Keep endpoint evaluation explicit and counted.  This makes the final
    # prediction phase distinguishable from the last Krylov update and keeps
    # resource accounting consistent with the benchmark protocol.
    prediction = _finish_prediction(operator, x)
    final_residual_tensor = prediction - measurement
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
            terminal_reason if terminal_reason else
            "discrepancy_and_krylov_native_patience" if status == "converged" and _policy_active(control) else
            "normal_residual_tolerance" if status == "converged" else
            "callback_cancelled" if cancelled else
            "maximum_iterations_reached"
        ),
        final_residual=_normalized(_global_norm(final_residual_tensor), _global_norm(measurement)),
        final_objective=_objective(final_residual_tensor),
        predicted_measurement=prediction,
        metadata={
            **norm_metadata,
            "criterion": "scale_aware_normal_residual",
            "tolerance": normal_tolerance,
            "normal_residual_formula": "||A^T(Ax-b)||_2 / max(||A||_estimate * ||Ax-b||_2, eps)",
        },
    )


def solve_lsqr_detailed(
    operator: LinearOperator,
    measurement: torch.Tensor,
    *,
    num_iterations: int = 10,
    damping: float = 0.0,
    atol: float = 1e-6,
    btol: float = 1e-6,
    x_init: torch.Tensor | None = None,
    min_value: float | None = None,
    max_value: float | None = None,
    eps: float = 1e-12,
    control: SolveControl | None = None,
    callback: Callback | None = None,
    operator_norm_estimate: float | None = None,
    operator_norm_squared: float | None = None,
) -> SolveResult:
    from inv_framework.solvers._iteration import _common_parameter_errors, _control_threshold, _finish_status, _finite_nonnegative, _finite_positive, _finite_tensor, _global_norm, _krylov_normal_metrics, _norm, _normalized, _numerical_result, _objective, _parameter_failure_result, _policy_active, _policy_status, _relative_change, _resolve_operator_norm_estimate, _safe_record
    require_linear_operator(operator, "lsqr")
    validate_measurement_shape(measurement, operator, "lsqr")
    parameter_errors = _common_parameter_errors(num_iterations, min_value, max_value)
    for name, value in (("damping", damping), ("atol", atol), ("btol", btol)):
        if not _finite_nonnegative(value):
            parameter_errors.append(f"{name} must be a finite nonnegative number")
    if not _finite_positive(eps):
        parameter_errors.append("eps must be a finite positive number")
    if operator_norm_estimate is not None and not _finite_nonnegative(operator_norm_estimate):
        parameter_errors.append("operator_norm_estimate must be finite and nonnegative")
    if operator_norm_squared is not None and not _finite_nonnegative(operator_norm_squared):
        parameter_errors.append("operator_norm_squared must be finite and nonnegative")
    if parameter_errors:
        return _parameter_failure_result(
            "lsqr", measurement, operator, x_init=x_init,
            errors=parameter_errors,
            max_iterations=num_iterations if isinstance(num_iterations, int) and not isinstance(num_iterations, bool) else None,
            parameters={
                "num_iterations": num_iterations,
                "damping": damping,
                "atol": atol,
                "btol": btol,
                "min_value": min_value,
                "max_value": max_value,
                "eps": eps,
                "operator_norm_estimate": operator_norm_estimate,
                "operator_norm_squared": operator_norm_squared,
            },
        )
    control = resolve_control(control, default_iterations=int(num_iterations), default_tolerance=None, callback=callback)
    limit = min(int(num_iterations), int(control.max_iterations or num_iterations))
    epsilon = float(eps)
    x = prepare_initial_image(measurement, operator, x_init=x_init, initial_value=0.0)
    parameters = {
        "num_iterations": int(num_iterations),
        "damping": float(damping),
        "atol": float(atol),
        "btol": float(btol),
        "min_value": min_value,
        "max_value": max_value,
        "eps": float(eps),
    }
    if not _finite_tensor(measurement):
        return _numerical_result(
            "lsqr", measurement, operator, x_init=x_init, max_iterations=limit,
            reason="non_finite_solver_state", parameters=parameters,
        )
    if not _finite_tensor(x):
        return _numerical_result(
            "lsqr", measurement, operator, x_init=x_init, max_iterations=limit,
            reason="non_finite_initial_reconstruction", parameters=parameters,
        )
    if _policy_active(control) or operator_norm_estimate is not None or operator_norm_squared is not None:
        try:
            operator_norm, norm_metadata = _resolve_operator_norm_estimate(
                operator,
                measurement,
                control,
                explicit=operator_norm_estimate,
                explicit_squared=operator_norm_squared,
            )
        except FloatingPointError:
            return _numerical_result(
                "lsqr", measurement, operator, x_init=x_init, max_iterations=limit,
                reason="non_finite_operator_norm_estimate", parameters=parameters,
            )
        except ValueError as error:
            return _parameter_failure_result(
                "lsqr", measurement, operator, x_init=x_init, errors=[str(error)],
                max_iterations=limit, parameters=parameters,
            )
    else:
        operator_norm = 1.0
        norm_metadata = {
            "operator_norm_estimate": None,
            "operator_norm_squared": None,
            "operator_norm_estimator": "not_required_for_legacy_stop",
        }
    recorder = IterationRecorder(control, measurement, operator, algorithm="lsqr")
    monitor = ConsecutiveStoppingMonitor(control)
    recorder.set_initial(x)
    rhs_norm = _global_norm(measurement)
    initial_prediction = operator.forward(x)
    u = measurement - initial_prediction
    beta = _norm(u)
    if not _finite_tensor(initial_prediction) or not _finite_tensor(u) or not bool(torch.isfinite(beta).all()):
        recorder.mark_numerical_error("non_finite_solver_state")
    u = u / beta.clamp_min(epsilon).reshape((beta.shape[0],) + (1,) * (u.ndim - 1))
    v = operator.adjoint(u)
    alpha = _norm(v)
    if not _finite_tensor(v) or not bool(torch.isfinite(alpha).all()):
        recorder.mark_numerical_error("non_finite_solver_state")
    v = v / alpha.clamp_min(epsilon).reshape((alpha.shape[0],) + (1,) * (v.ndim - 1))
    direction = v.clone()
    phi_bar = beta.clone()
    rho_bar = alpha.clone()
    actual = 0
    converged = False
    cancelled = False
    terminal_status = None
    terminal_reason = None
    stop_tol = float(atol) * rhs_norm + float(btol)
    residual_tensor = measurement - operator.forward(x)
    if not _finite_tensor(residual_tensor):
        recorder.mark_numerical_error("non_finite_solver_state")
    if (
        not recorder.numerical_failure
        and not _policy_active(control)
        and _global_norm(residual_tensor) <= stop_tol
    ):
        converged = True
        recorder.record(
            0,
            x,
            residual=-residual_tensor,
            objective=_objective(residual_tensor),
            algorithm_residual=_normalized(_global_norm(residual_tensor), rhs_norm),
            stopping_candidate=True,
            metadata={"criterion": "atol_btol_tolerance", **norm_metadata},
        )
    # A zero initial bidiagonal coefficient means that the residual has no
    # usable Krylov direction.  It is a legitimate exact fit when the native
    # tolerance already accepts the residual; otherwise it is a stationary
    # breakdown and must not be advertised as convergence.
    if (
        not recorder.numerical_failure
        and not converged
        and bool(torch.any(alpha <= epsilon))
    ):
        discrepancy_ok = (
            control.discrepancy_target is None
            or _normalized(_global_norm(residual_tensor), rhs_norm)
            <= float(control.discrepancy_target)
        )
        if _policy_active(control) and discrepancy_ok:
            # The zero initial bidiagonal coefficient can also represent an
            # exact initial fit.  Let the normal monitor collect its patience
            # window instead of converting that valid stationary state into a
            # premature stall.
            pass
        else:
            terminal_status = "stalled" if _policy_active(control) else "max_iterations"
            terminal_reason = (
                "stalled_before_discrepancy"
                if _policy_active(control) else "krylov_breakdown"
            )
    for iteration in range(1, limit + 1):
        if recorder.numerical_failure:
            break
        if converged and control.stop_on_convergence:
            break
        if terminal_status:
            break
        residual_norm = _global_norm(residual_tensor)
        u = operator.forward(v) - alpha.reshape((alpha.shape[0],) + (1,) * (u.ndim - 1)) * u
        beta = _norm(u)
        if not _finite_tensor(u) or not bool(torch.isfinite(beta).all()):
            recorder.mark_numerical_error("non_finite_bidiagonal_state")
            break
        u = u / beta.clamp_min(epsilon).reshape((beta.shape[0],) + (1,) * (u.ndim - 1))
        v = operator.adjoint(u) - beta.reshape((beta.shape[0],) + (1,) * (v.ndim - 1)) * v
        alpha = _norm(v)
        if not _finite_tensor(v) or not bool(torch.isfinite(alpha).all()):
            recorder.mark_numerical_error("non_finite_bidiagonal_state")
            break
        v = v / alpha.clamp_min(epsilon).reshape((alpha.shape[0],) + (1,) * (v.ndim - 1))
        if float(damping) > 0.0:
            rho_damped = torch.sqrt(rho_bar.square() + float(damping) ** 2).clamp_min(epsilon)
            phi_bar = (rho_bar / rho_damped) * phi_bar
            rho_bar = rho_damped
        rho = torch.sqrt(rho_bar.square() + beta.square()).clamp_min(epsilon)
        cosine = rho_bar / rho
        sine = beta / rho
        theta = sine * alpha
        rho_bar = -cosine * alpha
        phi = cosine * phi_bar
        phi_bar = sine * phi_bar
        previous = x
        x = x + (phi / rho).reshape((phi.shape[0],) + (1,) * (x.ndim - 1)) * direction
        x = apply_box_constraints(x, min_value=min_value, max_value=max_value)
        direction = v - (theta / rho).reshape((theta.shape[0],) + (1,) * (direction.ndim - 1)) * direction
        if not _finite_tensor(x) or not _finite_tensor(direction):
            recorder.mark_numerical_error("non_finite_reconstruction")
            break
        actual = iteration
        residual_tensor = measurement - operator.forward(x)
        if not _finite_tensor(residual_tensor):
            recorder.mark_numerical_error("non_finite_residual")
            break
        residual_value = _normalized(_global_norm(residual_tensor), rhs_norm)
        change = _relative_change(previous, x)
        if _policy_active(control):
            try:
                _gradient, normal_value, normalized_normal, normal_denominator = _krylov_normal_metrics(
                    operator,
                    -residual_tensor,
                    operator_norm,
                    eps=epsilon,
                    x=x,
                    damping=float(damping),
                )
            except (FloatingPointError, ValueError):
                recorder.mark_numerical_error("non_finite_normal_residual")
                break
        else:
            normal_value = residual_value
            normalized_normal = residual_value
            normal_denominator = _global_norm(residual_tensor)
        normal_tolerance_policy = _control_threshold(
            control, "normalized_normal_residual_tolerance"
        )
        iterate_tolerance_policy = _control_threshold(
            control, "relative_iterate_tolerance"
        )
        criteria = {
            "krylov_native": (
                normalized_normal <= normal_tolerance_policy
                or change <= iterate_tolerance_policy
            ),
        }
        if control.discrepancy_target is not None:
            criteria = {
                "discrepancy": residual_value <= float(control.discrepancy_target),
                **criteria,
            }
        decision = monitor.observe(actual, criteria=criteria, relative_change=change, monitor_value=normalized_normal)
        converged = decision.converged if _policy_active(control) else residual_norm <= stop_tol or (stop_tol > 0.0 and residual_value <= _normalized(stop_tol, rhs_norm))
        terminal_status, terminal_reason = _policy_status(decision) if _policy_active(control) else (None, None)
        if not _safe_record(
            recorder,
            actual,
            x,
            residual=-residual_tensor,
            objective=_objective(residual_tensor),
            algorithm_residual=normalized_normal if _policy_active(control) else residual_value,
            stopping_candidate=converged,
            consecutive_criteria_count=decision.consecutive,
            criteria=criteria,
            native_criterion_name="normalized_normal_residual",
            native_criterion_value=normalized_normal,
            native_criterion_threshold=(
                normal_tolerance_policy
                if _policy_active(control)
                else control.normalized_normal_residual_tolerance
            ),
            metadata={
                **norm_metadata,
                "criterion": "scale_aware_normal_residual" if _policy_active(control) else "atol_btol_tolerance",
                "normal_residual_denominator": normal_denominator,
                "damping": float(damping),
                "normal_residual_formula": (
                    "||A^T(Ax-b) + damping^2*x||_2 / "
                    "max(||A||_estimate*||Ax-b||_2 + damping^2*||x||_2, eps)"
                    if float(damping) > 0.0 else
                    "||A^T(Ax-b)||_2 / max(||A||_estimate * ||Ax-b||_2, eps)"
                ),
            },
        ):
            cancelled = True
            break
        if bool(torch.any(alpha <= epsilon)) and not converged:
            if not _policy_active(control) and _global_norm(residual_tensor) <= stop_tol:
                converged = True
            elif _policy_active(control) and (
                control.discrepancy_target is None
                or bool(criteria.get("discrepancy", False))
            ):
                # A zero Lanczos coefficient after a successful update is an
                # exact Krylov plateau, not a failure.  Keep recording the
                # stationary evidence so the configured patience window can
                # still be honored.
                pass
            else:
                terminal_status = "stalled" if _policy_active(control) else "max_iterations"
                terminal_reason = (
                    "stalled_before_discrepancy"
                    if _policy_active(control) else "krylov_breakdown"
                )
        if terminal_status or (converged and control.stop_on_convergence):
            break
    prediction = (measurement - residual_tensor).detach()
    final_residual_tensor = prediction - measurement
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
            terminal_reason if terminal_reason else
            "discrepancy_and_krylov_native_patience" if status == "converged" and _policy_active(control) else
            "atol_btol_tolerance" if status == "converged" else
            "callback_cancelled" if cancelled else
            "maximum_iterations_reached"
        ),
        final_residual=_normalized(_global_norm(final_residual_tensor), rhs_norm),
        final_objective=_objective(final_residual_tensor),
        predicted_measurement=prediction,
        metadata={
            **norm_metadata,
            "criterion": "scale_aware_normal_residual" if _policy_active(control) else "simplified_atol_times_measurement_norm_plus_btol",
            "atol": float(atol),
            "btol": float(btol),
            "damping": float(damping),
            "normal_residual_formula": (
                "||A^T(Ax-b) + damping^2*x||_2 / "
                "max(||A||_estimate*||Ax-b||_2 + damping^2*||x||_2, eps)"
                if float(damping) > 0.0 else
                "||A^T(Ax-b)||_2 / max(||A||_estimate * ||Ax-b||_2, eps)"
            ),
        },
    )


def solve_sart_detailed(
    operator: LinearOperator,
    measurement: torch.Tensor,
    *,
    num_iterations: int = 100,
    block_size: int = 1,
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
    from inv_framework.solvers.subset import _subset_solver_detailed
    return _subset_solver_detailed(
        "sart", operator, measurement, num_iterations=num_iterations, block_size=block_size,
        subset_indices=subset_indices, order_strategy=order_strategy, seed=seed,
        relaxation=relaxation, min_value=min_value, max_value=max_value, x_init=x_init,
        eps=eps, control=control, callback=callback,
    )

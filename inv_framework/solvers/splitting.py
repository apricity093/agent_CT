"""Composite, constrained and proximal reconstruction methods."""
from __future__ import annotations
from inv_framework.core.operator import GradientOperator, ForwardOperator, LinearOperator
from inv_framework.core.regularizer import TikhonovRegularizer, TVRegularizer
from inv_framework.solvers.base import InverseProblemSolver, IterationRecorder, ConsecutiveStoppingMonitor, resolve_control, SolveControl, SolveResult
from inv_framework.solvers._utils import prepare_initial_image, require_linear_operator, validate_measurement_shape, apply_box_constraints
from inv_framework.solvers.subset import make_angle_subsets, make_subset_operator, select_measurement_subset
from typing import Optional
from dataclasses import replace
from inv_framework.core.fidelity import ResidualBall
from inv_framework.solvers.subset import _safe_reciprocal
import math
import time
import torch


class ADMMSolver(InverseProblemSolver):
    algorithm = "admm"

    def __init__(self, regularizer=None, **parameters):
        self.parameters = dict(regularizer=regularizer, **parameters)

    def solve_detailed(
        self, measurement, operator, x_init=None, control=None, callback=None, **kwargs
    ):
        self.endpoint_state = {}
        return solve_split_detailed(
            operator,
            measurement,
            algorithm=self.algorithm,
            endpoint_state=self.endpoint_state,
            x_init=x_init,
            control=control,
            callback=callback,
            **{**self.parameters, **kwargs}
        )

    def solve(self, measurement, operator, x_init=None, **kwargs):
        result = self.solve_detailed(measurement, operator, x_init=x_init, **kwargs)
        if result.status == "invalid_parameters":
            raise ValueError(result.stopping_reason)
        if result.status == "numerical_error":
            raise FloatingPointError(result.stopping_reason)
        return result.reconstruction


class Penalty:
    def __init__(self, reg, shape):
        self.reg = reg
        self.calls = dict(
            transform_forward_calls=0,
            transform_adjoint_calls=0,
            coefficient_prox_calls=0,
            transform_seconds=0.0,
            coefficient_prox_seconds=0.0,
        )
        if not all(
            callable(getattr(reg, name, None))
            for name in ("split_transform", "coefficient_value", "coefficient_proximal")
        ):
            raise ValueError(
                "unsupported split regularizer: missing mathematical capabilities"
            )
        self.transform, self.bound = reg.split_transform(shape)

    def apply(self, x, adjoint=False):
        if self.bound == 0:
            return (
                torch.zeros((x.shape[0], *self.shape), dtype=x.dtype, device=x.device)
                if adjoint
                else x.new_zeros((x.shape[0], 0))
            )
        if x.is_cuda:
            torch.cuda.synchronize(x.device)
        start = time.perf_counter()
        self.calls[
            "transform_adjoint_calls" if adjoint else "transform_forward_calls"
        ] += 1
        result = (
            x
            if self.transform is None
            else (self.transform.adjoint(x) if adjoint else self.transform.forward(x))
        )
        if x.is_cuda:
            torch.cuda.synchronize(x.device)
        self.calls["transform_seconds"] += time.perf_counter() - start
        return result

    def mask(self, z):
        mask = getattr(self.reg, "coefficient_mask", None)
        return mask(z) if callable(mask) else torch.ones_like(z)

    def value(self, z):
        if self.bound == 0:
            return z.new_tensor(0.0)
        return self.reg.coefficient_value(z).sum()

    def prox(self, z, t):
        if self.bound == 0:
            return z
        if z.is_cuda:
            torch.cuda.synchronize(z.device)
        start = time.perf_counter()
        self.calls["coefficient_prox_calls"] += 1
        failed = True
        try:
            out = self.reg.coefficient_proximal(z, t)
            failed = False
            return out
        finally:
            if z.is_cuda:
                torch.cuda.synchronize(z.device)
            elapsed = time.perf_counter() - start
            self.calls["coefficient_prox_seconds"] += elapsed

    def dual_prox(self, z, sigma, weight):
        if self.bound == 0:
            return z
        # Moreau also handles zero-weight LL: its conjugate is the indicator {0}.
        result = z - sigma * self.prox(z / sigma, weight / sigma)
        result = result * self.mask(result)
        if weight == 0:
            result = torch.zeros_like(result)
        return result


def norm_bound(operator, x):
    """Schur bound using absolute row/column sums; signed generic matrices use columns."""
    return operator.squared_norm_upper_bound(x)


def solve_split_detailed(
    operator,
    measurement,
    *,
    algorithm,
    regularizer=None,
    strength=None,
    reg_strength=None,
    num_iterations=200,
    tolerance=1e-05,
    rho=1.0,
    cg_max_iterations=100,
    cg_tolerance=1e-06,
    tau=None,
    sigma=None,
    min_value=None,
    max_value=None,
    x_init=None,
    control=None,
    callback=None,
    endpoint_state=None,
    fidelity=None
):
    from inv_framework.solvers._iteration import _parameter_failure_result, _common_parameter_errors, _policy_status

    require_linear_operator(operator, algorithm)
    validate_measurement_shape(measurement, operator, algorithm)
    from inv_framework.core.fidelity import QuadraticFidelity

    weight = (
        strength
        if strength is not None
        else reg_strength if reg_strength is not None else 0.01
    )
    reg = regularizer if regularizer is not None else TikhonovRegularizer()
    from inv_framework.core.regularizer import ZeroRegularizer

    if weight == 0:
        reg = ZeroRegularizer()
    if reg == "h1":
        reg = TikhonovRegularizer(GradientOperator(operator.domain_shape))
    errors = _common_parameter_errors(num_iterations, min_value, max_value)
    if strength is not None and reg_strength is not None and (strength != reg_strength):
        errors.append("strength and reg_strength conflict")
    for name, v, positive in [
        ("strength", weight, False),
        ("tolerance", tolerance, False),
        ("rho", rho, True),
        ("cg_tolerance", cg_tolerance, True),
    ]:
        if not math.isfinite(float(v)) or (v <= 0 if positive else v < 0):
            errors.append(name + " invalid")
    for name, v in [("tau", tau), ("sigma", sigma)]:
        if v is not None and (not math.isfinite(float(v)) or v <= 0):
            errors.append(name + " must be positive finite")
    if (
        isinstance(cg_max_iterations, bool)
        or not isinstance(cg_max_iterations, int)
        or cg_max_iterations <= 0
    ):
        errors.append("cg_max_iterations must be positive integer")
    try:
        penalty = Penalty(reg, operator.domain_shape)
    except ValueError as e:
        errors.append(str(e))
    if errors:
        return _parameter_failure_result(
            algorithm,
            measurement,
            operator,
            x_init=x_init,
            errors=errors,
            max_iterations=num_iterations,
            parameters={},
        )
    penalty.shape = operator.domain_shape
    control = resolve_control(
        control,
        default_iterations=num_iterations,
        default_tolerance=tolerance,
        callback=callback,
    )
    x = prepare_initial_image(measurement, operator, x_init=x_init, initial_value=0.0)
    nonnegative = min_value == 0
    constrained = min_value is not None or max_value is not None
    project = lambda v: apply_box_constraints(
        v, min_value=min_value, max_value=max_value
    )
    x = project(x)
    recorder = IterationRecorder(control, measurement, operator, algorithm=algorithm)
    recorder.set_initial(x)
    monitor = ConsecutiveStoppingMonitor(control)
    norm = lambda v: float(v.norm())
    fidelity_before = {
        k: getattr(fidelity, k, 0)
        for k in (
            "prox_calls",
            "inner_iterations",
            "inner_failures",
            "prox_seconds",
            "weight_multiplications",
            "mask_calls",
        )
    }
    cg_started = None
    actual = 0
    cg_total = 0
    cg_seconds = 0.0
    cg_failures = 0
    status, reason = ("max_iterations", "maximum_iterations_reached")
    prediction = residual = objective = metric = None
    endpoint = False
    interruption = None
    bound = None
    primal = dual = None
    inner_relative = None
    tol = float(control.tolerance or 0)
    try:
        if not bool(torch.isfinite(x).all() and torch.isfinite(measurement).all()):
            raise FloatingPointError("non_finite_input")
        fidelity = fidelity if fidelity is not None else QuadraticFidelity(measurement)
        observation_split = algorithm == "admm"
        z = penalty.apply(x)
        u = torch.zeros_like(z)
        q = x.clone()
        v = torch.zeros_like(x)
        if algorithm == "pdhg":
            try:
                bound = norm_bound(operator, x) + penalty.bound
            except ValueError as error:
                return _parameter_failure_result(
                    algorithm,
                    measurement,
                    operator,
                    x_init=x_init,
                    errors=[str(error)],
                    max_iterations=num_iterations,
                    parameters={},
                )
            tau = float(tau) if tau is not None else 0.99 / math.sqrt(bound)
            sigma = float(sigma) if sigma is not None else 0.99 / math.sqrt(bound)
            if tau * sigma * bound >= 1:
                return _parameter_failure_result(
                    algorithm,
                    measurement,
                    operator,
                    x_init=x_init,
                    errors=["tau*sigma*norm_bound_squared must be < 1"],
                    max_iterations=num_iterations,
                    parameters={},
                )
            p = torch.zeros_like(measurement)
            d = torch.zeros_like(z)
            bar = x.clone()
        else:
            if not nonnegative and callable(getattr(fidelity, "validate_normal", None)):
                try:
                    fidelity.validate_normal(operator, penalty.transform, x)
                except ValueError as error:
                    return _parameter_failure_result(
                        algorithm,
                        measurement,
                        operator,
                        x_init=x_init,
                        errors=[str(error)],
                        max_iterations=num_iterations,
                        parameters={},
                    )
            if observation_split:
                observation = operator.forward(x)
                observation_dual = torch.zeros_like(observation)
            else:
                aty = operator.adjoint(fidelity.linear_rhs())
        for iteration in range(1, min(num_iterations, control.max_iterations) + 1):
            old = x
            if algorithm == "admm":
                rhs = (
                    (
                        rho * operator.adjoint(observation - observation_dual)
                        if observation_split
                        else aty
                    )
                    + rho * penalty.apply(z - u, True)
                    + rho * (q - v)
                )

                def normal(a):
                    prediction_a = operator.forward(a)
                    return (
                        operator.adjoint(
                            rho * prediction_a
                            if observation_split
                            else fidelity.normal_gradient(prediction_a)
                        )
                        + rho * penalty.apply(penalty.apply(a), True)
                        + rho * a
                    )

                cg_started = time.perf_counter()
                candidate = x.clone()
                r = rhs - normal(candidate)
                direction = r.clone()
                rr = (r * r).sum()
                target = cg_tolerance / iteration**2 * max(norm(rhs), 1.0)
                inner = 0
                while norm(r) > target and inner < cg_max_iterations:
                    nd = normal(direction)
                    curvature = (direction * nd).sum()
                    if not bool(torch.isfinite(curvature)) or float(curvature) <= 0:
                        raise FloatingPointError(
                            "cg_nonpositive_curvature_or_breakdown"
                        )
                    alpha = rr / curvature
                    candidate = candidate + alpha * direction
                    r = r - alpha * nd
                    newrr = (r * r).sum()
                    direction = r + newrr / rr * direction
                    rr = newrr
                    inner += 1
                    cg_total += 1
                cg_seconds += time.perf_counter() - cg_started
                cg_started = None
                inner_relative = norm(r) / max(norm(rhs), 1.0)
                if norm(r) > target:
                    cg_failures += 1
                    status, reason = ("max_iterations", "cg_inner_max_iterations")
                    break
                tx = penalty.apply(candidate)
                nz = penalty.prox(tx + u, weight / rho)
                nq = project(candidate + v)
                nu = u + tx - nz
                nv = v + candidate - nq
                obs_primal = 0.0
                ds = rho * (penalty.apply(nz - z, True) + (nq - q))
                if observation_split:
                    ax = operator.forward(candidate)
                    next_observation = fidelity.proximal(ax + observation_dual, 1 / rho)
                    next_observation_dual = observation_dual + ax - next_observation
                    obs_primal = norm(ax - next_observation) ** 2
                    ds = ds + rho * operator.adjoint(next_observation - observation)
                primal = math.sqrt(
                    norm(tx - nz) ** 2 + norm(candidate - nq) ** 2 + obs_primal
                )
                dual = norm(ds)
                eps_p = tol * (
                    math.sqrt(
                        tx.numel()
                        + candidate.numel()
                        + (ax.numel() if observation_split else 0)
                    )
                    + max(
                        math.sqrt(
                            norm(tx) ** 2
                            + norm(candidate) ** 2
                            + (norm(ax) ** 2 if observation_split else 0)
                        ),
                        math.sqrt(
                            norm(nz) ** 2
                            + norm(nq) ** 2
                            + (norm(next_observation) ** 2 if observation_split else 0)
                        ),
                    )
                )
                multiplier = rho * (penalty.apply(nu, True) + nv)
                if observation_split:
                    multiplier = multiplier + rho * operator.adjoint(
                        next_observation_dual
                    )
                eps_d = tol * (math.sqrt(x.numel()) + norm(multiplier))
                x, z, u, q, v = (candidate, nz, nu, nq, nv)
                if observation_split:
                    observation, observation_dual = (
                        next_observation,
                        next_observation_dual,
                    )
                metric = max(primal / max(eps_p, 1e-30), dual / max(eps_d, 1e-30))
                criteria = {"primal": primal <= eps_p, "dual": dual <= eps_d}
            else:
                np = fidelity.dual_prox(p + sigma * operator.forward(bar), sigma)
                nd = penalty.dual_prox(d + sigma * penalty.apply(bar), sigma, weight)
                nx = project(x - tau * (operator.adjoint(np) + penalty.apply(nd, True)))
                bar = 2 * nx - x
                x, p, d = (nx, np, nd)
                criteria = {}
            if endpoint_state is not None:
                endpoint_state.clear()
            endpoint = False
            actual = iteration
            returned = q if algorithm == "admm" and constrained else x
            prediction = operator.forward(returned)
            residual = prediction - measurement
            txr = penalty.apply(returned)
            objective = float(fidelity.value(prediction)) + weight * float(
                penalty.value(txr)
            )
            if algorithm == "pdhg":
                pg = operator.adjoint(p) + penalty.apply(d, True)
                rx = (returned - project(returned - tau * pg)) / tau
                rp = (p - fidelity.dual_prox(p + sigma * prediction, sigma)) / sigma
                rd = (d - penalty.dual_prox(d + sigma * txr, sigma, weight)) / sigma
                metric = math.sqrt(norm(rx) ** 2 + norm(rp) ** 2 + norm(rd) ** 2) / max(
                    1.0, norm(returned), norm(p), norm(d)
                )
                criteria = {"primal_dual_fixed_point": metric <= tol}
            if not bool(torch.isfinite(returned).all()) or not math.isfinite(
                objective + metric
            ):
                raise FloatingPointError("non_finite_split_state")
            if endpoint_state is not None:
                endpoint_state.clear()
                endpoint_state.update(
                    p=(
                        p.detach().clone()
                        if algorithm == "pdhg"
                        else (
                            rho * observation_dual
                            if observation_split
                            else fidelity.gradient(prediction)
                        )
                        .detach()
                        .clone()
                    ),
                    d=(
                        d.detach().clone()
                        if algorithm == "pdhg"
                        else (rho * u).detach().clone()
                    ),
                    penalty=penalty,
                    fidelity=fidelity,
                    tau=tau if algorithm == "pdhg" else 1.0,
                    sigma=sigma if algorithm == "pdhg" else 1.0,
                    weight=weight,
                    nonnegative=nonnegative,
                    min_value=min_value,
                    max_value=max_value,
                )
            endpoint = True
            if control.discrepancy_target is not None:
                criteria["discrepancy"] = (
                    norm(residual) / max(norm(measurement), 1e-12)
                    <= control.discrepancy_target
                )
            decision = monitor.observe(
                iteration,
                criteria=criteria,
                relative_change=norm(x - old) / max(norm(old), 1.0),
                monitor_value=objective,
            )
            if not recorder.record(
                iteration,
                returned,
                residual=residual,
                objective=objective,
                algorithm_residual=metric,
                criteria=criteria,
                stopping_candidate=decision.converged,
                consecutive_criteria_count=decision.consecutive,
                native_criterion_name="split_residual",
                native_criterion_value=metric,
                metadata={
                    "primal_residual": primal,
                    "dual_residual": dual,
                    "complete_objective": True,
                },
            ):
                status, reason = ("cancelled", "callback_cancelled")
                break
            terminal, terminal_reason = _policy_status(decision)
            if terminal:
                status, reason = (terminal, terminal_reason)
                break
            if decision.converged and control.stop_on_convergence:
                status, reason = ("converged", "split_residual_patience")
                break
    except RuntimeError as e:
        if str(e) not in {
            "poisson_prox_inner_max_iterations",
            "poisson_prox_returned_precision",
        }:
            raise
        status, reason = ("max_iterations", "poisson_prox_inner_max_iterations")
        endpoint = False
    except FloatingPointError as e:
        status, reason = ("numerical_error", str(e))
        recorder.mark_numerical_error(reason)
    if cg_started is not None:
        cg_seconds += time.perf_counter() - cg_started
    returned = q if algorithm == "admm" and constrained and ("q" in locals()) else x
    return recorder.finish(
        returned,
        actual_iterations=actual,
        status=status,
        stopping_reason=reason,
        final_objective=objective if endpoint else None,
        final_residual=(
            norm(residual) / max(norm(measurement), 1e-12) if endpoint else None
        ),
        predicted_measurement=prediction if endpoint else None,
        resources={
            **{
                "fidelity_" + k: getattr(fidelity, k, 0) - v
                for k, v in fidelity_before.items()
            },
            **penalty.calls,
            "transform_kind": (
                "I"
                if penalty.transform is None
                else getattr(penalty.transform, "transform_kind", "linear")
            ),
            "cg_iterations": cg_total,
            "cg_seconds": cg_seconds,
            "cg_unconverged_subproblems": cg_failures,
        },
        metadata={
            "native_termination_managed": True,
            "endpoint_diagnostics_complete": endpoint,
            "budget_interruption": interruption,
            "strength": weight,
            "rho": rho,
            "tau": tau,
            "sigma": sigma,
            "norm_bound_squared": bound,
            "norm_bound_is_upper_bound": True,
            "primal_residual": primal,
            "dual_residual": dual,
            "split_residual": metric if endpoint else None,
            "cg_relative_residual": inner_relative,
            "fidelity_precision": (
                fidelity.precision_report()
                if callable(getattr(fidelity, "precision_report", None))
                else None
            ),
            "regularizer": type(reg).__name__,
            "delta": getattr(reg, "delta", None),
            "theta": 1,
        },
    )


class PDHGSolver(ADMMSolver):
    algorithm='pdhg'


def admm(operator,y,**kwargs): return ADMMSolver(**kwargs).solve(y,operator)


def pdhg(operator,y,**kwargs): return PDHGSolver(**kwargs).solve(y,operator)


def _batch_inner(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left * right).reshape(left.shape[0], -1).sum(dim=1)


def _batch_norm(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(_batch_inner(x, x).clamp_min(0.0))


def _batch_view(coefficients: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return coefficients.reshape((coefficients.shape[0],) + (1,) * (target.ndim - 1))


def _validate_regularization_operator(
    reconstruction_operator: LinearOperator,
    regularization_operator: Optional[LinearOperator],
) -> None:
    if regularization_operator is None:
        return
    if not isinstance(regularization_operator, LinearOperator):
        raise TypeError(
            "regularization_operator must be a LinearOperator or None; "
            f"got {type(regularization_operator).__name__}."
        )
    if tuple(regularization_operator.domain_shape) != tuple(reconstruction_operator.domain_shape):
        raise ValueError(
            "regularization_operator.domain_shape must match operator.domain_shape; "
            f"got {tuple(regularization_operator.domain_shape)} and "
            f"{tuple(reconstruction_operator.domain_shape)}."
        )


def tikhonov(
    operator: LinearOperator,
    y: torch.Tensor,
    reg_strength: float = 1e-2,
    num_iterations: int = 100,
    tolerance: float = 1e-6,
    x_init: Optional[torch.Tensor] = None,
    regularization_operator: Optional[LinearOperator] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Solve a linear Tikhonov problem by batched conjugate gradients.

    The objective is ``0.5 ||Ax-y||^2 + 0.5 * reg_strength ||Lx||^2``.
    ``L`` is the identity unless ``regularization_operator`` is supplied.
    Optional box constraints are applied once to the converged result; they do
    not alter the conjugate-gradient recurrence.
    """
    require_linear_operator(operator, "tikhonov")
    validate_measurement_shape(y, operator, "tikhonov")
    _validate_regularization_operator(operator, regularization_operator)
    if float(reg_strength) < 0.0:
        raise ValueError("reg_strength must be nonnegative.")
    if int(num_iterations) <= 0:
        raise ValueError("num_iterations must be positive.")
    if float(tolerance) < 0.0:
        raise ValueError("tolerance must be nonnegative.")

    regularizer = TikhonovRegularizer(regularization_operator)
    strength = float(reg_strength)
    epsilon = float(eps)

    def normal_equation(z: torch.Tensor) -> torch.Tensor:
        return operator.adjoint(operator.forward(z)) + strength * regularizer.gradient(z)

    x = prepare_initial_image(y, operator, x_init=x_init, initial_value=0.0)
    rhs = operator.adjoint(y)
    residual = rhs - normal_equation(x)
    direction = residual.clone()
    residual_sq = _batch_inner(residual, residual)
    threshold = float(tolerance) * _batch_norm(rhs).clamp_min(1.0)

    for _ in range(int(num_iterations)):
        if bool(torch.all(torch.sqrt(residual_sq.clamp_min(0.0)) <= threshold)):
            break
        normal_direction = normal_equation(direction)
        denominator = _batch_inner(direction, normal_direction)
        alpha = torch.where(
            denominator.abs() > epsilon,
            residual_sq / denominator,
            torch.zeros_like(denominator),
        )
        x = x + _batch_view(alpha, x) * direction
        next_residual = residual - _batch_view(alpha, residual) * normal_direction
        next_residual_sq = _batch_inner(next_residual, next_residual)
        beta = torch.where(
            residual_sq > epsilon,
            next_residual_sq / residual_sq,
            torch.zeros_like(residual_sq),
        )
        direction = next_residual + _batch_view(beta, direction) * direction
        residual = next_residual
        residual_sq = next_residual_sq

    return apply_box_constraints(x, min_value=min_value, max_value=max_value)


def _estimate_lipschitz(
    operator: LinearOperator,
    reference: torch.Tensor,
    num_iterations: int,
    eps: float = 1e-12,
) -> float:
    sample_size = 1
    for value in operator.domain_shape:
        sample_size *= int(value)
    seed = torch.linspace(
        0.5,
        1.5,
        steps=sample_size,
        dtype=reference.dtype,
        device=reference.device,
    ).reshape((1, *tuple(operator.domain_shape))).expand(reference.shape[0], *tuple(operator.domain_shape))
    vector = seed / _batch_view(_batch_norm(seed).clamp_min(eps), seed)

    for _ in range(int(num_iterations)):
        next_vector = operator.adjoint(operator.forward(vector))
        norm = _batch_norm(next_vector)
        if bool(torch.all(norm <= eps)):
            return 0.0
        vector = next_vector / _batch_view(norm.clamp_min(eps), next_vector)

    image = operator.adjoint(operator.forward(vector))
    return float(_batch_inner(vector, image).clamp_min(0.0).max().item())


def tv_fista(
    operator: LinearOperator,
    y: torch.Tensor,
    reg_strength: float = 1e-3,
    num_iterations: int = 50,
    step_size: Optional[float] = None,
    x_init: Optional[torch.Tensor] = None,
    regularizer: Optional[TVRegularizer] = None,
    tolerance: float = 1e-5,
    power_iterations: int = 12,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> torch.Tensor:
    """Legacy TV entry point using the shared composite FISTA core."""
    pass

    tv = regularizer if regularizer is not None else TVRegularizer()
    if not isinstance(tv, TVRegularizer):
        raise TypeError("regularizer must be a TVRegularizer")
    return fista(operator, y, regularizer=tv, reg_strength=reg_strength,
                 num_iterations=num_iterations, step_size=step_size, x_init=x_init,
                 tolerance=tolerance, power_iterations=power_iterations,
                 min_value=min_value, max_value=max_value, legacy=True)


class TikhonovSolver(InverseProblemSolver):
    """Unified solver wrapper for conjugate-gradient Tikhonov reconstruction."""

    def __init__(
        self,
        reg_strength: float = 1e-2,
        num_iterations: int = 100,
        tolerance: float = 1e-6,
        regularization_operator: Optional[LinearOperator] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ):
        self.reg_strength = float(reg_strength)
        self.num_iterations = int(num_iterations)
        self.tolerance = float(tolerance)
        self.regularization_operator = regularization_operator
        self.min_value = min_value
        self.max_value = max_value

    def solve(
        self,
        measurement: torch.Tensor,
        operator: ForwardOperator,
        x_init: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if kwargs.get('fidelity') is not None:
            result=self.solve_detailed(measurement,operator,x_init=x_init,**kwargs)
            if result.status in ('invalid_parameters','numerical_error'):raise ValueError(result.stopping_reason)
            return result.reconstruction
        return tikhonov(
            operator,
            measurement,
            reg_strength=self.reg_strength,
            num_iterations=self.num_iterations,
            tolerance=self.tolerance,
            x_init=x_init,
            regularization_operator=self.regularization_operator,
            min_value=self.min_value,
            max_value=self.max_value,
        )

    def solve_detailed(
        self,
        measurement: torch.Tensor,
        operator: ForwardOperator,
        x_init: Optional[torch.Tensor] = None,
        *,
        control=None,
        callback=None,
        **kwargs,
    ):
        pass

        fidelity=kwargs.get("fidelity")
        fidelity_before={k:getattr(fidelity,k,0) for k in ("weight_multiplications","prox_calls","prox_seconds")}
        auxiliary = kwargs.get("regularization_operator", self.regularization_operator)
        if fidelity is not None and kwargs.get('min_value',self.min_value) is None and kwargs.get('max_value',self.max_value) is None and callable(getattr(fidelity,'validate_normal',None)):
            try:
                reference=prepare_initial_image(measurement,operator,x_init=x_init,initial_value=0.)
                fidelity.validate_normal(operator,auxiliary,reference)
            except ValueError as error:
                from inv_framework.solvers._iteration import _parameter_failure_result
                return _parameter_failure_result('tikhonov',measurement,operator,x_init=x_init,errors=[str(error)],max_iterations=self.num_iterations,parameters={})

        result = solve_tikhonov_detailed(
            operator,
            measurement,
            reg_strength=kwargs.pop("reg_strength", self.reg_strength),
            num_iterations=kwargs.pop("num_iterations", self.num_iterations),
            tolerance=kwargs.pop("tolerance", self.tolerance),
            x_init=x_init,
            regularization_operator=kwargs.pop("regularization_operator", self.regularization_operator),
            min_value=kwargs.pop("min_value", self.min_value),
            max_value=kwargs.pop("max_value", self.max_value),
            control=control,
            callback=callback,
            fidelity=kwargs.pop("fidelity",None),
        )
        if fidelity is not None:
            result=replace(result,resources={**result.resources,**{'fidelity_'+k:getattr(fidelity,k,0)-v for k,v in fidelity_before.items()}},metadata={**result.metadata,'regularized_normal_residual_formula':'||A* gradient f(Ax)+lambda gradient R(x)|| / max(||A* linear_rhs(f)||+lambda ||gradient R(x)||,eps)'})
        return result


class TVFISTASolver(InverseProblemSolver):
    """Unified solver wrapper for TV-regularized FISTA reconstruction."""

    def __init__(
        self,
        reg_strength: float = 1e-3,
        num_iterations: int = 50,
        step_size: Optional[float] = None,
        regularizer: Optional[TVRegularizer] = None,
        tolerance: float = 1e-5,
        power_iterations: int = 12,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ):
        self.reg_strength = float(reg_strength)
        self.num_iterations = int(num_iterations)
        self.step_size = step_size
        self.regularizer = regularizer or TVRegularizer()
        self.tolerance = float(tolerance)
        self.power_iterations = int(power_iterations)
        self.min_value = min_value
        self.max_value = max_value

    def solve(
        self,
        measurement: torch.Tensor,
        operator: ForwardOperator,
        x_init: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        return tv_fista(
            operator,
            measurement,
            reg_strength=self.reg_strength,
            num_iterations=self.num_iterations,
            step_size=self.step_size,
            x_init=x_init,
            regularizer=self.regularizer,
            tolerance=self.tolerance,
            power_iterations=self.power_iterations,
            min_value=self.min_value,
            max_value=self.max_value,
        )

    def solve_detailed(
        self,
        measurement: torch.Tensor,
        operator: ForwardOperator,
        x_init: Optional[torch.Tensor] = None,
        *,
        control=None,
        callback=None,
        **kwargs,
    ):
        pass

        return solve_tv_fista_detailed(
            operator,
            measurement,
            reg_strength=kwargs.pop("reg_strength", self.reg_strength),
            num_iterations=kwargs.pop("num_iterations", self.num_iterations),
            step_size=kwargs.pop("step_size", self.step_size),
            x_init=x_init,
            regularizer=kwargs.pop("regularizer", self.regularizer),
            tolerance=kwargs.pop("tolerance", self.tolerance),
            power_iterations=kwargs.pop("power_iterations", self.power_iterations),
            min_value=kwargs.pop("min_value", self.min_value),
            max_value=kwargs.pop("max_value", self.max_value),
            control=control,
            callback=callback,
        )


class ConstrainedTVSolver(InverseProblemSolver):
    """Explicit-radius constrained TV solver; no automatic radius."""

    def __init__(
        self,
        *,
        epsilon,
        epsilon_source,
        algorithm="pdhg",
        tv_mode="isotropic",
        num_iterations=2000,
        tolerance=1e-05,
        feasibility_tolerance=1e-05,
        relaxation=1.0,
        relaxation_decay=0.99,
        tv_step=0.2,
        tv_step_decay=0.95,
        tv_updates=20,
        update_ratio=0.95,
        block_size=1,
        residual_metric="raw",
        whitening=None
    ):
        if algorithm not in ("pdhg", "asd_pocs_tv"):
            raise ValueError("unsupported constrained TV algorithm")
        for name, value in dict(
            tolerance=tolerance,
            feasibility_tolerance=feasibility_tolerance,
            relaxation=relaxation,
            tv_step=tv_step,
            update_ratio=update_ratio,
        ).items():
            if not math.isfinite(float(value)) or value <= 0:
                raise ValueError(name + " must be positive finite")
        for value in (relaxation_decay, tv_step_decay):
            if not math.isfinite(float(value)) or not 0 < value < 1:
                raise ValueError("decay must lie in (0,1)")
        for value in (num_iterations, tv_updates, block_size):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("iteration and subset sizes must be positive integers")
        if (
            isinstance(epsilon, bool)
            or not math.isfinite(float(epsilon))
            or epsilon < 0
        ):
            raise ValueError("epsilon must be nonnegative finite")
        if not isinstance(epsilon_source, str) or not epsilon_source.strip():
            raise ValueError("epsilon source must be explicit")
        if relaxation > 1:
            raise ValueError("relaxation must be <= 1")
        self.parameters = dict(
            epsilon=epsilon,
            epsilon_source=epsilon_source,
            algorithm=algorithm,
            tv_mode=tv_mode,
            num_iterations=num_iterations,
            tolerance=tolerance,
            feasibility_tolerance=feasibility_tolerance,
            relaxation=relaxation,
            relaxation_decay=relaxation_decay,
            tv_step=tv_step,
            tv_step_decay=tv_step_decay,
            tv_updates=tv_updates,
            update_ratio=update_ratio,
            block_size=block_size,
        )
        self.regularizer = TVRegularizer(tv_mode)
        if residual_metric not in ("raw", "gaussian_whitened"):
            raise ValueError("unknown residual metric")
        self.parameters["residual_metric"] = residual_metric
        self.whitening = whitening

    def solve(self, measurement, operator, x_init=None, **kwargs):
        return self.solve_detailed(
            measurement, operator, x_init=x_init, **kwargs
        ).reconstruction

    def solve_detailed(
        self,
        measurement,
        operator,
        x_init=None,
        *,
        mask=None,
        control=None,
        callback=None
    ):
        cfg = self.parameters
        if cfg["residual_metric"] == "gaussian_whitened":
            from inv_framework.core.operator import WeightedLinearOperator

            if self.whitening is None:
                raise ValueError("gaussian_whitened residual requires known sigma")
            operator = WeightedLinearOperator(operator, self.whitening.square())
            measurement = operator.whiten(measurement)
        validate_measurement_shape(measurement, operator, cfg["algorithm"])
        if (
            len(operator.domain_shape) != 3
            or operator.domain_shape[0] != 1
            or measurement.shape[0] != 1
        ):
            raise ValueError(
                "constrained TV requires a single two-dimensional one-channel case"
            )
        ball = ResidualBall(
            measurement, cfg["epsilon"], source=cfg["epsilon_source"], mask=mask
        )
        control = resolve_control(
            control,
            default_iterations=cfg["num_iterations"],
            default_tolerance=cfg["tolerance"],
            callback=callback,
        )
        if control.discrepancy_target is not None:
            raise ValueError(
                "constrained TV uses epsilon; disable discrepancy stopping"
            )
        x = prepare_initial_image(
            measurement, operator, x_init=x_init, initial_value=0.0
        ).clamp_min(0)
        recorder = IterationRecorder(
            control, measurement, operator, algorithm=cfg["algorithm"]
        )
        recorder.set_initial(x)
        penalty = Penalty(self.regularizer, operator.domain_shape)
        status, reason = ("max_iterations", "maximum_iterations_reached")
        completed = 0
        endpoint = False
        prediction = tv = diagnostic = metric = None
        history = []
        projection_calls = 1
        monitor = ConsecutiveStoppingMonitor(control)
        tolerance = float(control.tolerance)
        beta, step = (cfg["relaxation"], None)
        try:
            if not bool(torch.isfinite(x).all()):
                raise FloatingPointError("non_finite_initialization")
            if cfg["algorithm"] == "pdhg":
                bound = norm_bound(operator, x) + penalty.bound
                if not math.isfinite(bound) or bound <= 0:
                    raise FloatingPointError("non_finite_operator_norm_bound")
                tau = sigma = 0.99 / math.sqrt(bound)
                p = torch.zeros_like(measurement)
                d = torch.zeros_like(penalty.apply(x))
                bar = x.clone()
            else:
                subsets = []
                for indices in make_angle_subsets(
                    measurement.shape[-2], cfg["block_size"], device=x.device
                ):
                    sub = make_subset_operator(operator, indices)
                    active = select_measurement_subset(ball.mask, indices)
                    row = (
                        _safe_reciprocal(sub.forward(torch.ones_like(x)), 1e-08)
                        * active
                    )
                    column = _safe_reciprocal(sub.adjoint(active.to(x.dtype)), 1e-08)
                    if not bool(
                        torch.isfinite(row).all() and torch.isfinite(column).all()
                    ):
                        raise FloatingPointError("non_finite_sart_normalization")
                    subsets.append(
                        (
                            sub,
                            select_measurement_subset(measurement, indices),
                            row,
                            column,
                        )
                    )
            for iteration in range(
                1, min(cfg["num_iterations"], control.max_iterations) + 1
            ):
                endpoint = False
                old = x
                if cfg["algorithm"] == "pdhg":
                    np = ball.dual_prox(p + sigma * operator.forward(bar), sigma)
                    nd = penalty.dual_prox(d + sigma * penalty.apply(bar), sigma, 1.0)
                    candidate = (
                        x - tau * (operator.adjoint(np) + penalty.apply(nd, True))
                    ).clamp_min(0)
                    projection_calls += 1
                else:
                    candidate = x.clone()
                    for sub, y, row, column in subsets:
                        candidate = (
                            candidate
                            - beta
                            * column
                            * sub.adjoint(row * (sub.forward(candidate) - y))
                        ).clamp_min(0)
                        projection_calls += 1
                    data_change = float((candidate - x).norm())
                    data_residual = float(
                        ball.residual(operator.forward(candidate)).norm()
                    )
                    if not math.isfinite(data_change + data_residual):
                        raise FloatingPointError("non_finite_data_update")
                    if step is None:
                        step = cfg["tv_step"] * max(
                            data_change, float(candidate.norm()) * cfg["tolerance"]
                        )
                    start_tv = candidate
                    shrinks = accepted = 0
                    for _ in range(cfg["tv_updates"]):
                        z = penalty.apply(candidate)
                        subgradient = self.regularizer.coefficient_subgradient(z)
                        gradient = penalty.apply(subgradient, True)
                        length = float(gradient.norm())
                        if not math.isfinite(length):
                            raise FloatingPointError("non_finite_tv_gradient")
                        if length == 0 or step == 0:
                            break
                        trial_step = step
                        for _ in range(40):
                            trial = (
                                candidate - trial_step * gradient / length
                            ).clamp_min(0)
                            projection_calls += 1
                            trial_tv = float(penalty.value(penalty.apply(trial)))
                            trial_residual = float(
                                ball.residual(operator.forward(trial)).norm()
                            )
                            if not math.isfinite(trial_tv + trial_residual):
                                raise FloatingPointError("non_finite_tv_trial")
                            ratio_ok = float((trial - start_tv).norm()) <= cfg[
                                "update_ratio"
                            ] * max(
                                data_change, cfg["tolerance"] * float(start_tv.norm())
                            )
                            feasible_ok = (
                                data_residual
                                > cfg["epsilon"] + cfg["feasibility_tolerance"]
                                or trial_residual
                                <= cfg["epsilon"] + cfg["feasibility_tolerance"]
                            )
                            if (
                                ratio_ok
                                and feasible_ok
                                and (trial_tv <= float(penalty.value(z)))
                            ):
                                candidate = trial
                                accepted += 1
                                break
                            trial_step *= cfg["tv_step_decay"]
                            shrinks += 1
                        else:
                            break
                        step = trial_step
                    history.append(
                        dict(
                            sweep=iteration,
                            relaxation=beta,
                            tv_step=step,
                            data_change=data_change,
                            tv_change=float((candidate - start_tv).norm()),
                            reductions=shrinks,
                            accepted_tv_updates=accepted,
                        )
                    )
                    beta *= cfg["relaxation_decay"]
                if not bool(torch.isfinite(candidate).all()):
                    raise FloatingPointError("non_finite_candidate")
                if cfg["algorithm"] == "pdhg":
                    bar, p, d = (2 * candidate - x, np, nd)
                x = candidate
                completed = iteration
                prediction = operator.forward(x)
                tx = penalty.apply(x)
                tv = float(penalty.value(tx))
                diagnostic = ball.diagnostics(prediction)
                feasible = (
                    diagnostic["constraint_violation"] <= cfg["feasibility_tolerance"]
                )
                if cfg["algorithm"] == "pdhg":
                    rx = (
                        x
                        - (
                            x - tau * (operator.adjoint(p) + penalty.apply(d, True))
                        ).clamp_min(0)
                    ) / tau
                    projection_calls += 1
                    rp = (p - ball.dual_prox(p + sigma * prediction, sigma)) / sigma
                    rd = (d - penalty.dual_prox(d + sigma * tx, sigma, 1.0)) / sigma
                    metric = float(
                        (
                            rx.square().sum() + rp.square().sum() + rd.square().sum()
                        ).sqrt()
                    ) / max(1.0, float(x.norm()), float(p.norm()), float(d.norm()))
                    optimal = metric <= tolerance
                else:
                    metric = float((x - old).norm()) / max(1.0, float(old.norm()))
                    optimal = metric <= tolerance and beta > 0.005
                if not math.isfinite(tv + metric + diagnostic["raw_residual_norm"]):
                    raise FloatingPointError("non_finite_diagnostics")
                endpoint = True
                decision = monitor.observe(
                    iteration,
                    criteria={"feasible": feasible, "stationary": optimal},
                    relative_change=float((x - old).norm())
                    / max(1.0, float(old.norm())),
                    monitor_value=metric,
                )
                if not recorder.record(
                    iteration,
                    x,
                    residual=ball.residual(prediction),
                    objective=tv,
                    algorithm_residual=metric,
                    criteria=decision.criteria if decision.checked else {},
                    stopping_candidate=decision.checked,
                    consecutive_criteria_count=decision.consecutive,
                    native_criterion_name=(
                        "primal_dual_fixed_point"
                        if cfg["algorithm"] == "pdhg"
                        else "relative_iterate_change"
                    ),
                    native_criterion_value=metric,
                    native_criterion_threshold=tolerance,
                    metadata={"checked": decision.checked},
                ):
                    status, reason = ("cancelled", "callback_cancelled")
                    break
                if decision.converged and control.stop_on_convergence:
                    status, reason = (
                        ("converged", "primal_dual_fixed_point")
                        if cfg["algorithm"] == "pdhg"
                        else ("stalled", "feasible_iterate_stagnation")
                    )
                    break
        except FloatingPointError as error:
            status, reason = ("numerical_error", str(error))
        result = recorder.finish(
            x,
            actual_iterations=completed,
            status=status,
            stopping_reason=reason,
            final_objective=tv if endpoint else None,
            final_residual=diagnostic["raw_residual_norm"] if endpoint else None,
            predicted_measurement=prediction if endpoint else None,
            resources={
                **penalty.calls,
                "nonnegative_projection_calls": projection_calls,
                "ball_prox_calls": ball.prox_calls,
                "ball_prox_seconds": ball.prox_seconds,
                "mask_calls": ball.mask_calls,
            },
            metadata=dict(
                objective_type="residual_ball_constrained_tv",
                variant_id=(
                    "sart2d_feasibility_backtracking_v1"
                    if cfg["algorithm"] == "asd_pocs_tv"
                    else "residual_ball_pdhg_v1"
                ),
                endpoint_diagnostics_complete=endpoint,
                parameters={**cfg, "tolerance": tolerance},
                adaptation_history=history,
                constraint_diagnostics=diagnostic if endpoint else None,
                feasibility_status=(
                    (
                        "feasible"
                        if diagnostic["constraint_violation"]
                        <= cfg["feasibility_tolerance"]
                        else "violated"
                    )
                    if endpoint
                    else "unconfirmed"
                ),
                optimization_residual=metric if endpoint else None,
                residual_metric=cfg["residual_metric"],
                nonnegative=bool((x >= 0).all()),
                native_termination_managed=True,
            ),
        )
        if not endpoint:
            result = replace(
                result,
                relative_iterate_change=None,
                metadata={
                    **result.metadata,
                    "terminal_record": None,
                    "patience_window": [],
                },
            )
        return result


def constrained_pdhg(measurement, operator, x_init=None, *, mask=None, **parameters):
    """Return nonnegative residual-ball TV reconstruction using PDHG."""
    return ConstrainedTVSolver(algorithm='pdhg', **parameters).solve(
        measurement, operator, x_init=x_init, mask=mask)


def asd_pocs_tv(measurement, operator, x_init=None, *, mask=None, **parameters):
    """Return the SART2D feasibility-backtracking TV variant reconstruction."""
    return ConstrainedTVSolver(algorithm='asd_pocs_tv', **parameters).solve(
        measurement, operator, x_init=x_init, mask=mask)


def proximal(
    regularizer,
    value,
    weight,
    operator,
    min_value=None,
    max_value=None,
    *,
    iteration=1,
    certified=False
):
    if weight == 0:
        return apply_box_constraints(value, min_value=min_value, max_value=max_value)
    if certified and (
        type(regularizer).__name__ == "TVRegularizer"
        or type(regularizer).__name__ == "HaarL1Regularizer"
        and (min_value is not None or max_value is not None)
    ):
        from inv_framework.core.regularizer import joint_prox
        import time

        started = time.perf_counter()
        try:
            return joint_prox(
                regularizer,
                value,
                weight,
                min_value=min_value,
                max_value=max_value,
                iteration=iteration,
                tolerance=getattr(regularizer, "tolerance", 1e-5),
                max_iterations=getattr(regularizer, "num_iterations", 1000),
                check_budget=getattr(operator, "check_wall_budget", None),
            )
        finally:
            elapsed = time.perf_counter() - started
            regularizer.prox_iterations = (
                getattr(regularizer, "prox_iterations", 0)
                + regularizer.last_prox_iterations
            )
            regularizer.prox_calls = getattr(regularizer, "prox_calls", 0) + 1
            regularizer.prox_seconds = (
                getattr(regularizer, "prox_seconds", 0.0) + elapsed
            )

    result = regularizer.proximal(value, weight)
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


def solve_fista_detailed(
    operator,
    measurement,
    *,
    regularizer=None,
    strength=None,
    reg_strength=None,
    num_iterations=50,
    step_size=None,
    x_init=None,
    tolerance=1e-05,
    power_iterations=12,
    min_value=None,
    max_value=None,
    control=None,
    callback=None,
    legacy=False,
    fidelity=None
):
    from inv_framework.solvers._iteration import _common_parameter_errors, _parameter_failure_result, _finite_nonnegative, _finite_positive, _numerical_result, _policy_active, _policy_status, _control_threshold
    pass
    from inv_framework.core.fidelity import TransmissionPoissonFidelity
    from inv_framework.core.regularizer import ZeroRegularizer
    from inv_framework.core.regularizer import ProximalAccuracyError

    poisson = isinstance(fidelity, TransmissionPoissonFidelity)
    algorithm = "tv_fista" if legacy else "fista"
    require_linear_operator(operator, algorithm)
    validate_measurement_shape(measurement, operator, algorithm)
    reg = regularizer if regularizer is not None else TikhonovRegularizer()
    errors = _common_parameter_errors(num_iterations, min_value, max_value)
    if strength is not None and reg_strength is not None and (strength != reg_strength):
        errors.append("strength and reg_strength conflict")
    weight = (
        strength
        if strength is not None
        else reg_strength if reg_strength is not None else 0.001 if legacy else 0.01
    )
    if weight == 0:
        reg = ZeroRegularizer()
    smooth_regularizer = type(reg).__name__ == "GradientHuberRegularizer" or (
        isinstance(reg, TikhonovRegularizer) and reg.operator is not None
    )
    if not _finite_nonnegative(weight):
        errors.append("strength must be finite and nonnegative")
    if not _finite_nonnegative(tolerance):
        errors.append("tolerance must be finite and nonnegative")
    if step_size is not None and (not _finite_positive(step_size)):
        errors.append("step_size must be finite and positive")
    if (
        isinstance(power_iterations, bool)
        or not isinstance(power_iterations, int)
        or power_iterations <= 0
    ):
        errors.append("power_iterations must be a positive integer")
    if not smooth_regularizer and (
        not getattr(reg, "has_full_proximal", False)
        or not callable(getattr(reg, "proximal", None))
    ):
        errors.append(
            "FISTA requires a full proximal; generalized Tikhonov/H1 is unsupported"
        )
    if not callable(getattr(reg, "value", None)):
        errors.append("unsupported FISTA regularizer")
    try:
        reg.validate_shape(operator.domain_shape)
    except ValueError as error:
        errors.append(str(error))
    if errors:
        return _parameter_failure_result(
            algorithm,
            measurement,
            operator,
            x_init=x_init,
            errors=errors,
            max_iterations=num_iterations if isinstance(num_iterations, int) else None,
            parameters={},
        )
    control = resolve_control(
        control,
        default_iterations=num_iterations,
        default_tolerance=tolerance,
        callback=callback,
    )
    limit = min(num_iterations, control.max_iterations)
    x = prepare_initial_image(measurement, operator, x_init=x_init, initial_value=0.0)
    x = apply_box_constraints(x, min_value=min_value, max_value=max_value)
    if not bool(torch.isfinite(measurement).all() and torch.isfinite(x).all()):
        return _numerical_result(
            algorithm,
            measurement,
            operator,
            x_init=x_init,
            max_iterations=limit,
            reason="non_finite_input",
            parameters={},
        )
    recorder = IterationRecorder(control, measurement, operator, algorithm=algorithm)
    recorder.set_initial(x)
    monitor = ConsecutiveStoppingMonitor(control)
    momentum, acceleration = (x.clone(), 1.0)
    actual = 0
    work = {"backtracking_rejections": 0, "backtracking_trials": 0}
    status, reason = ("max_iterations", "maximum_iterations_reached")
    prediction = residual = objective = mapping = None
    endpoint_complete = False
    budget_interruption = None
    estimate = None
    previous_objective = None
    plateau_run = 0
    norm = lambda v: float(v.norm())
    measurement_norm = max(norm(measurement), 1e-12)
    legacy_trace = legacy and (not _policy_active(control))
    fidelity_before = {
        k: getattr(fidelity, k, 0)
        for k in ("weight_multiplications", "prox_calls", "prox_seconds")
    }
    work_before = {
        name: getattr(reg, name, 0)
        for name in (
            "prox_calls",
            "prox_iterations",
            "prox_seconds",
            "gradient_calls",
            "gradient_adjoint_calls",
        )
    }
    auxiliary = getattr(reg, "operator", None)
    haar_before = (
        getattr(reg, "analysis_calls", 0),
        getattr(reg, "synthesis_calls", 0),
        getattr(reg, "transform_seconds", 0),
    )

    def endpoint(point, point_residual, current_step):
        gradient = operator.adjoint(
            fidelity.gradient(point_residual + measurement)
            if fidelity is not None
            else point_residual
        )
        if smooth_regularizer and weight:
            gradient = gradient + weight * reg.gradient(point)
        prox = proximal(
            reg,
            point - current_step * gradient,
            0.0 if smooth_regularizer else current_step * weight,
            operator,
            min_value,
            max_value,
            iteration=max(actual, 1),
            certified=not legacy,
        )
        if not bool(torch.isfinite(gradient).all() and torch.isfinite(prox).all()):
            raise FloatingPointError("non_finite_endpoint_mapping")
        value = norm((point - prox) / current_step) / max(
            norm(point) / current_step, 1e-12
        )
        if not math.isfinite(value):
            raise FloatingPointError("non_finite_endpoint_mapping")
        return value

    try:
        if step_size is None:
            if poisson:
                pass

                estimate = (
                    norm_bound(operator, x)
                    * float((fidelity.I0 * fidelity.mask).max())
                    * fidelity.scale**2
                )
            else:
                estimate = _estimate_lipschitz(
                    (
                        fidelity.whiten_operator(operator)
                        if fidelity is not None
                        else operator
                    ),
                    measurement,
                    power_iterations,
                )
            if smooth_regularizer and weight:
                estimate += 8 * weight / getattr(reg, "delta", 1.0)
        else:
            estimate = None
        step = (
            float(step_size)
            if step_size is not None
            else 0.99 / estimate if estimate > 1e-12 else 1.0
        )
        if not math.isfinite(step):
            raise FloatingPointError("non_finite_initial_step")
        for iteration in range(1, limit + 1):
            if poisson and bool((momentum < 0).any()):
                momentum, acceleration = (x.clone(), 1.0)
            r = operator.forward(momentum) - measurement
            g = operator.adjoint(
                fidelity.gradient(r + measurement) if fidelity is not None else r
            )
            if smooth_regularizer and weight:
                g = g + weight * reg.gradient(momentum)
            if previous_objective is None:
                previous_objective = (
                    float(fidelity.value(r + measurement))
                    if fidelity is not None
                    else 0.5 * norm(r) ** 2
                ) + weight * float(reg.value(x).sum())
            next_x, next_residual, step, _ = fista_update(
                operator,
                measurement,
                momentum,
                g,
                r,
                step,
                reg,
                weight,
                min_value,
                max_value,
                work,
                fidelity,
                smooth_regularizer,
                iteration,
                not legacy,
            )
            old_x, x = (x, next_x)
            actual = iteration
            residual, prediction = (next_residual, next_residual + measurement)
            objective = (
                float(fidelity.value(prediction))
                if fidelity is not None
                else 0.5 * norm(residual) ** 2
            ) + weight * float(reg.value(x).sum())
            if not math.isfinite(objective):
                raise FloatingPointError("non_finite_composite_objective")
            endpoint_complete = False
            change = norm(x - old_x) / max(norm(old_x), 1e-12)
            mapping = endpoint(x, residual, step) if not legacy_trace else None
            endpoint_complete = mapping is not None
            objective_change = (
                math.inf
                if previous_objective is None
                else abs(objective - previous_objective)
                / max(abs(previous_objective), 1e-12)
            )
            mapping_tol = _control_threshold(
                control,
                "prox_gradient_mapping_tolerance",
                float(control.tolerance or 0),
            )
            objective_tol = _control_threshold(
                control, "relative_objective_tolerance", float(control.tolerance or 0)
            )
            criteria = {
                "prox_gradient_mapping": mapping is not None and mapping <= mapping_tol,
                "relative_composite_objective_change": objective_change
                <= objective_tol,
            }
            if control.discrepancy_target is not None:
                criteria["discrepancy"] = (
                    norm(residual) / measurement_norm <= control.discrepancy_target
                )
            decision = monitor.observe(
                iteration,
                criteria=criteria,
                relative_change=change,
                monitor_value=objective,
            )
            plateau = (
                control.stall_enabled
                and decision.checked
                and (criteria.get("discrepancy") is False)
                and criteria["prox_gradient_mapping"]
                and criteria["relative_composite_objective_change"]
                and (change <= control.stall_relative_iterate_tolerance)
            )
            if decision.checked:
                plateau_run = plateau_run + 1 if plateau else 0
            record_x = momentum if legacy_trace else x
            record_r = r if legacy_trace else residual
            record_obj = (
                (
                    float(fidelity.value(r + measurement))
                    if fidelity is not None
                    else 0.5 * norm(r) ** 2
                )
                + weight * float(reg.value(momentum).sum())
                if legacy_trace
                else objective
            )
            keep_going = recorder.record(
                iteration,
                record_x,
                residual=record_r,
                objective=record_obj,
                algorithm_residual=mapping,
                step_size=step,
                stopping_candidate=decision.converged,
                consecutive_criteria_count=decision.consecutive,
                criteria=criteria,
                native_criterion_name="normalized_prox_gradient_mapping",
                native_criterion_value=mapping,
                native_criterion_threshold=mapping_tol,
                metadata={
                    "objective_state": "momentum" if legacy_trace else "returned",
                    "mapping_state": "returned",
                    "complete_objective": True,
                    "normalized_prox_gradient_mapping": mapping,
                    "relative_composite_objective_change": (
                        None
                        if not math.isfinite(objective_change)
                        else objective_change
                    ),
                    "native_plateau": bool(
                        control.stall_enabled
                        and criteria.get("discrepancy") is False
                        and criteria["prox_gradient_mapping"]
                        and criteria["relative_composite_objective_change"]
                    ),
                    "checked": decision.checked,
                    "native_plateau_consecutive": plateau_run,
                },
            )
            previous_objective = objective
            if not keep_going:
                status, reason = ("cancelled", "callback_cancelled")
                break
            terminal, terminal_reason = _policy_status(decision)
            if terminal or plateau_run >= control.stall_patience:
                status, reason = (
                    terminal or "stalled",
                    terminal_reason or "stalled_before_discrepancy",
                )
                break
            if decision.converged and control.stop_on_convergence:
                status, reason = (
                    "converged",
                    "discrepancy_prox_gradient_and_objective_patience",
                )
                break
            if (
                legacy_trace
                and float(control.tolerance or 0) > 0
                and bool(
                    torch.all(
                        (x - old_x).flatten(1).norm(dim=1)
                        <= control.tolerance
                        * old_x.flatten(1).norm(dim=1).clamp_min(1.0)
                    )
                )
            ):
                status, reason = ("stalled", "legacy_iterate_tolerance")
                break
            next_acceleration = 0.5 * (1 + math.sqrt(1 + 4 * acceleration**2))
            momentum = x + (acceleration - 1) / next_acceleration * (x - old_x)
            acceleration = next_acceleration
        if not endpoint_complete:
            if residual is None:
                prediction = operator.forward(x)
                residual = prediction - measurement
            mapping = endpoint(x, residual, step)
            endpoint_complete = True
    except ProximalAccuracyError as error:
        reason = str(error)
        status = "stalled" if reason == "joint_prox_returned_precision" else "max_iterations"
    except FloatingPointError as error:
        status, reason = ("numerical_error", str(error))
        recorder.mark_numerical_error(reason)
    resources = {
        **{
            "fidelity_" + k: getattr(fidelity, k, 0) - v
            for k, v in fidelity_before.items()
        },
        **{name: getattr(reg, name, 0) - value for name, value in work_before.items()},
        **work,
        "haar_analysis_calls": getattr(reg, "analysis_calls", 0) - haar_before[0],
        "haar_synthesis_calls": getattr(reg, "synthesis_calls", 0) - haar_before[1],
        "haar_transform_seconds": getattr(reg, "transform_seconds", 0) - haar_before[2],
    }
    return recorder.finish(
        x,
        actual_iterations=actual,
        status=status,
        stopping_reason=reason,
        final_residual=None if residual is None else norm(residual) / measurement_norm,
        final_objective=objective,
        predicted_measurement=prediction,
        resources=resources,
        metadata={
            "step_size": locals().get("step"),
            "operator_norm_squared": estimate,
            "norm_estimate_is_upper_bound": False,
            "smooth_majorizer_checked": actual > 0,
            "endpoint_mapping_complete": endpoint_complete,
            "budget_interruption": budget_interruption,
            "normalized_prox_gradient_mapping": mapping if endpoint_complete else None,
            "trajectory_state": "momentum" if legacy_trace else "returned",
            "proximal_accuracy": reg.proximal_accuracy,
            "joint_prox_gap": getattr(reg, "last_prox_gap", None),
            "joint_prox_error_bound": getattr(reg, "last_prox_error_bound", None),
            "joint_prox_target": getattr(reg, "last_prox_target", None),
            "legacy_box_projection": legacy
            and (min_value is not None or max_value is not None),
            "strength": weight,
            "regularizer": type(reg).__name__,
            "level": getattr(reg, "level", None),
            "tv_num_iterations": getattr(reg, "num_iterations", None),
            "tv_tolerance": getattr(reg, "tolerance", None),
            "native_plateau_stall": {
                "enabled": control.stall_enabled,
                "consecutive": plateau_run,
                "patience": control.stall_patience,
                "relative_iterate_tolerance": control.stall_relative_iterate_tolerance,
            },
            "native_termination_managed": True,
        },
    )


def fista(operator, y, **kwargs):
    result = solve_fista_detailed(operator, y, **kwargs)
    if result.status == "invalid_parameters":
        raise ValueError(result.stopping_reason)
    if result.status == "numerical_error":
        raise FloatingPointError(result.stopping_reason)
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


def _estimate_lipschitz_for_detail(operator: LinearOperator, reference: torch.Tensor, iterations: int) -> float:
    pass

    return float(_estimate_lipschitz(operator, reference, int(iterations)))


def solve_tikhonov_detailed(
    operator: LinearOperator,
    measurement: torch.Tensor,
    *,
    reg_strength: float = 1e-2,
    num_iterations: int = 100,
    tolerance: float = 1e-6,
    x_init: torch.Tensor | None = None,
    regularization_operator: LinearOperator | None = None,
    min_value: float | None = None,
    max_value: float | None = None,
    fidelity=None,
    eps: float = 1e-12,
    control: SolveControl | None = None,
    callback: Callback | None = None,
) -> SolveResult:
    from inv_framework.solvers._iteration import _common_parameter_errors, _control_threshold, _finish_prediction, _finish_status, _finite_nonnegative, _finite_positive, _finite_tensor, _global_norm, _normalized, _numerical_result, _objective, _parameter_failure_result, _policy_active, _policy_status, _relative_change, _safe_record, _sqnorm, _tolerance_reached
    require_linear_operator(operator, "tikhonov")
    validate_measurement_shape(measurement, operator, "tikhonov")
    if fidelity is not None and not all(callable(getattr(fidelity,k,None)) for k in ('linear_rhs','normal_gradient')):
        raise ValueError('Tikhonov CG requires quadratic fidelity')
    data_value = lambda residual: float(fidelity.value(residual+measurement)) if fidelity is not None else _objective(residual)
    normal_data = lambda prediction: fidelity.normal_gradient(prediction) if fidelity is not None else prediction
    parameter_errors = _common_parameter_errors(num_iterations, min_value, max_value)
    if fidelity is not None and (min_value is not None or max_value is not None):
        parameter_errors.append("weighted Tikhonov CG supports unconstrained reconstruction only")
    if not _finite_nonnegative(reg_strength):
        parameter_errors.append("reg_strength must be a finite nonnegative number")
    if not _finite_nonnegative(tolerance):
        parameter_errors.append("tolerance must be a finite nonnegative number")
    if not _finite_positive(eps):
        parameter_errors.append("eps must be a finite positive number")
    if regularization_operator is not None and not isinstance(regularization_operator, LinearOperator):
        parameter_errors.append("regularization_operator must be a LinearOperator or None")
    elif (
        regularization_operator is not None
        and tuple(regularization_operator.domain_shape) != tuple(operator.domain_shape)
    ):
        parameter_errors.append(
            "regularization_operator.domain_shape must match operator.domain_shape"
        )
    if parameter_errors:
        return _parameter_failure_result(
            "tikhonov", measurement, operator, x_init=x_init,
            errors=parameter_errors,
            max_iterations=num_iterations if isinstance(num_iterations, int) and not isinstance(num_iterations, bool) else None,
            parameters={
                "num_iterations": num_iterations,
                "reg_strength": reg_strength,
                "tolerance": tolerance,
                "min_value": min_value,
                "max_value": max_value,
                "eps": eps,
            },
        )
    control = resolve_control(control, default_iterations=int(num_iterations), default_tolerance=float(tolerance), callback=callback)
    limit = min(int(num_iterations), int(control.max_iterations or num_iterations))
    # Auxiliary L work is separate from the primary A/A* operator budget.
    regularizer = TikhonovRegularizer(regularization_operator)
    strength = float(reg_strength)
    parameters = {
        "num_iterations": int(num_iterations),
        "reg_strength": strength,
        "tolerance": float(tolerance),
        "min_value": min_value,
        "max_value": max_value,
        "eps": float(eps),
    }
    if not _finite_tensor(measurement):
        return _numerical_result(
            "tikhonov", measurement, operator, x_init=x_init, max_iterations=limit,
            reason="non_finite_measurement", parameters=parameters,
        )
    x = prepare_initial_image(measurement, operator, x_init=x_init, initial_value=0.0)
    if not _finite_tensor(x):
        return _numerical_result(
            "tikhonov", measurement, operator, x_init=x_init, max_iterations=limit,
            reason="non_finite_initial_reconstruction", parameters=parameters,
        )
    recorder = IterationRecorder(control, measurement, operator, algorithm="tikhonov")
    monitor = ConsecutiveStoppingMonitor(control)
    recorder.set_initial(x)
    # Cache A*x and A*d.  The cached predictions keep the per-iteration
    # objective exact while avoiding a second forward call solely for
    # diagnostics.  A final explicit endpoint call remains part of the
    # accounting contract.  Box constraints invalidate the linear update,
    # so that less common path recomputes the prediction safely.
    cache_prediction = min_value is None and max_value is None
    prediction = operator.forward(x)
    rhs = operator.adjoint(fidelity.linear_rhs() if fidelity is not None else measurement)
    normal_residual = rhs - (operator.adjoint(normal_data(prediction)) + strength * regularizer.gradient(x))
    if not (
        _finite_tensor(prediction)
        and _finite_tensor(rhs)
        and _finite_tensor(normal_residual)
    ):
        recorder.mark_numerical_error("non_finite_solver_state")
    direction = normal_residual.clone()
    residual_sq = _sqnorm(normal_residual)
    rhs_norm = max(_global_norm(rhs), 1.0)
    converged = False
    cancelled = False
    terminal_status = None
    terminal_reason = None
    for iteration in range(1, limit + 1):
        if recorder.numerical_failure:
            break
        if converged and control.stop_on_convergence:
            break
        normal_value = _global_norm(normal_residual)
        if not _policy_active(control) and float(control.tolerance or 0.0) > 0.0 and _normalized(normal_value, rhs_norm) <= float(control.tolerance):
            converged = True
            break
        direction_prediction = operator.forward(direction)
        normal_direction = operator.adjoint(normal_data(direction_prediction)) + strength * regularizer.gradient(direction)
        denominator = (direction * normal_direction).reshape(direction.shape[0], -1).sum(dim=1)
        if (
            not _finite_tensor(direction_prediction)
            or not _finite_tensor(normal_direction)
            or not bool(torch.isfinite(denominator).all())
        ):
            recorder.mark_numerical_error("non_finite_normal_equation")
            break
        if bool(torch.any(denominator.abs() <= float(eps))):
            # Conjugate-gradient breakdown after the normal residual has
            # already reached zero is a harmless exact-solution plateau.  In
            # fixed-compute mode keep the remaining counted iterations and
            # report the configured budget stop; a nonzero residual remains a
            # genuine numerical failure.
            # Floating point CG can leave a tiny nonzero normal residual even
            # for an exact solution (for example, identity operators with a
            # positive Tikhonov term).  Treat that machine-precision plateau
            # as a benign breakdown, but keep a materially nonzero residual
            # as a numerical failure.
            if _global_norm(normal_residual) <= max(float(eps), 1e-6 * rhs_norm):
                # A stationary normal equation with no usable CG direction
                # is benign.  It may still miss the discrepancy target (for
                # instance, an inconsistent or zero operator), so do not
                # promote it to convergence.
                if _policy_active(control):
                    residual = prediction - measurement
                    data_relative = _normalized(_global_norm(residual), _global_norm(measurement))
                    regularization_gradient = regularizer.gradient(x)
                    reg_denominator = _global_norm(rhs) + strength * _global_norm(regularization_gradient)
                    reg_normalized = normal_value / max(reg_denominator, float(eps))
                    normal_tolerance_policy = _control_threshold(
                        control, "normalized_normal_residual_tolerance"
                    )
                    iterate_tolerance_policy = _control_threshold(
                        control, "relative_iterate_tolerance"
                    )
                    criteria = {
                        "regularized_normal_residual": reg_normalized <= normal_tolerance_policy,
                        "relative_iterate_change": 0.0 <= iterate_tolerance_policy,
                    }
                    if control.discrepancy_target is not None:
                        criteria = {
                            "discrepancy": data_relative <= float(control.discrepancy_target),
                            **criteria,
                        }
                    objective = data_value(residual) + strength * float(regularizer.value(x).sum().item())
                    decision = monitor.observe(
                        iteration,
                        criteria=criteria,
                        relative_change=0.0,
                        monitor_value=objective,
                    )
                    converged = decision.converged
                    terminal_status, terminal_reason = _policy_status(decision)
                    actual = iteration
                    if not _safe_record(
                        recorder,
                        iteration,
                        x,
                        residual=residual,
                        objective=objective,
                        algorithm_residual=reg_normalized,
                        stopping_candidate=converged,
                        consecutive_criteria_count=decision.consecutive,
                        criteria=criteria,
                        native_criterion_name="normalized_regularized_normal_residual",
                        native_criterion_value=reg_normalized,
                        native_criterion_threshold=normal_tolerance_policy,
                        metadata={
                            "normal_residual": normal_value,
                            "normal_residual_denominator": reg_denominator,
                            "complete_objective": True,
                            "regularization_operator": "identity" if regularization_operator is None else "linear_operator",
                            "regularized_normal_residual_formula": (
                                "||A^T(Ax-b) + lambda L^T Lx||_2 / "
                                "max(||A^T b||_2 + lambda ||L^T Lx||_2, eps)"
                            ),
                            "breakdown": "stationary_normal_equation",
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
                # Preserve the fixed-compute legacy accounting contract for
                # exact/stationary plateaus: each counted iteration still
                # produces a diagnostic row and consumes its native operator
                # calls.  This keeps the detailed wrapper comparable with
                # the original solver even though no further CG step exists.
                prediction = _finish_prediction(operator, x) if not cache_prediction else prediction
                residual = prediction - measurement
                actual = iteration
                if not _safe_record(
                    recorder,
                    iteration,
                    x,
                    residual=residual,
                    objective=data_value(residual) + strength * float(regularizer.value(x).sum().item()),
                    algorithm_residual=0.0,
                    stopping_candidate=False,
                    metadata={"normal_residual": 0.0, "breakdown": "exact_solution_plateau"},
                ):
                    cancelled = True
                    break
                continue
            recorder.mark_numerical_error("normal_equation_breakdown")
            break
        alpha = residual_sq / denominator
        if not bool(torch.isfinite(alpha).all()):
            recorder.mark_numerical_error("non_finite_normal_equation_step")
            break
        previous = x
        x = x + alpha.reshape((alpha.shape[0],) + (1,) * (x.ndim - 1)) * direction
        x = apply_box_constraints(x, min_value=min_value, max_value=max_value)
        if cache_prediction:
            next_normal_residual = normal_residual - alpha.reshape((alpha.shape[0],) + (1,) * (normal_residual.ndim - 1)) * normal_direction
            next_residual_sq = _sqnorm(next_normal_residual)
            beta = next_residual_sq / residual_sq.clamp_min(float(eps))
            direction = next_normal_residual + beta.reshape((beta.shape[0],) + (1,) * (direction.ndim - 1)) * direction
            normal_residual = next_normal_residual
            residual_sq = next_residual_sq
            prediction = prediction + alpha.reshape((alpha.shape[0],) + (1,) * (prediction.ndim - 1)) * direction_prediction
            # The unconstrained CG recurrence is exact for this linear
            # normal equation.  With a supplied box constraint the projection
            # destroys that recurrence; recompute the native state instead.
        else:
            prediction = operator.forward(x)
            exact_normal_residual = rhs - (
                operator.adjoint(normal_data(prediction)) + strength * regularizer.gradient(x)
            )
            exact_residual_sq = _sqnorm(exact_normal_residual)
            if not _finite_tensor(exact_normal_residual) or not bool(torch.isfinite(exact_residual_sq).all()):
                recorder.mark_numerical_error("non_finite_normal_equation")
                break
            beta = exact_residual_sq / residual_sq.clamp_min(float(eps))
            direction = exact_normal_residual + beta.reshape((beta.shape[0],) + (1,) * (direction.ndim - 1)) * direction
            if not _finite_tensor(direction) or not bool(torch.isfinite(beta).all()):
                recorder.mark_numerical_error("non_finite_normal_equation_step")
                break
            normal_residual = exact_normal_residual
            residual_sq = exact_residual_sq
        residual = prediction - measurement
        objective = data_value(residual) + strength * float(regularizer.value(x).sum().item())
        normal_value = _global_norm(normal_residual)
        normal_relative = _normalized(normal_value, rhs_norm)
        change = _relative_change(previous, x)
        data_relative = _normalized(_global_norm(residual), _global_norm(measurement))
        regularization_gradient = regularizer.gradient(x)
        reg_denominator = _global_norm(rhs) + strength * _global_norm(regularization_gradient)
        reg_normalized = normal_value / max(reg_denominator, float(eps))
        normal_tolerance_policy = _control_threshold(
            control, "normalized_normal_residual_tolerance"
        )
        iterate_tolerance_policy = _control_threshold(
            control, "relative_iterate_tolerance"
        )
        criteria = {
            "regularized_normal_residual": reg_normalized <= normal_tolerance_policy,
            "relative_iterate_change": change <= iterate_tolerance_policy,
        }
        if control.discrepancy_target is not None:
            criteria = {
                "discrepancy": data_relative <= float(control.discrepancy_target),
                **criteria,
            }
        decision = monitor.observe(iteration, criteria=criteria, relative_change=change, monitor_value=objective)
        converged = decision.converged if _policy_active(control) else float(control.tolerance or 0.0) > 0.0 and _tolerance_reached(control, residual=normal_relative, change=change, require_both=True)
        terminal_status, terminal_reason = _policy_status(decision) if _policy_active(control) else (None, None)
        actual = iteration
        if not _safe_record(
            recorder,
            iteration,
            x,
            residual=residual,
            objective=objective,
            algorithm_residual=reg_normalized if _policy_active(control) else normal_relative,
            stopping_candidate=converged,
            consecutive_criteria_count=decision.consecutive,
            criteria=criteria,
            native_criterion_name="normalized_regularized_normal_residual",
            native_criterion_value=reg_normalized,
            native_criterion_threshold=(
                normal_tolerance_policy
                if _policy_active(control)
                else control.normalized_normal_residual_tolerance
            ),
            metadata={
                "normal_residual": normal_value,
                "normal_residual_denominator": reg_denominator,
                "complete_objective": True,
                "regularization_operator": "identity" if regularization_operator is None else "linear_operator",
                "regularized_normal_residual_formula": (
                    "||A^T(Ax-b) + lambda L^T Lx||_2 / "
                    "max(||A^T b||_2 + lambda ||L^T Lx||_2, eps)"
                ),
            },
        ):
            cancelled = True
            break
        if terminal_status or (converged and control.stop_on_convergence):
            break
    # Keep the final evaluation explicit, even though the prediction above is
    # already exact in the unconstrained linear path.  This distinguishes the
    # solver trajectory from the endpoint diagnostics and gives the equal-call
    # protocol its advertised n+2 forward / n+2 adjoint shape.
    prediction = _finish_prediction(operator, x)
    residual = prediction - measurement
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
            "discrepancy_regularized_normal_and_iterate_patience" if status == "converged" and _policy_active(control) else
            "normal_residual_and_iterate_tolerance" if status == "converged" else
            "callback_cancelled" if cancelled else
            "maximum_iterations_reached"
        ),
        final_residual=_normalized(_global_norm(residual), _global_norm(measurement)),
        final_objective=data_value(residual) + strength * float(regularizer.value(x).sum().item()),
        predicted_measurement=prediction,
        metadata={
            "criterion": "normalized_regularized_normal_residual_and_relative_iterate_change",
            "regularization_strength": strength,
            "regularization_operator": "identity" if regularization_operator is None else "linear_operator",
            "regularized_normal_residual_formula": (
                "||A^T(Ax-b) + lambda L^T Lx||_2 / "
                "max(||A^T b||_2 + lambda ||L^T Lx||_2, eps)"
            ),
        },
    )


def solve_tv_fista_detailed(
    operator: LinearOperator,
    measurement: torch.Tensor,
    *,
    reg_strength: float = 1e-3,
    num_iterations: int = 50,
    step_size: float | None = None,
    x_init: torch.Tensor | None = None,
    regularizer: TVRegularizer | None = None,
    tolerance: float = 1e-5,
    power_iterations: int = 12,
    min_value: float | None = None,
    max_value: float | None = None,
    control: SolveControl | None = None,
    callback: Callback | None = None,
) -> SolveResult:
    pass

    return solve_fista_detailed(operator, measurement,
        regularizer=regularizer if regularizer is not None else TVRegularizer(),
        reg_strength=reg_strength, num_iterations=num_iterations, step_size=step_size,
        x_init=x_init, tolerance=tolerance, power_iterations=power_iterations,
        min_value=min_value, max_value=max_value, control=control, callback=callback,
        legacy=True)

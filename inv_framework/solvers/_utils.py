"""solvers _utils implementation."""
from __future__ import annotations
from inv_framework.core.operator import ForwardOperator, LinearOperator
from typing import Iterable, List, Optional, Sequence
import torch


def require_linear_operator(operator: ForwardOperator, solver_name: str) -> None:
    """Raise a clear error when an explicit adjoint is required."""
    if not isinstance(operator, LinearOperator):
        raise TypeError(
            f"{solver_name} requires a LinearOperator (needs an explicit "
            f"adjoint); got {type(operator).__name__}. Use "
            "DIPSolver/INRSolver/DPSSolver for non-linear forward models."
        )


def validate_measurement_shape(
    measurement: torch.Tensor,
    operator: ForwardOperator,
    solver_name: str,
) -> int:
    """Validate batched measurement shape and return the batch size."""
    if not isinstance(measurement, torch.Tensor):
        raise TypeError(
            f"{solver_name} measurement must be a torch.Tensor; "
            f"got {type(measurement).__name__}."
        )
    expected_tail = tuple(operator.range_shape)
    actual_tail = tuple(measurement.shape[1:])
    if actual_tail != expected_tail:
        raise ValueError(
            f"{solver_name} expected measurement shape "
            f"(B, {expected_tail}), got {tuple(measurement.shape)}."
        )
    return int(measurement.shape[0])


def prepare_initial_image(
    measurement: torch.Tensor,
    operator: ForwardOperator,
    x_init: Optional[torch.Tensor] = None,
    initial_value: float = 0.0,
) -> torch.Tensor:
    """Create or validate a batched reconstruction tensor."""
    expected = (int(measurement.shape[0]), *tuple(operator.domain_shape))
    if x_init is None:
        return torch.full(
            expected,
            float(initial_value),
            device=measurement.device,
            dtype=measurement.dtype,
        )
    if not isinstance(x_init, torch.Tensor):
        raise TypeError(f"x_init must be a torch.Tensor; got {type(x_init).__name__}.")
    if tuple(x_init.shape) != expected:
        raise ValueError(f"x_init must have shape {expected}; got {tuple(x_init.shape)}.")
    return x_init.to(device=measurement.device, dtype=measurement.dtype).clone()


def apply_box_constraints(
    x: torch.Tensor,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> torch.Tensor:
    """Apply optional value bounds."""
    if min_value is None and max_value is None:
        return x
    return x.clamp(min_value, max_value)

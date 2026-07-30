"""Statistical CT reconstruction solvers for nonnegative projection data."""

from typing import Iterable, Optional, Sequence

import torch

from ..operators.base import ForwardOperator, LinearOperator
from .base import InverseProblemSolver
from ._utils import (
    apply_box_constraints,
    make_angle_subsets,
    make_subset_operator,
    prepare_initial_image,
    require_linear_operator,
    select_measurement_subset,
    validate_measurement_shape,
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


def _store_subset_indices(subset_indices: Optional[Iterable[Sequence[int]]]):
    if subset_indices is None:
        return None
    return [tuple(int(v) for v in indices) for indices in subset_indices]


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

    x = _positive_initial(y, operator, x_init, initial_value, min_value)
    y_nonnegative = y.clamp_min(0.0)
    sensitivity = operator.adjoint(torch.ones_like(y_nonnegative)).clamp_min(float(eps))

    for _ in range(int(num_iterations)):
        prediction = operator.forward(x).clamp_min(float(eps))
        ratio = y_nonnegative / prediction
        correction = operator.adjoint(ratio) / sensitivity
        x = x * correction.clamp_min(0.0)
        x = apply_box_constraints(x, min_value=min_value, max_value=max_value)
    return x


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
    require_linear_operator(operator, "osem")
    validate_measurement_shape(y, operator, "osem")

    if block_size is None:
        block_size = max(int(operator.range_shape[-2]) // 10, 1)
    x = _positive_initial(y, operator, x_init, initial_value, min_value)
    y_nonnegative = y.clamp_min(0.0)
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

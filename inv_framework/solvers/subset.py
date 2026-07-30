"""Subset-based algebraic CT reconstruction solvers."""

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


def _safe_reciprocal(z: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.where(z.abs() < eps, torch.full_like(z, float("inf")), z).reciprocal()


def _store_subset_indices(subset_indices: Optional[Iterable[Sequence[int]]]):
    if subset_indices is None:
        return None
    return [tuple(int(v) for v in indices) for indices in subset_indices]


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

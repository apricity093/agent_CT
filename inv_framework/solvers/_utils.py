"""Internal helpers for add-on CT solvers."""

from typing import Iterable, List, Optional, Sequence

import torch

from ..operators.base import ForwardOperator, LinearOperator


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

    from ..operators.ct.radon_torch import ParallelBeamRadon2D

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

"""Two-dimensional total-variation regularization and its proximal map."""

from typing import Literal, Tuple

import torch

from .base import ProximalOperator, Regularizer
from ..operators.gradient import gradient, gradient_adjoint


class TVRegularizer(Regularizer, ProximalOperator):
    """Isotropic or anisotropic TV over the final two tensor dimensions.

    The proximal map solves the ROF denoising subproblem with fast gradient
    projection on the dual problem. Batch and channel dimensions are kept
    independent; the implementation therefore works for ``(B, C, H, W)`` CT
    images and tensors with additional leading dimensions.
    """

    has_full_proximal = True
    proximal_accuracy = 'finite_FGP_no_gap_certificate'

    def __init__(
        self,
        mode: Literal["isotropic", "anisotropic"] = "isotropic",
        num_iterations: int = 50,
        tolerance: float = 1e-5,
    ):
        if mode not in ("isotropic", "anisotropic"):
            raise ValueError("TV mode must be 'isotropic' or 'anisotropic'.")
        if int(num_iterations) <= 0:
            raise ValueError("TV num_iterations must be positive.")
        if float(tolerance) < 0.0:
            raise ValueError("TV tolerance must be nonnegative.")
        self.mode = mode
        self.num_iterations = int(num_iterations)
        self.tolerance = float(tolerance)
        # Diagnostics only: detailed solvers read this value after each prox
        # call so early stopping inside the proximal map is accounted for
        # without changing the numerical update.
        self.last_prox_iterations = 0
        self.gradient_calls = self.gradient_adjoint_calls = 0

    def split_transform(self, shape):
        from ..operators.gradient import GradientOperator
        return GradientOperator(shape), 8.

    def coefficient_value(self, z):
        return z.square().sum(1).sqrt() if self.mode == 'isotropic' else z.abs()

    def coefficient_subgradient(self, z):
        """Minimum-norm coefficient subgradient, including zero differences."""
        if self.mode == 'anisotropic':
            return z.sign()
        return z / z.square().sum(1, keepdim=True).sqrt().clamp_min(torch.finfo(z.dtype).tiny)

    def coefficient_proximal(self, z, t):
        if self.mode == 'isotropic':
            return z*(1-t/z.square().sum(1,keepdim=True).sqrt().clamp_min(torch.finfo(z.dtype).tiny)).clamp_min(0)
        return z.sign()*(z.abs()-t).clamp_min(0)

    _gradient = staticmethod(gradient)
    _gradient_adjoint = staticmethod(gradient_adjoint)

    def _apply_gradient(self, x):
        self.gradient_calls += 1
        return self._gradient(x)

    def _apply_gradient_adjoint(self, vertical, horizontal):
        self.gradient_adjoint_calls += 1
        return self._gradient_adjoint(vertical, horizontal)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        vertical, horizontal = self._apply_gradient(x)
        if self.mode == "isotropic":
            density = torch.sqrt(vertical.square() + horizontal.square())
        else:
            density = vertical.abs() + horizontal.abs()
        return density.reshape(density.shape[0], -1).sum(dim=1)

    def proximal(self, x: torch.Tensor, step_size: float) -> torch.Tensor:
        self._validate_input(x)
        weight = float(step_size)
        if weight < 0.0:
            raise ValueError("TV proximal step_size must be nonnegative.")
        self.last_prox_iterations = 0
        if weight == 0.0:
            return x.clone()

        dual_v = torch.zeros_like(x)
        dual_h = torch.zeros_like(x)
        momentum_v = dual_v.clone()
        momentum_h = dual_h.clone()
        acceleration = 1.0
        previous = x

        # ||grad||^2 <= 8 for 2D forward differences, so 1 / (8 * weight)
        # is a valid dual gradient step after scaling the dual by weight.
        dual_step = 1.0 / (8.0 * weight)

        for iteration in range(1, self.num_iterations + 1):
            self.last_prox_iterations = iteration
            primal = x - weight * self._apply_gradient_adjoint(momentum_v, momentum_h)
            grad_v, grad_h = self._apply_gradient(primal)
            next_v = momentum_v + dual_step * grad_v
            next_h = momentum_h + dual_step * grad_h
            next_v, next_h = self._project_dual(next_v, next_h)

            next_acceleration = 0.5 * (1.0 + (1.0 + 4.0 * acceleration * acceleration) ** 0.5)
            momentum_scale = (acceleration - 1.0) / next_acceleration
            momentum_v = next_v + momentum_scale * (next_v - dual_v)
            momentum_h = next_h + momentum_scale * (next_h - dual_h)
            dual_v, dual_h = next_v, next_h
            acceleration = next_acceleration

            current = x - weight * self._apply_gradient_adjoint(dual_v, dual_h)
            if self.tolerance > 0.0:
                delta = (current - previous).reshape(current.shape[0], -1).norm(dim=1)
                scale = previous.reshape(previous.shape[0], -1).norm(dim=1).clamp_min(1.0)
                if bool(torch.all(delta <= self.tolerance * scale)):
                    return current
            previous = current

        return x - weight * self._apply_gradient_adjoint(dual_v, dual_h)

    def _project_dual(
        self,
        vertical: torch.Tensor,
        horizontal: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "anisotropic":
            return vertical.clamp(-1.0, 1.0), horizontal.clamp(-1.0, 1.0)
        norm = torch.sqrt(vertical.square() + horizontal.square()).clamp_min(1.0)
        return vertical / norm, horizontal / norm

    @staticmethod
    def _validate_input(x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"TV input must be a torch.Tensor; got {type(x).__name__}.")
        if x.ndim < 3:
            raise ValueError(
                "TV input must include a batch dimension and two spatial dimensions; "
                f"got shape {tuple(x.shape)}."
            )

"""Unit-grid forward differences on the last two dimensions.

The final vertical row and horizontal column are zero. Their dual components
are outside the difference range and are ignored by the transpose, even when
nonzero. All leading dimensions are independent.
"""

from typing import Tuple

import torch

from .base import LinearOperator


class GradientOperator(LinearOperator):
    """D with stacked vertical/horizontal components at axis 1 (after batch)."""

    transform_kind = 'D'

    def __init__(self, domain_shape):
        self.domain_shape = tuple(domain_shape)
        if len(self.domain_shape) < 2 or any(d <= 0 for d in self.domain_shape):
            raise ValueError("GradientOperator requires positive spatial dimensions.")
        self.range_shape = (2, *self.domain_shape)
        self.forward_calls = self.adjoint_calls = 0
        self.runtime_seconds = 0.0

    def nullspace_basis(self,reference):
        leading=reference[0].numel()//(reference.shape[-2]*reference.shape[-1])
        basis=[]
        for index in range(leading):
            vector=torch.zeros_like(reference)
            vector.reshape(reference.shape[0],leading,*reference.shape[-2:])[:,index]=1
            basis.append(vector)
        return basis

    def forward(self, x):
        import time
        if x.is_cuda: torch.cuda.synchronize(x.device)
        started = time.perf_counter()
        self.forward_calls += 1
        result = torch.stack(gradient(x), dim=1)
        if x.is_cuda: torch.cuda.synchronize(x.device)
        self.runtime_seconds += time.perf_counter() - started
        return result

    def adjoint(self, y):
        import time
        if y.is_cuda: torch.cuda.synchronize(y.device)
        started = time.perf_counter()
        self.adjoint_calls += 1
        result = gradient_adjoint(y[:, 0], y[:, 1])
        if y.is_cuda: torch.cuda.synchronize(y.device)
        self.runtime_seconds += time.perf_counter() - started
        return result

    def stats(self):
        return {"auxiliary_D_forward_calls": self.forward_calls,
                "auxiliary_D_adjoint_calls": self.adjoint_calls,
                "auxiliary_D_seconds": self.runtime_seconds}

    def pseudo_inverse(self, y, **kwargs):
        raise NotImplementedError("GradientOperator has no configured pseudo-inverse.")


def gradient(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return vertical and horizontal differences with zero terminal entries."""
    vertical = torch.zeros_like(x)
    horizontal = torch.zeros_like(x)
    vertical[..., :-1, :] = x[..., 1:, :] - x[..., :-1, :]
    horizontal[..., :, :-1] = x[..., :, 1:] - x[..., :, :-1]
    return vertical, horizontal


def gradient_adjoint(vertical: torch.Tensor, horizontal: torch.Tensor) -> torch.Tensor:
    """Apply D* to an arbitrary pair of equally shaped dual tensors."""
    if vertical.shape != horizontal.shape:
        raise ValueError("Gradient dual components must have the same shape.")
    out = torch.zeros_like(vertical)
    out[..., :-1, :] -= vertical[..., :-1, :]
    out[..., 1:, :] += vertical[..., :-1, :]
    out[..., :, :-1] -= horizontal[..., :, :-1]
    out[..., :, 1:] += horizontal[..., :, :-1]
    return out

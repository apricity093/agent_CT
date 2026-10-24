"""Orthonormal 2D Haar, even-origin pairs and periodic dyadic grids.

Each active rectangle is packed as [[LL, LH], [HL, HH]]. L/H refer to
the vertical then horizontal filter; high-pass is (even - odd)/sqrt(2).
Only LL is recursively transformed. No padding or change of resolution.
"""

import math

import torch

from .base import LinearOperator


def validate_haar(shape, level):
    if isinstance(level, bool) or not isinstance(level, int) or level <= 0:
        raise ValueError("Haar level must be a positive integer")
    if len(shape) < 2 or any(d <= 0 or d % (2 ** level) for d in shape[-2:]):
        raise ValueError("Haar H/W must be positive and divisible by 2**level")


class HaarOperator(LinearOperator):
    """Square orthogonal transform preserving leading dimensions and dtype/device."""

    transform_kind = 'W'

    def __init__(self, domain_shape, level=2):
        validate_haar(domain_shape, level)
        self.domain_shape = tuple(domain_shape)
        self.range_shape = self.domain_shape
        self.level = level

    def forward(self, x):
        validate_haar(x.shape, self.level)
        out = x.clone()
        h, w = x.shape[-2:]
        s = math.sqrt(2.0)
        for _ in range(self.level):
            block = out[..., :h, :w].clone()
            low = (block[..., 0::2, :] + block[..., 1::2, :]) / s
            high = (block[..., 0::2, :] - block[..., 1::2, :]) / s
            out[..., :h//2, :w//2] = (low[..., 0::2] + low[..., 1::2]) / s
            out[..., :h//2, w//2:w] = (low[..., 0::2] - low[..., 1::2]) / s
            out[..., h//2:h, :w//2] = (high[..., 0::2] + high[..., 1::2]) / s
            out[..., h//2:h, w//2:w] = (high[..., 0::2] - high[..., 1::2]) / s
            h, w = h//2, w//2
        return out

    def adjoint(self, y):
        validate_haar(y.shape, self.level)
        out = y.clone()
        s = math.sqrt(2.0)
        for depth in range(self.level - 1, -1, -1):
            h, w = y.shape[-2] // (2**depth), y.shape[-1] // (2**depth)
            block = out[..., :h, :w].clone()
            ll, lh = block[..., :h//2, :w//2], block[..., :h//2, w//2:w]
            hl, hh = block[..., h//2:h, :w//2], block[..., h//2:h, w//2:w]
            low, high = torch.empty_like(block[..., :h//2, :]), torch.empty_like(block[..., :h//2, :])
            low[..., 0::2], low[..., 1::2] = (ll + lh)/s, (ll - lh)/s
            high[..., 0::2], high[..., 1::2] = (hl + hh)/s, (hl - hh)/s
            out[..., :h:2, :w], out[..., 1:h:2, :w] = (low + high)/s, (low - high)/s
        return out

    def pseudo_inverse(self, y, **kwargs):
        return self.adjoint(y)

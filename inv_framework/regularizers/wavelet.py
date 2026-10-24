"""Haar detail L1; coarsest LL has zero weight, all details weight one."""

import math
import torch

from ..operators.wavelet import HaarOperator, validate_haar
from .base import Regularizer, ProximalOperator


class HaarL1Regularizer(Regularizer, ProximalOperator):
    has_full_proximal = True
    def validate_shape(self, shape):
        validate_haar(shape, self.level)

    def __init__(self, level=2):
        if isinstance(level, bool) or not isinstance(level, int) or level <= 0:
            raise ValueError("Haar level must be a positive integer")
        self.level = level
        self.analysis_calls = self.synthesis_calls = 0
        self.transform_seconds = 0.0

    def split_transform(self, shape):
        return HaarOperator(shape, self.level), 1.

    def coefficient_mask(self, z):
        m = torch.ones_like(z)
        h,w = z.shape[-2]//2**self.level,z.shape[-1]//2**self.level
        m[..., :h, :w] = 0
        return m

    def coefficient_value(self, z):
        return z.abs()*self.coefficient_mask(z)

    def coefficient_proximal(self, z, t):
        return z.sign()*(z.abs()-t*self.coefficient_mask(z)).clamp_min(0)

    def _transform(self, x, synthesis=False):
        import time

        if x.is_cuda: torch.cuda.synchronize(x.device)
        started = time.perf_counter()
        if synthesis:
            self.synthesis_calls += 1
        else:
            self.analysis_calls += 1
        operator = HaarOperator(x.shape[1:], self.level)
        result = operator.adjoint(x) if synthesis else operator.forward(x)
        if x.is_cuda: torch.cuda.synchronize(x.device)
        self.transform_seconds += time.perf_counter() - started
        return result

    def value(self, x):
        coefficients = self._transform(x)
        return self.coefficient_value(coefficients).reshape(x.shape[0], -1).sum(dim=1)

    def proximal(self, x, step_size):
        if not math.isfinite(float(step_size)) or step_size < 0:
            raise ValueError("Haar proximal step_size must be finite and nonnegative")
        c = self._transform(x)
        result = self.coefficient_proximal(c, step_size)
        return self._transform(result, synthesis=True)

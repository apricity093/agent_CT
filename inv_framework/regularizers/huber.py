"""Componentwise Huber penalty of unit-grid image differences."""
import math
import torch
from .base import Regularizer
from ..operators.gradient import gradient, gradient_adjoint

class GradientHuberRegularizer(Regularizer):
    def __init__(self, delta=0.01):
        if not math.isfinite(float(delta)) or delta <= 0:
            raise ValueError("delta must be finite and positive")
        self.delta = float(delta)

    def split_transform(self, shape):
        from ..operators.gradient import GradientOperator
        return GradientOperator(shape), 8.

    def coefficient_value(self, z):
        a = z.abs()
        return torch.where(a <= self.delta, z.square()/(2*self.delta), a-self.delta/2)

    def coefficient_proximal(self, v, t):
        if not math.isfinite(float(t)) or t < 0:
            raise ValueError("proximal step must be finite and nonnegative")
        return torch.where(v.abs() <= self.delta+t, self.delta*v/(self.delta+t), v-t*v.sign())

    def value(self, x):
        z = torch.stack(gradient(x), dim=1)
        return self.coefficient_value(z).flatten(1).sum(1)

    def gradient(self, x):
        v, h = gradient(x)
        return gradient_adjoint((v/self.delta).clamp(-1,1), (h/self.delta).clamp(-1,1))

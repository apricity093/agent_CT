"""Weighted augmented least squares for identity and H1 penalties."""
import math
import torch
from .base import LinearOperator


class AugmentedLeastSquaresOperator(LinearOperator):
    def __init__(self, operator, fidelity, strength, regularizer):
        from .gradient import GradientOperator
        self.operator, self.fidelity = operator, fidelity
        self.domain_shape = operator.domain_shape
        self.data_size = math.prod(operator.range_shape)
        self.transform = GradientOperator(self.domain_shape) if regularizer == 'h1' else None
        self.strength = strength
        self.extra_size = (math.prod(self.transform.range_shape) if self.transform else math.prod(self.domain_shape)) if strength else 0
        self.range_shape = (self.data_size+self.extra_size,)
        self.gradient_calls = self.gradient_adjoint_calls = 0

    def forward(self, x):
        data = self.operator.forward(x)*self.fidelity.weights.sqrt()
        if not self.strength: return data.flatten(1)
        self.gradient_calls += int(self.transform is not None)
        extra = self.transform.forward(x) if self.transform else x
        return torch.cat((data.flatten(1), math.sqrt(self.strength)*extra.flatten(1)), 1)

    def adjoint(self, y):
        data = y[:, :self.data_size].reshape(y.shape[0], *self.operator.range_shape)
        x = self.operator.adjoint(data*self.fidelity.weights.sqrt())
        if self.strength:
            self.gradient_adjoint_calls += int(self.transform is not None)
            shape = self.transform.range_shape if self.transform else self.domain_shape
            extra = y[:, self.data_size:].reshape(y.shape[0], *shape)
            x = x+math.sqrt(self.strength)*(self.transform.adjoint(extra) if self.transform else extra)
        return x

    def rhs(self, measurement):
        data = (measurement*self.fidelity.weights.sqrt()).flatten(1)
        return torch.cat((data, data.new_zeros((data.shape[0], self.extra_size))), 1)

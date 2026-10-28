"""The zero penalty, with no image transform work."""
import torch
from .base import Regularizer, ProximalOperator


class ZeroRegularizer(Regularizer, ProximalOperator):
    has_full_proximal = supports_box_proximal = True

    def value(self, x):
        return x.new_zeros(x.shape[0])

    def gradient(self, x):
        return torch.zeros_like(x)

    def proximal(self, x, step_size):
        return x.clone()

    def split_transform(self, shape):
        return None, 0.

    def coefficient_value(self, z):
        return z.new_zeros(z.shape)

    def coefficient_proximal(self, z, t):
        return z.clone()

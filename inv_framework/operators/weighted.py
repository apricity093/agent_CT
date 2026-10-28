"""Quadratic whitening with paired forward and adjoint operations."""
import torch
from .base import LinearOperator

class WeightedLinearOperator(LinearOperator):
    def __init__(self,operator,weights):
        if tuple(weights.shape[1:])!=tuple(operator.range_shape):
            raise ValueError('weights must match the batched operator range')
        if not weights.is_floating_point() or not bool(torch.isfinite(weights).all()) or bool((weights<0).any()):
            raise ValueError('weights must be finite nonnegative floating values')
        self.operator=operator;self.sqrt_weights=weights.sqrt()
        self.domain_shape=operator.domain_shape;self.range_shape=operator.range_shape
        self.weight_multiplications=0
    @property
    def nonnegative_matrix(self):return self.operator.nonnegative_matrix
    def whiten(self,value):
        if value.shape!=self.sqrt_weights.shape or value.device!=self.sqrt_weights.device or value.dtype!=self.sqrt_weights.dtype:
            raise ValueError('whitening requires matching shape, dtype and device')
        self.weight_multiplications+=1
        accounting=getattr(self,"accounting_fidelity",None)
        if accounting is not None: accounting.weight_multiplications+=1
        return self.sqrt_weights*value
    def forward(self,x):return self.whiten(self.operator.forward(x))
    def adjoint(self,y):return self.operator.adjoint(self.whiten(y))
    def squared_norm_upper_bound(self,x):
        return float(self.sqrt_weights.square().max())*self.operator.squared_norm_upper_bound(x)
    def subset(self, indices):
        from ..solvers._utils import make_subset_operator
        return WeightedLinearOperator(make_subset_operator(self.operator, indices),
            self.sqrt_weights.square().index_select(-2, indices.to(self.sqrt_weights.device)))
    def stats(self):
        function=getattr(self.operator,'stats',None)
        return function() if callable(function) else {}
    def record_prox(self, **kwargs):
        function=getattr(self.operator,'record_prox',None)
        if callable(function): return function(**kwargs)

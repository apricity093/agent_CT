"""Public regularization interfaces and implementations."""

from .base import ProximalOperator, Regularizer
from .tikhonov import TikhonovRegularizer
from .tv import TVRegularizer

__all__ = [
    "Regularizer",
    "ProximalOperator",
    "TikhonovRegularizer",
    "TVRegularizer",
]

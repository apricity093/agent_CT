"""Public regularization interfaces and implementations."""

from .base import ProximalOperator, Regularizer
from .tikhonov import TikhonovRegularizer
from .tv import TVRegularizer

__all__ = [
    "Regularizer",
    "ProximalOperator",
    "TikhonovRegularizer",
    "TVRegularizer",
    "HaarL1Regularizer",
]
from .wavelet import HaarL1Regularizer

from .huber import GradientHuberRegularizer
__all__ += ['GradientHuberRegularizer']
from .zero import ZeroRegularizer
__all__ += ['ZeroRegularizer']

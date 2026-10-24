"""Regularization interfaces for variational inverse problems."""

from abc import ABC, abstractmethod

import torch


class Regularizer(ABC):
    """Penalty evaluated independently for each batch item."""

    has_full_proximal = False
    supports_box_proximal = False
    proximal_accuracy = 'closed_form'

    def validate_shape(self, shape):
        pass

    @abstractmethod
    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Return one regularization value per batch item."""


class ProximalOperator(ABC):
    """Penalty with an explicitly available proximal operator."""

    @abstractmethod
    def proximal(self, x: torch.Tensor, step_size: float) -> torch.Tensor:
        """Return ``prox_{step_size R}(x)`` without modifying ``x``."""

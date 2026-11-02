"""solvers deep diffusion implementation."""
from __future__ import annotations

from inv_framework.solvers.deep.diffusion.scheduler import NoiseSchedule, VPSchedule
from inv_framework.solvers.deep.diffusion.conditioning import ConditioningMethod, PosteriorSampling
from inv_framework.solvers.deep.diffusion.dps import DPSSolver


__all__ = [
    "NoiseSchedule",
    "VPSchedule",
    "ConditioningMethod",
    "PosteriorSampling",
    "DPSSolver",
]


try:
    from inv_framework.solvers.deep.diffusion.diffusers_compat import DiffusersScheduleAdapter, DiffusersUNetWrapper
    __all__.extend(["DiffusersScheduleAdapter", "DiffusersUNetWrapper"])
except ImportError:
    pass

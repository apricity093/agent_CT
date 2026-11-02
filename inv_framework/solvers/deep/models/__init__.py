"""solvers deep models implementation."""
from __future__ import annotations

from inv_framework.solvers.deep.models.siren import SIREN, SineLayer
from inv_framework.solvers.deep.models.skip_unet import SkipUNet
from inv_framework.solvers.deep.models.tiny_unet import TinyUNet


__all__ = ["SIREN", "SineLayer", "SkipUNet", "TinyUNet"]

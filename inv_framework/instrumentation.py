"""Runtime instrumentation for CT linear operators.

The wrapper is deliberately an adapter at the solver seam.  It preserves the
``LinearOperator`` interface, shares counters with subset operators, and
raises a bounded error when a caller-supplied operator budget is exceeded.
"""

from __future__ import annotations

import ctypes
import os
import time
import tracemalloc
from dataclasses import dataclass
from typing import Any

import torch

from inv_framework.operators.base import LinearOperator


class OperatorBudgetExceeded(RuntimeError):
    """Raised before an operator call would exceed its configured budget."""


@dataclass
class OperatorCounters:
    forward_calls: int = 0
    adjoint_calls: int = 0
    forward_seconds: float = 0.0
    adjoint_seconds: float = 0.0
    peak_memory_mb: float | None = None
    peak_gpu_memory_mb: float | None = None

    @property
    def total_calls(self) -> int:
        return self.forward_calls + self.adjoint_calls

    @property
    def runtime_seconds(self) -> float:
        return self.forward_seconds + self.adjoint_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "forward_calls": self.forward_calls,
            "adjoint_calls": self.adjoint_calls,
            "total_operator_calls": self.total_calls,
            "forward_seconds": self.forward_seconds,
            "adjoint_seconds": self.adjoint_seconds,
            "operator_runtime_seconds": self.runtime_seconds,
            "peak_memory_mb": self.peak_memory_mb,
            "peak_gpu_memory_mb": self.peak_gpu_memory_mb,
        }


def _memory_mb() -> float | None:
    """Best-effort process/Python peak memory measurement without dependencies."""

    try:
        import resource

        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB, macOS reports bytes.  Windows has no resource.
        return value / (1024.0 if value < 1024.0**4 else 1024.0**2)
    except (ImportError, OSError, AttributeError):
        if tracemalloc.is_tracing():
            _current, peak = tracemalloc.get_traced_memory()
            python_peak = float(peak) / (1024.0**2)
        else:
            python_peak = None
        working_set = _windows_working_set_mb()
        candidates = [value for value in (python_peak, working_set) if value is not None]
        return max(candidates) if candidates else None


def _windows_working_set_mb() -> float | None:
    """Read process peak working set on Windows without adding psutil."""

    if os.name != "nt":
        return None
    try:
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        process = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            process,
            ctypes.byref(counters),
            counters.cb,
        )
        return float(counters.PeakWorkingSetSize) / (1024.0**2) if ok else None
    except (AttributeError, OSError, TypeError):
        return None


def _gpu_memory_mb() -> float | None:
    """Return PyTorch's device allocator high-water mark when available."""

    try:
        if not torch.cuda.is_available():
            return None
        return float(torch.cuda.max_memory_allocated()) / (1024.0**2)
    except (RuntimeError, AttributeError):
        return None


class CountingLinearOperator(LinearOperator):
    """Count and time ``forward``/``adjoint`` calls on a linear operator."""

    def __init__(
        self,
        operator: LinearOperator,
        *,
        max_forward_calls: int | None = None,
        max_adjoint_calls: int | None = None,
        counters: OperatorCounters | None = None,
    ) -> None:
        if not isinstance(operator, LinearOperator):
            raise TypeError(
                "CountingLinearOperator requires a LinearOperator; "
                f"got {type(operator).__name__}"
            )
        for name, limit in (("max_forward_calls", max_forward_calls), ("max_adjoint_calls", max_adjoint_calls)):
            if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 0):
                raise ValueError(f"{name} must be a nonnegative integer")
        self.operator = operator
        self.domain_shape = tuple(operator.domain_shape)
        self.range_shape = tuple(operator.range_shape)
        self.max_forward_calls = max_forward_calls
        self.max_adjoint_calls = max_adjoint_calls
        self.counters = counters or OperatorCounters()

    def _before(self, kind: str) -> None:
        current = self.counters.forward_calls if kind == "forward" else self.counters.adjoint_calls
        limit = self.max_forward_calls if kind == "forward" else self.max_adjoint_calls
        if limit is not None and current >= int(limit):
            raise OperatorBudgetExceeded(
                f"{kind} operator-call budget exhausted: limit={int(limit)}"
            )
        if kind == "forward":
            self.counters.forward_calls += 1
        else:
            self.counters.adjoint_calls += 1

    def _after(self) -> None:
        memory = _memory_mb()
        if memory is not None:
            if self.counters.peak_memory_mb is None:
                self.counters.peak_memory_mb = memory
            else:
                self.counters.peak_memory_mb = max(self.counters.peak_memory_mb, memory)
        gpu_memory = _gpu_memory_mb()
        if gpu_memory is not None:
            if self.counters.peak_gpu_memory_mb is None:
                self.counters.peak_gpu_memory_mb = gpu_memory
            else:
                self.counters.peak_gpu_memory_mb = max(self.counters.peak_gpu_memory_mb, gpu_memory)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._before("forward")
        started = time.perf_counter()
        try:
            return self.operator.forward(x)
        finally:
            self.counters.forward_seconds += time.perf_counter() - started
            self._after()

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        self._before("adjoint")
        started = time.perf_counter()
        try:
            return self.operator.adjoint(y)
        finally:
            self.counters.adjoint_seconds += time.perf_counter() - started
            self._after()

    def subset(self, indices: torch.Tensor) -> "CountingLinearOperator":
        """Preserve shared counters when a subset solver creates a child."""

        subset = getattr(self.operator, "subset", None)
        if callable(subset):
            child = subset(indices)
        else:
            # The pure-torch Radon operator intentionally has no public subset
            # method.  Recreate it from its public geometry so the internal
            # subset-solver seam remains instrumentable.
            from inv_framework.operators.ct.radon_torch import ParallelBeamRadon2D

            if not isinstance(self.operator, ParallelBeamRadon2D):
                raise NotImplementedError(
                    f"{type(self.operator).__name__} does not provide subset(indices)"
                )
            idx = indices.to(device=self.operator.angles.device, dtype=torch.long)
            child = ParallelBeamRadon2D(
                image_size=int(self.operator.image_size),
                angles=self.operator.angles.index_select(0, idx),
                device=self.operator.angles.device,
                in_channels=int(self.operator.domain_shape[0]),
            )
        return CountingLinearOperator(
            child,
            max_forward_calls=self.max_forward_calls,
            max_adjoint_calls=self.max_adjoint_calls,
            counters=self.counters,
        )

    def fdk(self, y: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        backend = getattr(self.operator, "fdk", None)
        if not callable(backend):
            raise AttributeError(f"{type(self.operator).__name__} has no FDK backend")
        return backend(y, **kwargs)

    def stats(self) -> dict[str, Any]:
        return self.counters.to_dict()

    def __getattr__(self, name: str) -> Any:
        # Geometry-specific solver adapters may inspect public operator
        # attributes such as ``angles`` or ``num_angles``.
        if name in {"operator", "counters"}:
            raise AttributeError(name)
        return getattr(self.operator, name)


def start_memory_tracing() -> None:
    """Enable the optional Python allocation counter used on Windows."""

    if not tracemalloc.is_tracing():
        tracemalloc.start()
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except (RuntimeError, AttributeError):
        pass

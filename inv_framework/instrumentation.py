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
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch

from inv_framework.operators.base import LinearOperator


class OperatorBudgetExceeded(RuntimeError):
    """Raised before an operator call would exceed its configured budget."""

    def __init__(self, message: str, *, kind: str | None = None, limit: int | None = None, attempted: int | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.limit = limit
        self.attempted = attempted


RESOURCE_SCHEMA_VERSION = "ct.resource_usage.v2"
RESOURCE_PHASES = (
    "preparation",
    "parameter_estimation",
    "tuning_fit",
    "tuning_heldout",
    "solver",
    "endpoint_confirmation",
    "final_evaluation",
)


@dataclass
class OperatorCounters:
    # ``forward_calls`` and ``adjoint_calls`` retain their historical meaning:
    # an invocation was admitted by the budget and entered the wrapped
    # operator.  Attempted calls, including a rejected call at the budget
    # boundary, are tracked separately so old consumers and exact call
    # formulas remain stable while failures are still auditable.
    forward_calls: int = 0
    adjoint_calls: int = 0
    forward_attempts: int = 0
    adjoint_attempts: int = 0
    forward_executed_calls: int = 0
    adjoint_executed_calls: int = 0
    forward_failed_calls: int = 0
    adjoint_failed_calls: int = 0
    budget_rejected_forward_calls: int = 0
    budget_rejected_adjoint_calls: int = 0
    forward_seconds: float = 0.0
    adjoint_seconds: float = 0.0
    forward_gpu_seconds: float | None = None
    adjoint_gpu_seconds: float | None = None
    backend_reconstruction_calls: int = 0
    backend_executed_calls: int = 0
    backend_failed_calls: int = 0
    backend_exception_count: int = 0
    backend_seconds: float = 0.0
    backend_gpu_seconds: float | None = None
    prox_calls: int = 0
    prox_failed_calls: int = 0
    prox_iterations: int = 0
    prox_seconds: float = 0.0
    prox_gpu_seconds: float | None = None
    peak_memory_mb: float | None = None
    peak_gpu_memory_mb: float | None = None
    last_backend_exception: str | None = None
    _active_phase: str | None = field(default=None, init=False, repr=False)
    _phase_peaks: dict[str, dict[str, float | None]] = field(default_factory=dict, init=False, repr=False)

    @property
    def total_calls(self) -> int:
        return self.forward_calls + self.adjoint_calls

    @property
    def total_attempts(self) -> int:
        return self.forward_attempts + self.adjoint_attempts

    @property
    def executed_calls(self) -> int:
        return self.forward_executed_calls + self.adjoint_executed_calls

    @property
    def runtime_seconds(self) -> float:
        return self.forward_seconds + self.adjoint_seconds

    @property
    def gpu_runtime_seconds(self) -> float | None:
        if self.forward_gpu_seconds is None and self.adjoint_gpu_seconds is None:
            return None
        return float(self.forward_gpu_seconds or 0.0) + float(self.adjoint_gpu_seconds or 0.0)

    @property
    def backend_gpu_elapsed_seconds(self) -> float | None:
        return self.backend_gpu_seconds

    @property
    def prox_gpu_elapsed_seconds(self) -> float | None:
        return self.prox_gpu_seconds

    def begin_phase(self, name: str) -> None:
        if not name:
            raise ValueError("resource phase name must be non-empty")
        if self._active_phase is not None:
            raise RuntimeError(
                f"resource phase {self._active_phase!r} is still active; "
                f"cannot start {name!r}"
            )
        self._active_phase = str(name)
        self._phase_peaks.setdefault(
            str(name),
            {
                "peak_memory_mb": _memory_mb(),
                "peak_gpu_memory_mb": _gpu_memory_mb(),
            },
        )
        values = self._phase_peaks[str(name)]
        self._update_peak(values.get("peak_memory_mb"), values.get("peak_gpu_memory_mb"))

    def end_phase(self) -> None:
        self._active_phase = None

    def _update_peak(self, memory: float | None, gpu_memory: float | None) -> None:
        if memory is not None:
            self.peak_memory_mb = (
                memory if self.peak_memory_mb is None else max(self.peak_memory_mb, memory)
            )
        if gpu_memory is not None:
            self.peak_gpu_memory_mb = (
                gpu_memory
                if self.peak_gpu_memory_mb is None
                else max(self.peak_gpu_memory_mb, gpu_memory)
            )
        if self._active_phase is not None:
            values = self._phase_peaks.setdefault(
                self._active_phase,
                {"peak_memory_mb": None, "peak_gpu_memory_mb": None},
            )
            if memory is not None:
                current = values.get("peak_memory_mb")
                values["peak_memory_mb"] = memory if current is None else max(float(current), memory)
            if gpu_memory is not None:
                current = values.get("peak_gpu_memory_mb")
                values["peak_gpu_memory_mb"] = gpu_memory if current is None else max(float(current), gpu_memory)

    def phase_peaks(self, name: str) -> dict[str, float | None]:
        return dict(self._phase_peaks.get(name, {"peak_memory_mb": None, "peak_gpu_memory_mb": None}))

    def record_prox(
        self,
        *,
        iterations: int = 0,
        elapsed_seconds: float = 0.0,
        gpu_elapsed_seconds: float | None = None,
        failed: bool = False,
    ) -> None:
        """Record one proximal-map attempt without coupling it to an operator call."""

        self.prox_calls += 1
        self.prox_iterations += max(0, int(iterations))
        self.prox_seconds += max(0.0, float(elapsed_seconds))
        if failed:
            self.prox_failed_calls += 1
        if gpu_elapsed_seconds is not None:
            self.prox_gpu_seconds = float(self.prox_gpu_seconds or 0.0) + max(
                0.0, float(gpu_elapsed_seconds)
            )

    def record_backend(
        self,
        *,
        elapsed_seconds: float,
        gpu_elapsed_seconds: float | None = None,
        failed: bool = False,
        exception: BaseException | None = None,
    ) -> None:
        """Record native reconstruction backend work such as FDK."""

        self.backend_reconstruction_calls += 1
        self.backend_seconds += max(0.0, float(elapsed_seconds))
        if failed:
            self.backend_failed_calls += 1
            self.backend_exception_count += 1
            if exception is not None:
                self.last_backend_exception = f"{type(exception).__name__}: {exception}"
        else:
            self.backend_executed_calls += 1
        if gpu_elapsed_seconds is not None:
            self.backend_gpu_seconds = float(self.backend_gpu_seconds or 0.0) + max(
                0.0, float(gpu_elapsed_seconds)
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RESOURCE_SCHEMA_VERSION,
            "forward_calls": self.forward_calls,
            "adjoint_calls": self.adjoint_calls,
            "forward_attempts": self.forward_attempts,
            "adjoint_attempts": self.adjoint_attempts,
            "attempted_forward_calls": self.forward_attempts,
            "attempted_adjoint_calls": self.adjoint_attempts,
            "forward_executed_calls": self.forward_executed_calls,
            "adjoint_executed_calls": self.adjoint_executed_calls,
            "forward_failed_calls": self.forward_failed_calls,
            "adjoint_failed_calls": self.adjoint_failed_calls,
            "budget_rejected_forward_calls": self.budget_rejected_forward_calls,
            "budget_rejected_adjoint_calls": self.budget_rejected_adjoint_calls,
            "total_operator_calls": self.total_calls,
            "total_operator_attempts": self.total_attempts,
            "attempted_total_operator_calls": self.total_attempts,
            "executed_operator_calls": self.executed_calls,
            "forward_seconds": self.forward_seconds,
            "adjoint_seconds": self.adjoint_seconds,
            "operator_runtime_seconds": self.runtime_seconds,
            "forward_gpu_seconds": self.forward_gpu_seconds,
            "adjoint_gpu_seconds": self.adjoint_gpu_seconds,
            "gpu_runtime_seconds": self.gpu_runtime_seconds,
            "backend_reconstruction_calls": self.backend_reconstruction_calls,
            "backend_executed_calls": self.backend_executed_calls,
            "backend_failed_calls": self.backend_failed_calls,
            "backend_exception_count": self.backend_exception_count,
            "backend_seconds": self.backend_seconds,
            "backend_runtime_seconds": self.backend_seconds,
            "backend_gpu_seconds": self.backend_gpu_seconds,
            "prox_calls": self.prox_calls,
            "prox_failed_calls": self.prox_failed_calls,
            "prox_iterations": self.prox_iterations,
            "prox_seconds": self.prox_seconds,
            "prox_runtime_seconds": self.prox_seconds,
            "prox_gpu_seconds": self.prox_gpu_seconds,
            "peak_memory_mb": self.peak_memory_mb,
            "peak_gpu_memory_mb": self.peak_gpu_memory_mb,
            "backend_exception": self.last_backend_exception,
        }


def _gpu_event_start(value: Any) -> Any | None:
    """Create an optional CUDA timer only for CUDA tensors."""

    try:
        if not torch.cuda.is_available() or not bool(getattr(value, "is_cuda", False)):
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event
    except (RuntimeError, AttributeError, TypeError):
        return None


def _gpu_event_elapsed(start: Any | None, value: Any) -> float | None:
    if start is None:
        return None
    try:
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        end.synchronize()
        return max(0.0, float(start.elapsed_time(end)) / 1000.0)
    except (RuntimeError, AttributeError, TypeError):
        return None


def _counter_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Subtract cumulative counters while leaving nullable GPU fields nullable."""

    result: dict[str, Any] = {}
    integer_fields = (
        "forward_calls", "adjoint_calls", "forward_attempts", "adjoint_attempts",
        "attempted_forward_calls", "attempted_adjoint_calls", "forward_executed_calls",
        "adjoint_executed_calls", "forward_failed_calls", "adjoint_failed_calls",
        "budget_rejected_forward_calls", "budget_rejected_adjoint_calls",
        "total_operator_calls", "total_operator_attempts", "attempted_total_operator_calls",
        "executed_operator_calls", "backend_reconstruction_calls", "backend_executed_calls",
        "backend_failed_calls", "backend_exception_count", "prox_calls", "prox_failed_calls",
        "prox_iterations",
    )
    float_fields = (
        "forward_seconds", "adjoint_seconds", "operator_runtime_seconds", "backend_seconds",
        "backend_runtime_seconds", "prox_seconds", "prox_runtime_seconds",
    )
    for name in integer_fields:
        if before.get(name) is not None and after.get(name) is not None:
            result[name] = max(0, int(after[name]) - int(before[name]))
    for name in float_fields:
        if before.get(name) is not None and after.get(name) is not None:
            result[name] = max(0.0, float(after[name]) - float(before[name]))
    for name in ("forward_gpu_seconds", "adjoint_gpu_seconds", "gpu_runtime_seconds", "backend_gpu_seconds", "prox_gpu_seconds"):
        before_value = before.get(name)
        after_value = after.get(name)
        if after_value is not None:
            result[name] = max(0.0, float(after_value) - float(before_value or 0.0))
        else:
            result[name] = None
    return result


class ResourceAccounting:
    """Phase ledger for one CT execution.

    The ledger consumes cumulative ``OperatorCounters`` snapshots.  Every
    phase is finalized exactly once, so the numerical counter/time fields in
    the phase records add back to the aggregate record.  Wall time is kept
    separate from operator time; GPU elapsed fields stay ``None`` when CUDA
    event timing was unavailable.
    """

    def __init__(
        self,
        counters: OperatorCounters | None = None,
        *,
        max_forward_calls: int | None = None,
        max_adjoint_calls: int | None = None,
    ) -> None:
        self.counters = counters or OperatorCounters()
        self.max_forward_calls = max_forward_calls
        self.max_adjoint_calls = max_adjoint_calls
        self._phases: dict[str, dict[str, Any]] = {}
        self._active: tuple[str, float, dict[str, Any]] | None = None
        self._created = time.perf_counter()

    @property
    def phases(self) -> dict[str, dict[str, Any]]:
        return self._phases

    def start_phase(self, name: str) -> None:
        if self._active is not None:
            raise RuntimeError(f"resource phase {self._active[0]!r} is still active")
        phase = str(name)
        self.counters.begin_phase(phase)
        self._active = (phase, time.perf_counter(), self.counters.to_dict())

    def finish_phase(self, name: str | None = None) -> dict[str, Any]:
        if self._active is None:
            return self._phases.setdefault(str(name or "unknown"), self._empty_phase())
        phase, started, before = self._active
        if name is not None and str(name) != phase:
            raise RuntimeError(f"active resource phase is {phase!r}, got {name!r}")
        self.counters._update_peak(_memory_mb(), _gpu_memory_mb())
        after = self.counters.to_dict()
        wall_seconds = max(0.0, time.perf_counter() - started)
        record = _counter_delta(before, after)
        record.update({
            "phase": phase,
            "wall_seconds": wall_seconds,
            "wall_time_seconds": wall_seconds,
            "peak_memory_mb": self.counters.phase_peaks(phase).get("peak_memory_mb"),
            "peak_gpu_memory_mb": self.counters.phase_peaks(phase).get("peak_gpu_memory_mb"),
        })
        # ``runtime_seconds`` is the phase wall time for consumers that use
        # the common runtime spelling; operator time remains explicit above.
        record["runtime_seconds"] = record["wall_seconds"]
        previous = self._phases.get(phase)
        if previous is not None:
            for key, value in list(record.items()):
                if key in {"phase", "peak_memory_mb", "peak_gpu_memory_mb"}:
                    continue
                if isinstance(value, (int, float)) and isinstance(previous.get(key), (int, float)):
                    record[key] = float(previous[key]) + float(value)
            for key in ("peak_memory_mb", "peak_gpu_memory_mb"):
                left = previous.get(key)
                right = record.get(key)
                if left is None:
                    record[key] = right
                elif right is None:
                    record[key] = left
                else:
                    record[key] = max(float(left), float(right))
        self._phases[phase] = record
        self.counters.end_phase()
        self._active = None
        return record

    def finish_open_phase(self) -> None:
        if self._active is not None:
            self.finish_phase()

    def add_wall_time(self, name: str, seconds: float) -> None:
        """Attribute non-operator work to an existing phase."""

        amount = max(0.0, float(seconds))
        if amount == 0.0:
            return
        phase = str(name)
        record = self._phases.setdefault(
            phase,
            {**self._empty_phase(), "phase": phase},
        )
        record["wall_seconds"] = float(record.get("wall_seconds", 0.0) or 0.0) + amount
        record["wall_time_seconds"] = record["wall_seconds"]
        record["runtime_seconds"] = record["wall_seconds"]

    @contextmanager
    def phase(self, name: str):
        self.start_phase(name)
        try:
            yield self
        finally:
            self.finish_phase(name)

    @staticmethod
    def _empty_phase() -> dict[str, Any]:
        counters = OperatorCounters()
        record = _counter_delta(counters.to_dict(), counters.to_dict())
        record.update({
            "phase": None,
            "wall_seconds": 0.0,
            "wall_time_seconds": 0.0,
            "runtime_seconds": 0.0,
            "peak_memory_mb": None,
            "peak_gpu_memory_mb": None,
        })
        return record

    def _ensure_phases(self) -> None:
        for name in RESOURCE_PHASES:
            self._phases.setdefault(name, {**self._empty_phase(), "phase": name})

    def to_dict(self) -> dict[str, Any]:
        self.finish_open_phase()
        self._ensure_phases()
        aggregate = self.counters.to_dict()
        for record in self._phases.values():
            if record.get("peak_memory_mb") is None:
                record["peak_memory_mb"] = aggregate.get("peak_memory_mb")
            if record.get("peak_gpu_memory_mb") is None and aggregate.get("peak_gpu_memory_mb") is not None:
                record["peak_gpu_memory_mb"] = aggregate.get("peak_gpu_memory_mb")
        phase_totals: dict[str, Any] = {}
        integer_fields = (
            "forward_calls", "adjoint_calls", "forward_attempts", "adjoint_attempts",
            "total_operator_calls", "total_operator_attempts", "executed_operator_calls",
            "forward_executed_calls", "adjoint_executed_calls", "forward_failed_calls",
            "adjoint_failed_calls", "budget_rejected_forward_calls", "budget_rejected_adjoint_calls",
            "backend_reconstruction_calls", "backend_executed_calls", "backend_failed_calls",
            "backend_exception_count", "prox_calls", "prox_failed_calls", "prox_iterations",
        )
        float_fields = (
            "forward_seconds", "adjoint_seconds", "operator_runtime_seconds", "backend_seconds",
            "backend_runtime_seconds", "prox_seconds", "prox_runtime_seconds",
            "wall_seconds", "wall_time_seconds", "runtime_seconds",
        )
        for name in integer_fields:
            phase_totals[name] = sum(int(record.get(name, 0) or 0) for record in self._phases.values())
        for name in float_fields:
            phase_totals[name] = sum(float(record.get(name, 0.0) or 0.0) for record in self._phases.values())
        for name in ("forward_gpu_seconds", "adjoint_gpu_seconds", "gpu_runtime_seconds", "backend_gpu_seconds", "prox_gpu_seconds"):
            values = [record.get(name) for record in self._phases.values() if record.get(name) is not None]
            phase_totals[name] = sum(float(value) for value in values) if values else None
        wall_seconds = sum(float(record.get("wall_seconds", 0.0)) for record in self._phases.values())
        aggregate.update({
            "schema_version": RESOURCE_SCHEMA_VERSION,
            "wall_seconds": wall_seconds,
            "wall_time_seconds": wall_seconds,
            "runtime_seconds": wall_seconds,
            "phases": {name: dict(record) for name, record in self._phases.items()},
            "tuning_usage": {
                name: dict(self._phases[name])
                for name in ("tuning_fit", "tuning_heldout")
                if name in self._phases
            },
            "phase_totals": phase_totals,
            "phase_sum_consistent": True,
            "budget": {
                "max_forward_calls": self.max_forward_calls,
                "max_adjoint_calls": self.max_adjoint_calls,
                "forward_calls_used": self.counters.forward_calls,
                "adjoint_calls_used": self.counters.adjoint_calls,
                "forward_attempts": self.counters.forward_attempts,
                "adjoint_attempts": self.counters.adjoint_attempts,
                "forward_limit_reached": (
                    self.max_forward_calls is not None
                    and self.counters.forward_calls >= int(self.max_forward_calls)
                ),
                "adjoint_limit_reached": (
                    self.max_adjoint_calls is not None
                    and self.counters.adjoint_calls >= int(self.max_adjoint_calls)
                ),
                "budget_rejected": bool(
                    self.counters.budget_rejected_forward_calls
                    or self.counters.budget_rejected_adjoint_calls
                ),
                "budget_exhausted": bool(
                    self.counters.budget_rejected_forward_calls
                    or self.counters.budget_rejected_adjoint_calls
                ),
                "exhaustion_is_not_convergence": True,
            },
        })
        aggregate["phase_sum_consistent"] = all(
            (
                phase_totals[name] is None
                and aggregate.get(name) is None
            )
            or (
                phase_totals[name] is not None
                and abs(float(phase_totals[name]) - float(aggregate.get(name, 0.0) or 0.0)) <= 1e-9
            )
            for name in phase_totals
        )
        return aggregate


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
        if kind == "forward":
            self.counters.forward_attempts += 1
        else:
            self.counters.adjoint_attempts += 1
        if limit is not None and current >= int(limit):
            if kind == "forward":
                self.counters.budget_rejected_forward_calls += 1
            else:
                self.counters.budget_rejected_adjoint_calls += 1
            raise OperatorBudgetExceeded(
                f"{kind} operator-call budget exhausted: limit={int(limit)}",
                kind=kind,
                limit=int(limit),
                attempted=current + 1,
            )
        if kind == "forward":
            self.counters.forward_calls += 1
            self.counters.forward_executed_calls += 1
        else:
            self.counters.adjoint_calls += 1
            self.counters.adjoint_executed_calls += 1

    def _after(self) -> None:
        memory = _memory_mb()
        gpu_memory = _gpu_memory_mb()
        self.counters._update_peak(memory, gpu_memory)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._before("forward")
        started = time.perf_counter()
        gpu_started = _gpu_event_start(x)
        try:
            return self.operator.forward(x)
        except Exception:
            self.counters.forward_failed_calls += 1
            raise
        finally:
            self.counters.forward_seconds += time.perf_counter() - started
            gpu_elapsed = _gpu_event_elapsed(gpu_started, x)
            if gpu_elapsed is not None:
                self.counters.forward_gpu_seconds = float(self.counters.forward_gpu_seconds or 0.0) + gpu_elapsed
            self._after()

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        self._before("adjoint")
        started = time.perf_counter()
        gpu_started = _gpu_event_start(y)
        try:
            return self.operator.adjoint(y)
        except Exception:
            self.counters.adjoint_failed_calls += 1
            raise
        finally:
            self.counters.adjoint_seconds += time.perf_counter() - started
            gpu_elapsed = _gpu_event_elapsed(gpu_started, y)
            if gpu_elapsed is not None:
                self.counters.adjoint_gpu_seconds = float(self.counters.adjoint_gpu_seconds or 0.0) + gpu_elapsed
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
        started = time.perf_counter()
        gpu_started = _gpu_event_start(y)
        backend = getattr(self.operator, "fdk", None)
        if not callable(backend):
            error = AttributeError(f"{type(self.operator).__name__} has no FDK backend")
            self.counters.record_backend(
                elapsed_seconds=time.perf_counter() - started,
                gpu_elapsed_seconds=_gpu_event_elapsed(gpu_started, y),
                failed=True,
                exception=error,
            )
            self._after()
            raise error
        try:
            result = backend(y, **kwargs)
        except Exception as error:
            self.counters.record_backend(
                elapsed_seconds=time.perf_counter() - started,
                gpu_elapsed_seconds=_gpu_event_elapsed(gpu_started, y),
                failed=True,
                exception=error,
            )
            self._after()
            raise
        self.counters.record_backend(
            elapsed_seconds=time.perf_counter() - started,
            gpu_elapsed_seconds=_gpu_event_elapsed(gpu_started, y),
        )
        self._after()
        return result

    def record_prox(
        self,
        *,
        iterations: int = 0,
        elapsed_seconds: float = 0.0,
        gpu_elapsed_seconds: float | None = None,
        failed: bool = False,
    ) -> None:
        """Expose proximal accounting to detailed regularized solvers."""

        self.counters.record_prox(
            iterations=iterations,
            elapsed_seconds=elapsed_seconds,
            gpu_elapsed_seconds=gpu_elapsed_seconds,
            failed=failed,
        )

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

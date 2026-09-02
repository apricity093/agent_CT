from __future__ import annotations

import pytest
import torch
from pathlib import Path

from inv_framework.instrumentation import (
    CountingLinearOperator,
    OperatorBudgetExceeded,
    OperatorCounters,
    ResourceAccounting,
    start_memory_tracing,
)
from inv_framework.operators.base import LinearOperator
from inv_framework.regularizers import TVRegularizer
from inv_framework.solvers import SolveControl, TVFISTASolver


class IdentityOperator(LinearOperator):
    domain_shape = (1, 4, 4)
    range_shape = (1, 4, 4)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def adjoint(self, value: torch.Tensor) -> torch.Tensor:
        return value


class SubsetIdentityOperator(IdentityOperator):
    def __init__(self, calls: list[tuple[int, ...]] | None = None) -> None:
        self.calls = calls if calls is not None else []

    def subset(self, indices: torch.Tensor) -> "SubsetIdentityOperator":
        selected = tuple(int(value) for value in indices.detach().cpu().tolist())
        self.calls.append(selected)
        return SubsetIdentityOperator(self.calls)


class FDKOperator(IdentityOperator):
    def fdk(self, value: torch.Tensor, **_kwargs: object) -> torch.Tensor:
        return value


class FailingOperator(IdentityOperator):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("synthetic forward failure")


class FailingFDKOperator(FDKOperator):
    def fdk(self, value: torch.Tensor, **_kwargs: object) -> torch.Tensor:
        raise RuntimeError("synthetic fdk failure")


def test_attempts_and_budget_rejection_are_explicit_without_changing_call_formula():
    start_memory_tracing()
    counters = OperatorCounters()
    accounting = ResourceAccounting(counters, max_forward_calls=1)
    operator = CountingLinearOperator(
        IdentityOperator(), max_forward_calls=1, counters=counters
    )
    accounting.start_phase("solver")
    value = torch.ones(1, 1, 4, 4)
    operator.forward(value)
    with pytest.raises(OperatorBudgetExceeded):
        operator.forward(value)
    accounting.finish_phase("solver")

    record = accounting.to_dict()
    assert record["forward_calls"] == 1
    assert record["forward_attempts"] == 2
    assert record["forward_executed_calls"] == 1
    assert record["budget_rejected_forward_calls"] == 1
    assert record["total_operator_calls"] == 1
    assert record["total_operator_attempts"] == 2
    assert record["budget"]["budget_exhausted"] is True
    assert record["budget"]["exhaustion_is_not_convergence"] is True
    assert record["phases"]["solver"]["forward_attempts"] == 2
    assert record["phase_sum_consistent"] is True


def test_underlying_exception_is_counted_as_an_attempted_and_entered_call():
    operator = CountingLinearOperator(FailingOperator())
    with pytest.raises(RuntimeError, match="synthetic forward failure"):
        operator.forward(torch.ones(1, 1, 4, 4))
    stats = operator.stats()
    assert stats["forward_attempts"] == 1
    assert stats["forward_calls"] == 1
    assert stats["forward_executed_calls"] == 1
    assert stats["forward_failed_calls"] == 1


def test_subset_children_share_all_resource_counters():
    parent = CountingLinearOperator(SubsetIdentityOperator())
    child = parent.subset(torch.tensor([0, 1]))
    value = torch.ones(1, 1, 4, 4)
    child.forward(value)
    parent.adjoint(value)
    stats = parent.stats()
    assert stats["forward_calls"] == 1
    assert stats["adjoint_calls"] == 1
    assert stats["total_operator_calls"] == 2
    assert stats["forward_attempts"] == 1
    assert stats["adjoint_attempts"] == 1


@pytest.mark.parametrize(
    ("operator_type", "failed"),
    [(FDKOperator, False), (FailingFDKOperator, True)],
)
def test_native_fdk_backend_work_has_an_explicit_exception_path(operator_type, failed):
    operator = CountingLinearOperator(operator_type())
    value = torch.ones(1, 1, 4, 4)
    if failed:
        with pytest.raises(RuntimeError, match="synthetic fdk failure"):
            operator.fdk(value)
    else:
        assert torch.equal(operator.fdk(value), value)
    stats = operator.stats()
    assert stats["backend_reconstruction_calls"] == 1
    assert stats["backend_executed_calls"] == (0 if failed else 1)
    assert stats["backend_failed_calls"] == (1 if failed else 0)
    assert stats["backend_seconds"] >= 0.0
    if failed:
        assert "synthetic fdk failure" in stats["backend_exception"]


def test_tv_prox_work_reports_actual_calls_and_iterations():
    operator = CountingLinearOperator(IdentityOperator())
    result = TVFISTASolver(
        num_iterations=1,
        step_size=0.1,
        tolerance=0.0,
        regularizer=TVRegularizer(num_iterations=2, tolerance=0.0),
    ).solve_detailed(
        torch.ones(1, 1, 4, 4),
        operator,
        control=SolveControl(max_iterations=1, tolerance=0.0),
    )
    assert result.status == "max_iterations"
    assert operator.stats()["prox_calls"] == 1
    assert operator.stats()["prox_iterations"] == 2
    assert operator.stats()["prox_seconds"] >= 0.0


def test_cpu_resource_record_has_memory_and_null_gpu_fields():
    start_memory_tracing()
    counters = OperatorCounters()
    accounting = ResourceAccounting(counters)
    accounting.start_phase("preparation")
    CountingLinearOperator(IdentityOperator(), counters=counters).forward(
        torch.ones(1, 1, 4, 4)
    )
    accounting.finish_phase("preparation")
    record = accounting.to_dict()
    assert record["peak_memory_mb"] is not None
    assert record["peak_gpu_memory_mb"] is None
    assert record["gpu_runtime_seconds"] is None
    assert record["phases"]["preparation"]["peak_memory_mb"] is not None
    assert record["phases"]["preparation"]["peak_gpu_memory_mb"] is None


def test_runtime_budget_failure_keeps_phase_and_status_evidence(tmp_path: Path):
    root = Path(__file__).parents[1]
    from inv_framework.ct_runtime import run_case

    result = run_case(
        "sirt",
        "parallel_2d/shepp_logan_sparse_poisson_32",
        root / "configs" / "algorithms" / "sirt.yaml",
        tmp_path / "run",
        data_root=root / "test" / "data",
        max_iterations=1,
        max_forward_calls=1,
        parameter_overrides={"num_iterations": 1},
    )
    resources = result["diagnostics"]["resources"]
    assert result["status"] == "resource_exhausted"
    assert result["convergence_status"] == "resource_exhausted"
    assert resources["budget"]["budget_exhausted"] is True
    assert resources["forward_attempts"] >= 1
    assert resources["budget_rejected_forward_calls"] >= 1
    assert resources["phase_sum_consistent"] is True
    assert resources["phases"]["solver"]["forward_attempts"] >= 1

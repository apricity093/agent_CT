"""Filtered backprojection for parallel and cone-beam geometry."""
from __future__ import annotations
from torch.fft import rfft, irfft
from inv_framework.core.operator import LinearOperator
from inv_framework.solvers.base import InverseProblemSolver, SolveControl, SolveResult, IterationRecorder, resolve_control
from typing import Any, Mapping
from inv_framework.solvers._utils import require_linear_operator, validate_measurement_shape
import math
import torch


def _ram_lak(n: int, device) -> torch.Tensor:
    f = torch.zeros(n, device=device)
    f[0] = 0.25
    odd = torch.arange(1, n, 2, device=device).float()
    odd_adj = torch.where(2 * odd > n, n - odd, odd)
    f[1::2] = -1.0 / (math.pi * odd_adj) ** 2
    return f


def _filter_sino(sino: torch.Tensor, padded: bool = True) -> torch.Tensor:
    """Apply the Ram-Lak filter along the last (detector) dimension."""
    W = sino.shape[-1]
    filt_w = 2 * W if padded else W
    filt = _ram_lak(filt_w, sino.device)
    filt_rfft = rfft(filt)
    sino_rfft = rfft(sino, n=filt_w) * filt_rfft
    out = irfft(sino_rfft, n=filt_w)
    return out[..., :W]


def fbp(operator: LinearOperator,
        y: torch.Tensor,
        scale: float = None) -> torch.Tensor:
    """Filtered Back-Projection reconstruction."""
    require_linear_operator(operator, "fbp")
    y_filt = _filter_sino(y)
    rec = operator.adjoint(y_filt)
    if scale is None:
        # Standard parallel-beam FBP scaling.
        num_angles = getattr(operator, "num_angles", y.shape[-2])
        scale = math.pi / num_angles
    return rec * scale


class FBPSolver(InverseProblemSolver):
    def __init__(self, scale: float = None):
        self.scale = scale

    def solve(self, measurement, operator, **kwargs):
        return fbp(operator, measurement, scale=self.scale)

    def solve_detailed(self, measurement, operator, *, control=None, callback=None, **kwargs):
        pass

        return solve_fbp_detailed(
            operator, measurement, scale=kwargs.pop("scale", self.scale),
            control=control, callback=callback, **kwargs,
        )


def fdk(operator: LinearOperator, y: torch.Tensor, **kwargs) -> torch.Tensor:
    """Geometry-specific FDK reconstruction via an operator backend."""

    require_linear_operator(operator, "fdk")
    validate_measurement_shape(y, operator, "fdk")
    backend = getattr(operator, "fdk", None)
    if not callable(backend):
        raise NotImplementedError(
            "fdk requires a LinearOperator with an operator.fdk(y, **kwargs) "
            "backend; this project does not provide a generic FDK kernel."
        )
    rec = backend(y, **kwargs)
    if not isinstance(rec, torch.Tensor):
        raise TypeError(f"operator.fdk(...) must return a torch.Tensor; got {type(rec).__name__}.")
    expected = (y.shape[0], *tuple(operator.domain_shape))
    if tuple(rec.shape) != expected:
        raise ValueError(f"operator.fdk(...) must return shape {expected}; got {tuple(rec.shape)}.")
    return rec.to(device=y.device, dtype=y.dtype)


class FDKSolver(InverseProblemSolver):
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)

    def solve(self, measurement, operator, **kwargs):
        params = dict(self.kwargs)
        params.update(kwargs)
        return fdk(operator, measurement, **params)

    def solve_detailed(self, measurement, operator, *, control=None, callback=None, **kwargs):
        pass

        params = dict(self.kwargs)
        params.update(kwargs)
        return solve_fdk_detailed(operator, measurement, control=control, callback=callback, **params)


def _direct_result(
    algorithm: str,
    measurement: Any,
    operator: Any,
    *,
    status: str,
    stopping_reason: str,
    parameters: Mapping[str, Any],
    error: BaseException | None = None,
) -> SolveResult:
    """Return a structured direct-solver outcome without entering a loop."""
    from inv_framework.solvers._iteration import _placeholder_reconstruction

    metadata: dict[str, Any] = {
        "algorithm": algorithm,
        "direct": True,
        "max_iterations": 1,
        "parameters": dict(parameters),
    }
    if error is not None:
        metadata["failure_reason"] = str(error)
    return SolveResult(
        reconstruction=_placeholder_reconstruction(measurement, operator),
        actual_iterations=0,
        status=status,
        stopping_reason=stopping_reason,
        resources={"trajectory_available": False},
        metadata=metadata,
    )


def solve_fbp_detailed(
    operator: LinearOperator,
    measurement: torch.Tensor,
    *,
    scale: float | None = None,
    control: SolveControl | None = None,
    callback: Callback | None = None,
    **kwargs: Any,
) -> SolveResult:
    from inv_framework.solvers._iteration import _finite_positive, _finite_tensor
    pass

    _ = kwargs
    parameters = {"scale": scale}
    try:
        require_linear_operator(operator, "fbp")
        validate_measurement_shape(measurement, operator, "fbp")
    except (TypeError, ValueError) as error:
        return _direct_result(
            "fbp", measurement, operator, status="invalid_parameters",
            stopping_reason="parameter_validation_failed", parameters=parameters,
            error=error,
        )
    if scale is not None and not _finite_positive(scale):
        return _direct_result(
            "fbp", measurement, operator, status="invalid_parameters",
            stopping_reason="invalid_scale", parameters=parameters,
            error=ValueError("scale must be a finite positive number or None"),
        )
    if not _finite_tensor(measurement):
        return _direct_result(
            "fbp", measurement, operator, status="numerical_error",
            stopping_reason="non_finite_measurement", parameters=parameters,
            error=FloatingPointError("FBP measurement contains NaN or Inf"),
        )
    try:
        control = resolve_control(control, default_iterations=1, callback=callback)
    except (TypeError, ValueError) as error:
        return _direct_result(
            "fbp", measurement, operator, status="invalid_parameters",
            stopping_reason="invalid_control", parameters=parameters, error=error,
        )
    try:
        reconstruction = fbp(operator, measurement, scale=scale)
    except Exception as error:
        return _direct_result(
            "fbp", measurement, operator, status="numerical_error",
            stopping_reason="fbp_execution_failed", parameters=parameters, error=error,
        )
    expected = (measurement.shape[0], *tuple(operator.domain_shape))
    if not isinstance(reconstruction, torch.Tensor) or tuple(reconstruction.shape) != expected:
        return _direct_result(
            "fbp", measurement, operator, status="numerical_error",
            stopping_reason="invalid_reconstruction_shape", parameters=parameters,
            error=ValueError(
                f"FBP returned {type(reconstruction).__name__} with shape "
                f"{getattr(reconstruction, 'shape', None)}; expected {expected}"
            ),
        )
    if not _finite_tensor(reconstruction):
        return _direct_result(
            "fbp", measurement, operator, status="numerical_error",
            stopping_reason="non_finite_reconstruction", parameters=parameters,
            error=FloatingPointError("FBP returned NaN or Inf"),
        )
    recorder = IterationRecorder(control, measurement, operator, algorithm="fbp")
    return recorder.finish(
        reconstruction,
        actual_iterations=0,
        status="completed_valid",
        stopping_reason="direct_reconstruction_valid",
        resources={"trajectory_available": False, "direct": True},
        metadata={"direct": True, "parameters": parameters, "completed_valid": True},
    )


def solve_fdk_detailed(
    operator: LinearOperator,
    measurement: torch.Tensor,
    *,
    control: SolveControl | None = None,
    callback: Callback | None = None,
    **kwargs: Any,
) -> SolveResult:
    from inv_framework.solvers._iteration import _finite_tensor
    pass

    parameters = dict(kwargs)
    for name, value in (
        ("filter_type", parameters.get("filter_type", "ram-lak")),
        ("filter", parameters.get("filter")),
    ):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            return _direct_result(
                "fdk", measurement, operator, status="invalid_parameters",
                stopping_reason=f"invalid_{name}", parameters=parameters,
                error=ValueError(f"{name} must be a non-empty string"),
            )
    filter_type = str(parameters.get("filter_type", "ram-lak")).strip().lower().replace("_", "-")
    filter_alias = parameters.get("filter")
    if filter_type in {"ramp", "ramlak"}:
        filter_type = "ram-lak"
    if filter_alias is not None:
        filter_alias = str(filter_alias).strip().lower().replace("_", "-")
        if filter_alias in {"ramp", "ramlak"}:
            filter_alias = "ram-lak"
    if filter_type != "ram-lak" or (
        filter_alias is not None and filter_alias != filter_type
    ):
        return _direct_result(
            "fdk", measurement, operator, status="invalid_parameters",
            stopping_reason="invalid_filter", parameters=parameters,
            error=ValueError(
                "FDK supports only the native Ram-Lak filter and compatible aliases"
            ),
        )
    short_scan = parameters.get("short_scan", False)
    if not isinstance(short_scan, bool):
        return _direct_result(
            "fdk", measurement, operator, status="invalid_parameters",
            stopping_reason="invalid_short_scan", parameters=parameters,
            error=TypeError("short_scan must be a bool"),
        )
    supersampling = parameters.get("voxel_supersampling", 1)
    if (
        isinstance(supersampling, bool)
        or not isinstance(supersampling, int)
        or supersampling <= 0
    ):
        return _direct_result(
            "fdk", measurement, operator, status="invalid_parameters",
            stopping_reason="invalid_voxel_supersampling", parameters=parameters,
            error=ValueError("voxel_supersampling must be a positive integer"),
        )
    try:
        require_linear_operator(operator, "fdk")
        validate_measurement_shape(measurement, operator, "fdk")
    except (TypeError, ValueError) as error:
        return _direct_result(
            "fdk", measurement, operator, status="invalid_parameters",
            stopping_reason="parameter_validation_failed", parameters=parameters,
            error=error,
        )
    if not _finite_tensor(measurement):
        return _direct_result(
            "fdk", measurement, operator, status="numerical_error",
            stopping_reason="non_finite_measurement", parameters=parameters,
            error=FloatingPointError("FDK measurement contains NaN or Inf"),
        )
    try:
        control = resolve_control(control, default_iterations=1, callback=callback)
    except (TypeError, ValueError) as error:
        return _direct_result(
            "fdk", measurement, operator, status="invalid_parameters",
            stopping_reason="invalid_control", parameters=parameters, error=error,
        )
    try:
        reconstruction = fdk(operator, measurement, **parameters)
    except ImportError as error:
        return _direct_result(
            "fdk", measurement, operator, status="unavailable",
            stopping_reason="backend_unavailable", parameters=parameters, error=error,
        )
    except NotImplementedError as error:
        message = str(error).lower()
        invalid_backend_options = (
            "would be ignored",
            "requires cone geometry",
            "unsupported",
            "filter",
        )
        status = "invalid_parameters" if any(token in message for token in invalid_backend_options) else "unavailable"
        return _direct_result(
            "fdk", measurement, operator, status=status,
            stopping_reason="parameter_validation_failed" if status == "invalid_parameters" else "backend_unavailable",
            parameters=parameters, error=error,
        )
    except RuntimeError as error:
        message = str(error).lower()
        unavailable_tokens = ("cuda", "astra", "not installed", "unavailable", "no fdk backend")
        status = "unavailable" if any(token in message for token in unavailable_tokens) else "numerical_error"
        return _direct_result(
            "fdk", measurement, operator, status=status,
            stopping_reason="backend_unavailable" if status == "unavailable" else "fdk_execution_failed",
            parameters=parameters, error=error,
        )
    except (TypeError, ValueError) as error:
        message = str(error).lower()
        # The wrapper's own return-contract errors describe an invalid
        # backend result, while adapter option/geometry errors are invalid
        # parameters.  Preserve that distinction in the detailed API.
        output_error = "must return" in message or "returned shape" in message
        return _direct_result(
            "fdk", measurement, operator,
            status="numerical_error" if output_error else "invalid_parameters",
            stopping_reason="invalid_reconstruction_output" if output_error else "parameter_validation_failed",
            parameters=parameters, error=error,
        )
    except AttributeError as error:
        return _direct_result(
            "fdk", measurement, operator, status="unavailable",
            stopping_reason="backend_unavailable", parameters=parameters, error=error,
        )
    except Exception as error:
        return _direct_result(
            "fdk", measurement, operator, status="numerical_error",
            stopping_reason="fdk_execution_failed", parameters=parameters, error=error,
        )
    expected = (measurement.shape[0], *tuple(operator.domain_shape))
    if not isinstance(reconstruction, torch.Tensor) or tuple(reconstruction.shape) != expected:
        return _direct_result(
            "fdk", measurement, operator, status="numerical_error",
            stopping_reason="invalid_reconstruction_shape", parameters=parameters,
            error=ValueError(f"FDK returned shape {getattr(reconstruction, 'shape', None)}; expected {expected}"),
        )
    if not _finite_tensor(reconstruction):
        return _direct_result(
            "fdk", measurement, operator, status="numerical_error",
            stopping_reason="non_finite_reconstruction", parameters=parameters,
            error=FloatingPointError("FDK returned NaN or Inf"),
        )
    recorder = IterationRecorder(control, measurement, operator, algorithm="fdk")
    return recorder.finish(
        reconstruction,
        actual_iterations=0,
        status="completed_valid",
        stopping_reason="direct_reconstruction_valid",
        resources={"trajectory_available": False, "direct": True},
        metadata={
            "direct": True,
            "parameters": parameters,
            "completed_valid": True,
        },
    )

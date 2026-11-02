"""ct backends leap_adapter implementation."""
from __future__ import annotations

import torch
from inv_framework.core.operator import LinearOperator


def _new_leap_model():
    try:
        from leapctype import tomographicModels
    except ImportError as error:
        raise ImportError(
            "LEAPOperator3D requires LLNL LEAP. Install LEAP and make its "
            "leapctype Python module importable, or pass a configured model."
        ) from error
    return tomographicModels()


def _model_size(model, getter_name: str) -> int:
    getter = getattr(model, getter_name, None)
    if not callable(getter):
        raise TypeError(f"LEAP model must provide {getter_name}().")
    value = int(getter())
    if value <= 0:
        raise ValueError(
            f"LEAP geometry is not configured: {getter_name}() returned {value}."
        )
    return value


class _LEAPProjectFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, volume, adapter):
        ctx.adapter = adapter
        return adapter._project_impl(volume)

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.adapter._backproject_impl(grad_output), None


class _LEAPBackprojectFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, projection, adapter):
        ctx.adapter = adapter
        return adapter._backproject_impl(projection)

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.adapter._project_impl(grad_output), None


class LEAPOperator3D(LinearOperator):
    """Wrap a configured ``leapctype.tomographicModels`` as a LinearOperator.

    The framework interface is batched even though calls into the stateful
    LEAP model are made one sample at a time.  Internal calculations use
    contiguous float32 tensors, while results preserve the caller's dtype and
    device.  A single adapter/model instance must not be used concurrently
    from multiple threads.
    """

    def __init__(self, model=None):
        self.model = _new_leap_model() if model is None else model
        self.domain_shape = (
            _model_size(self.model, "get_numZ"),
            _model_size(self.model, "get_numY"),
            _model_size(self.model, "get_numX"),
        )
        self.range_shape = (
            _model_size(self.model, "get_numAngles"),
            _model_size(self.model, "get_numRows"),
            _model_size(self.model, "get_numCols"),
        )
        self.num_angles = self.range_shape[0]

    @staticmethod
    def _validate_batch(tensor: torch.Tensor, sample_shape: tuple, name: str) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor; got {type(tensor).__name__}.")
        expected_ndim = len(sample_shape) + 1
        if tensor.ndim != expected_ndim or tuple(tensor.shape[1:]) != sample_shape:
            raise ValueError(
                f"{name} must have shape (B, {', '.join(map(str, sample_shape))}); "
                f"got {tuple(tensor.shape)}."
            )

    @staticmethod
    def _accept_backend_result(result, output: torch.Tensor, method_name: str) -> None:
        if result is False:
            raise RuntimeError(f"LEAP {method_name}() reported failure.")
        if isinstance(result, torch.Tensor) and result.data_ptr() != output.data_ptr():
            if tuple(result.shape) != tuple(output.shape):
                raise ValueError(
                    f"LEAP {method_name}() returned shape {tuple(result.shape)}; "
                    f"expected {tuple(output.shape)}."
                )
            output.copy_(result.to(device=output.device, dtype=output.dtype))

    def _apply_model(
        self,
        method_name: str,
        source: torch.Tensor,
        output_shape: tuple,
        output_first: bool = False,
        **kwargs,
    ):
        original_device = source.device
        original_dtype = source.dtype
        source_work = source.detach().to(dtype=torch.float32).contiguous()
        output_work = torch.empty(
            (source.shape[0], *output_shape), dtype=torch.float32, device=source.device
        )
        method = getattr(self.model, method_name, None)
        if not callable(method):
            raise TypeError(f"LEAP model must provide {method_name}().")

        for index in range(source.shape[0]):
            if output_first:
                result = method(output_work[index], source_work[index], **kwargs)
            else:
                result = method(source_work[index], output_work[index], **kwargs)
            self._accept_backend_result(result, output_work[index], method_name)
        return output_work.to(device=original_device, dtype=original_dtype)

    def _project_impl(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_batch(x, self.domain_shape, "x")
        return self._apply_model("project", x, self.range_shape, output_first=True)

    def _backproject_impl(self, y: torch.Tensor) -> torch.Tensor:
        self._validate_batch(y, self.range_shape, "y")
        return self._apply_model("backproject", y, self.domain_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _LEAPProjectFunction.apply(x, self)

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        return _LEAPBackprojectFunction.apply(y, self)

    def fdk(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        """Run LEAP's generic FBP entry, which performs FDK for cone geometry."""
        self._validate_batch(y, self.range_shape, "y")
        geometry_getter = getattr(self.model, "get_geometry", None)
        if not callable(geometry_getter):
            raise TypeError("LEAP model must provide get_geometry() so fdk can verify cone geometry.")
        geometry = geometry_getter()
        is_cone = (
            isinstance(geometry, str) and geometry.strip().lower() == "cone"
        ) or (not isinstance(geometry, str) and geometry == 0)
        if not is_cone:
            raise NotImplementedError(
                f"LEAPOperator3D.fdk requires cone geometry; got {geometry!r}."
            )
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(
                f"LEAP FBP accepts no adapter options other than controlled inplace=False; got {names}."
            )
        return self._apply_model("FBP", y, self.domain_shape, inplace=False)


class LEAPOperator2D(LinearOperator):
    """Parallel-beam slices using LEAP's single-row projection geometry."""

    nonnegative_matrix = True

    def __init__(self, image_size, angles, device="cpu", in_channels=1):
        import numpy as np
        self.angles = torch.as_tensor(angles).cpu()
        self.device = device
        self.domain_shape = (in_channels, image_size, image_size)
        self.range_shape = (in_channels, len(angles), image_size)
        self.num_angles = len(angles)
        model = _new_leap_model()
        model.set_gpu(-1 if torch.device(device).type == "cpu" else (torch.device(device).index or 0))
        phis = (self.angles.numpy()*180/np.pi + 90).astype(np.float32)
        if not model.set_parallelbeam(len(angles), 1, image_size, 1.0, 1.0, 0.0,
                                      (image_size-1)/2, phis):
            raise ValueError("LEAP rejected parallel geometry")
        if not model.set_volume(image_size, image_size, 1, 1.0, 1.0):
            raise ValueError("LEAP rejected volume geometry")
        self.operator = LEAPOperator3D(model)
        self.subsets = {}

    def forward(self, x):
        batch, channels, height, width = x.shape
        result = self.operator.forward(x.reshape(batch*channels, 1, height, width))
        return result.reshape(batch, channels, self.num_angles, width)

    def adjoint(self, y):
        batch, channels, views, width = y.shape
        result = self.operator.adjoint(y.reshape(batch*channels, views, 1, width))
        return result.reshape(batch, *self.domain_shape)

    def subset(self, indices):
        key = tuple(indices.cpu().tolist())
        if key not in self.subsets:
            self.subsets[key] = LEAPOperator2D(self.domain_shape[-1], self.angles[list(key)],
                                              self.device, self.domain_shape[0])
        return self.subsets[key]

    def pseudo_inverse(self, y, **kwargs):
        from inv_framework.solvers.analytic import fbp
        return fbp(self, y)

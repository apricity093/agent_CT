"""Complete cone-beam FDK reconstruction through ASTRA's CUDA backend.

The numerical FDK pipeline is provided by ASTRA ``FDK_CUDA``.  This adapter
keeps ASTRA geometry and data-object details behind the framework's existing
``LinearOperator.fdk`` seam.

Source: https://github.com/astra-toolbox/astra-toolbox (GPL-3.0).
"""

from numbers import Integral

import numpy as np
import torch

from . import astra_adapter as _astra_backend
from .astra_adapter import ASTRAOperator3D


_FILTER_ALIASES = {
    "ramp": "ram-lak",
    "ramlak": "ram-lak",
    "ram_lak": "ram-lak",
    "shepp_logan": "shepp-logan",
}


def _normalise_filter_name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("filter_type must be a non-empty string.")
    name = value.strip().lower()
    return _FILTER_ALIASES.get(name, name.replace("_", "-"))


def _voxel_sizes(volume_geometry) -> tuple:
    option = volume_geometry.get("option", {})
    axes = (
        ("X", "GridColCount"),
        ("Y", "GridRowCount"),
        ("Z", "GridSliceCount"),
    )
    sizes = []
    for axis, count_key in axes:
        min_key = f"WindowMin{axis}"
        max_key = f"WindowMax{axis}"
        if count_key not in volume_geometry or min_key not in option or max_key not in option:
            raise ValueError(
                "ASTRA FDK requires a volume geometry with explicit grid counts "
                "and WindowMin/WindowMax bounds for X, Y, and Z."
            )
        count = int(volume_geometry[count_key])
        if count <= 0:
            raise ValueError(f"{count_key} must be positive; got {count}.")
        sizes.append((float(option[max_key]) - float(option[min_key])) / count)
    return tuple(sizes)


def _validate_fdk_geometry(volume_geometry, projection_geometry) -> None:
    geometry_type = str(projection_geometry.get("type", "")).lower()
    if geometry_type != "cone":
        raise ValueError(
            "ASTRAFDKOperator3D supports only regular ASTRA 'cone' geometry; "
            f"got {geometry_type or 'an unspecified geometry'!r}. "
            "Parallel and cone_vec geometries are not accepted."
        )
    voxel_sizes = _voxel_sizes(volume_geometry)
    if not np.allclose(voxel_sizes, voxel_sizes[0], rtol=1e-6, atol=1e-7):
        raise ValueError(
            "ASTRA FDK_CUDA requires cubic voxels; computed voxel sizes are "
            f"{voxel_sizes}."
        )


def _gpu_link(tensor: torch.Tensor):
    if tensor.ndim != 3 or not tensor.is_cuda or tensor.dtype != torch.float32:
        raise ValueError("ASTRA GPU linking requires a contiguous 3D CUDA float32 tensor.")
    if not tensor.is_contiguous():
        raise ValueError("ASTRA GPU linking requires contiguous tensor storage.")
    z, y, x = (int(size) for size in tensor.shape)
    return _astra_backend.astra.data3d.GPULink(
        tensor.data_ptr(), x, y, z, x * tensor.element_size()
    )


class _ASTRAFDKForwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, volume, operator):
        ctx.operator = operator
        return operator._apply_projection_algorithm(volume, forward=True)

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.operator._apply_projection_algorithm(grad_output, forward=False), None


class _ASTRAFDKAdjointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, projection, operator):
        ctx.operator = operator
        return operator._apply_projection_algorithm(projection, forward=False)

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.operator._apply_projection_algorithm(grad_output, forward=True), None


class ASTRAFDKOperator3D(ASTRAOperator3D):
    """ASTRA 3D projector with a complete ``FDK_CUDA`` reconstruction backend.

    Only ASTRA's regular circular ``cone`` geometry with cubic voxels is
    accepted.  Each input must be batched as ``(B, detector_rows, angles,
    detector_cols)``.  CUDA float32 tensors are linked directly to ASTRA;
    other inputs use ASTRA data objects and are restored to the caller's
    original device and dtype.
    """

    def __init__(self, volume_geometry, projection_geometry):
        _validate_fdk_geometry(volume_geometry, projection_geometry)
        _astra_backend._require_astra()
        astra = _astra_backend.astra
        self.volume_geometry = volume_geometry
        self.projection_geometry = projection_geometry
        self.projection_shape = tuple(astra.geom_size(projection_geometry))
        self.volume_shape = tuple(astra.geom_size(volume_geometry))
        self.domain_shape = self.volume_shape
        self.range_shape = self.projection_shape
        if "ProjectionAngles" in projection_geometry:
            self.num_angles = len(projection_geometry["ProjectionAngles"])
        else:
            self.num_angles = len(projection_geometry["Vectors"])

    def _run_projection_one(self, source: torch.Tensor, forward: bool) -> torch.Tensor:
        astra = _astra_backend.astra
        volume_id = None
        projection_id = None
        algorithm_id = None
        output = None
        try:
            if forward:
                volume = source
                projection = torch.zeros(
                    self.projection_shape, dtype=torch.float32, device=source.device
                ) if source.is_cuda else None
            else:
                projection = source
                volume = torch.zeros(
                    self.volume_shape, dtype=torch.float32, device=source.device
                ) if source.is_cuda else None

            if source.is_cuda:
                torch.cuda.synchronize(source.device)
                volume_id = astra.data3d.link(
                    "-vol", self.volume_geometry, _gpu_link(volume)
                )
                projection_id = astra.data3d.link(
                    "-sino", self.projection_geometry, _gpu_link(projection)
                )
                output = projection if forward else volume
            else:
                if forward:
                    volume_id = astra.data3d.create(
                        "-vol",
                        self.volume_geometry,
                        np.ascontiguousarray(source.detach().cpu().numpy(), dtype=np.float32),
                    )
                    projection_id = astra.data3d.create(
                        "-sino", self.projection_geometry, 0.0
                    )
                else:
                    projection_id = astra.data3d.create(
                        "-sino",
                        self.projection_geometry,
                        np.ascontiguousarray(source.detach().cpu().numpy(), dtype=np.float32),
                    )
                    volume_id = astra.data3d.create("-vol", self.volume_geometry, 0.0)

            algorithm_type = "FP3D_CUDA" if forward else "BP3D_CUDA"
            config = astra.astra_dict(algorithm_type)
            config["ProjectionDataId"] = projection_id
            if forward:
                config["VolumeDataId"] = volume_id
            else:
                config["ReconstructionDataId"] = volume_id
            algorithm_id = astra.algorithm.create(config)
            astra.algorithm.run(algorithm_id)

            if output is not None:
                torch.cuda.synchronize(source.device)
                return output
            output_id = projection_id if forward else volume_id
            output_array = np.asarray(astra.data3d.get(output_id), dtype=np.float32)
            return torch.from_numpy(np.ascontiguousarray(output_array))
        finally:
            if algorithm_id is not None:
                astra.algorithm.delete(algorithm_id)
            if projection_id is not None:
                astra.data3d.delete(projection_id)
            if volume_id is not None:
                astra.data3d.delete(volume_id)

    def _apply_projection_algorithm(self, tensor: torch.Tensor, forward: bool) -> torch.Tensor:
        input_shape = self.domain_shape if forward else self.range_shape
        name = "x" if forward else "y"
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor; got {type(tensor).__name__}.")
        if tensor.ndim != len(input_shape) + 1 or tuple(tensor.shape[1:]) != input_shape:
            raise ValueError(
                f"{name} must have shape (B, {', '.join(map(str, input_shape))}); "
                f"got {tuple(tensor.shape)}."
            )
        if tensor.shape[0] == 0:
            raise ValueError("ASTRA projection algorithms require a non-empty batch.")

        work = tensor.detach().to(dtype=torch.float32).contiguous()
        outputs = [self._run_projection_one(work[index], forward) for index in range(tensor.shape[0])]
        return torch.stack(outputs, dim=0).to(device=tensor.device, dtype=tensor.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _ASTRAFDKForwardFunction.apply(x, self)

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        return _ASTRAFDKAdjointFunction.apply(y, self)

    def _run_fdk_one(self, projection: torch.Tensor, options: dict) -> torch.Tensor:
        astra = _astra_backend.astra
        projection_id = None
        reconstruction_id = None
        algorithm_id = None
        reconstruction = None
        try:
            if projection.is_cuda:
                reconstruction = torch.zeros(
                    self.volume_shape, dtype=torch.float32, device=projection.device
                )
                projection_id = astra.data3d.link(
                    "-sino", self.projection_geometry, _gpu_link(projection)
                )
                reconstruction_id = astra.data3d.link(
                    "-vol", self.volume_geometry, _gpu_link(reconstruction)
                )
            else:
                projection_array = np.ascontiguousarray(
                    projection.detach().cpu().numpy(), dtype=np.float32
                )
                projection_id = astra.data3d.create(
                    "-sino", self.projection_geometry, projection_array
                )
                reconstruction_id = astra.data3d.create(
                    "-vol", self.volume_geometry, 0.0
                )

            config = astra.astra_dict("FDK_CUDA")
            config["ProjectionDataId"] = projection_id
            config["ReconstructionDataId"] = reconstruction_id
            config["option"] = dict(options)
            algorithm_id = astra.algorithm.create(config)
            if projection.is_cuda:
                torch.cuda.synchronize(projection.device)
            astra.algorithm.run(algorithm_id)

            if reconstruction is not None:
                torch.cuda.synchronize(projection.device)
                return reconstruction
            reconstruction_array = np.asarray(
                astra.data3d.get(reconstruction_id), dtype=np.float32
            )
            return torch.from_numpy(np.ascontiguousarray(reconstruction_array))
        finally:
            if algorithm_id is not None:
                astra.algorithm.delete(algorithm_id)
            if reconstruction_id is not None:
                astra.data3d.delete(reconstruction_id)
            if projection_id is not None:
                astra.data3d.delete(projection_id)

    def fdk(
        self,
        y: torch.Tensor,
        filter_type: str = "ram-lak",
        short_scan: bool = False,
        voxel_supersampling: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """Reconstruct a batch using ASTRA's weighting/filter/backprojection pipeline.

        ``filter=...`` is accepted as an alias for ``filter_type``.  The ASTRA
        FDK backend provides Ram-Lak filtering and short-scan Parker weighting.
        """
        if not isinstance(y, torch.Tensor):
            raise TypeError(f"y must be a torch.Tensor; got {type(y).__name__}.")
        expected = (y.shape[0], *self.range_shape) if y.ndim else None
        if y.ndim != len(self.range_shape) + 1 or tuple(y.shape) != expected:
            raise ValueError(
                f"ASTRA FDK expects shape (B, {', '.join(map(str, self.range_shape))}); "
                f"got {tuple(y.shape)}."
            )
        if y.shape[0] == 0:
            raise ValueError("ASTRA FDK requires a non-empty batch.")

        filter_alias = kwargs.pop("filter", None)
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported ASTRA FDK option(s): {names}.")
        canonical_filter = _normalise_filter_name(filter_type)
        if filter_alias is not None:
            alias_filter = _normalise_filter_name(filter_alias)
            if canonical_filter != "ram-lak" and canonical_filter != alias_filter:
                raise ValueError("filter and filter_type specify different ASTRA filters.")
            canonical_filter = alias_filter
        if canonical_filter != "ram-lak":
            raise NotImplementedError(
                "ASTRA FDK_CUDA exposes only its native Ram-Lak filtering path; "
                f"filter {canonical_filter!r} would be ignored by the backend."
            )
        if not isinstance(short_scan, bool):
            raise TypeError("short_scan must be a bool.")
        if isinstance(voxel_supersampling, bool) or not isinstance(voxel_supersampling, Integral):
            raise TypeError("voxel_supersampling must be a positive integer.")
        if int(voxel_supersampling) < 1:
            raise ValueError("voxel_supersampling must be at least 1.")

        options = {
            "ShortScan": bool(short_scan),
            "VoxelSuperSampling": int(voxel_supersampling),
        }
        original_device = y.device
        original_dtype = y.dtype
        work = y.detach().to(dtype=torch.float32).contiguous()
        reconstructions = [self._run_fdk_one(work[index], options) for index in range(y.shape[0])]
        result = torch.stack(reconstructions, dim=0)
        return result.to(device=original_device, dtype=original_dtype)

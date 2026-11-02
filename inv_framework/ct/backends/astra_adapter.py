"""ct backends astra_adapter implementation."""
from __future__ import annotations

import torch
from inv_framework.core.operator import LinearOperator


try:
    import astra
    import astra.experimental  # noqa: F401
    _HAS_ASTRA = True
except ImportError as _e:
    _HAS_ASTRA = False
    _IMPORT_ERROR = _e


def _require_astra():
    if not _HAS_ASTRA:
        raise ImportError(
            "ASTRAOperator3D requires astra-toolbox. "
            "Install with: conda install -c astra-toolbox astra-toolbox"
        ) from _IMPORT_ERROR


class _ASTRAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, volume, projector, projection_shape, volume_shape):
        if volume.ndim == 4:
            B = volume.shape[0]
            proj = torch.zeros((B, *projection_shape), dtype=torch.float32,
                               device='cuda')
            for i in range(B):
                astra.experimental.direct_FP3D(
                    projector, vol=volume[i].contiguous().detach(),
                    proj=proj[i])
        elif volume.ndim == 3:
            proj = torch.zeros(projection_shape, dtype=torch.float32,
                               device='cuda')
            astra.experimental.direct_FP3D(
                projector, vol=volume.detach(), proj=proj)
        else:
            raise ValueError(f"Unsupported volume.ndim={volume.ndim}")
        ctx.projector = projector
        ctx.volume_shape = volume_shape
        return proj

    @staticmethod
    def backward(ctx, grad_output):
        projector = ctx.projector
        volume_shape = ctx.volume_shape
        if grad_output.ndim == 4:
            B = grad_output.shape[0]
            grad_vol = torch.zeros((B, *volume_shape), dtype=torch.float32,
                                   device='cuda')
            for i in range(B):
                astra.experimental.direct_BP3D(
                    projector, vol=grad_vol[i],
                    proj=grad_output[i].contiguous().detach())
        else:
            grad_vol = torch.zeros(volume_shape, dtype=torch.float32,
                                   device='cuda')
            astra.experimental.direct_BP3D(
                projector, vol=grad_vol, proj=grad_output.detach())
        return grad_vol, None, None, None


class ASTRAOperator3D(LinearOperator):
    """ASTRA CUDA3D projector wrapped as a LinearOperator.

    Parameters
    ----------
    volume_geometry : dict
        astra volume geometry (e.g. astra.create_vol_geom(...)).
    projection_geometry : dict
        astra projection geometry (parallel or cone).
    """

    def __init__(self, volume_geometry, projection_geometry):
        _require_astra()
        self.volume_geometry = volume_geometry
        self.projection_geometry = projection_geometry
        self.projector = astra.create_projector(
            'cuda3d', projection_geometry, volume_geometry)
        self.projection_shape = tuple(astra.geom_size(projection_geometry))
        self.volume_shape = tuple(astra.geom_size(volume_geometry))
        self.domain_shape = self.volume_shape
        self.range_shape = self.projection_shape
        if 'ProjectionAngles' in projection_geometry:
            self.num_angles = len(projection_geometry['ProjectionAngles'])
        else:
            self.num_angles = len(projection_geometry['Vectors'])

    def forward(self, x):
        return _ASTRAFunction.apply(
            x, self.projector, self.projection_shape, self.volume_shape)

    def adjoint(self, y):
        if y.ndim == 4:
            B = y.shape[0]
            vol = torch.zeros((B, *self.volume_shape), dtype=torch.float32,
                              device='cuda')
            for i in range(B):
                astra.experimental.direct_BP3D(
                    self.projector, vol=vol[i],
                    proj=y[i].contiguous().detach())
        elif y.ndim == 3:
            vol = torch.zeros(self.volume_shape, dtype=torch.float32,
                              device='cuda')
            astra.experimental.direct_BP3D(
                self.projector, vol=vol, proj=y.detach())
        else:
            raise ValueError(f"Unsupported y.ndim={y.ndim}")
        return vol


    def close(self):
        if self.projector is not None:
            astra.projector.delete(self.projector)
            self.projector = None


class _ASTRA2DFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, operator, adjoint):
        ctx.operator, ctx.adjoint = operator, adjoint
        return operator._apply(value, adjoint)

    @staticmethod
    def backward(ctx, gradient):
        return _ASTRA2DFunction.apply(gradient, ctx.operator, not ctx.adjoint), None, None


class ASTRAOperator2D(LinearOperator):
    """Parallel-beam ASTRA projector with per-call data ownership."""

    nonnegative_matrix = True

    def __init__(self, image_size, angles, device="cpu", in_channels=1):
        _require_astra()
        self.angles = torch.as_tensor(angles).cpu()
        self.device = device
        self.subsets = {}
        self.domain_shape = (in_channels, image_size, image_size)
        self.range_shape = (in_channels, len(angles), image_size)
        self.num_angles = len(angles)
        volume = astra.create_vol_geom(image_size, image_size)
        projection = astra.create_proj_geom("parallel", 1.0, image_size,
                                            torch.as_tensor(angles).cpu().numpy())
        self.projector = astra.create_projector("linear", projection, volume)
        self.matrix = None
        if torch.device(device).type == "cuda":
            # ASTRA's native CUDA FP/BP discretizations are not a transpose pair.
            # Export the linear projector once and apply its exact transpose on GPU.
            matrix_id = None
            try:
                matrix_id = astra.projector.matrix(self.projector)
                matrix = astra.matrix.get(matrix_id).tocoo()
                import numpy as np
                indices = torch.from_numpy(np.stack((matrix.row, matrix.col))).long()
                values = torch.from_numpy(matrix.data).float()
                self.matrix = torch.sparse_coo_tensor(indices, values, matrix.shape,
                                                       device=device).coalesce()
            except BaseException:
                self.close()
                raise
            finally:
                if matrix_id is not None:
                    astra.matrix.delete(matrix_id)

    def _apply(self, value, adjoint):
        shape = self.domain_shape if adjoint else self.range_shape
        if self.matrix is not None:
            matrix = self.matrix.transpose(0, 1) if adjoint else self.matrix
            flat = value.reshape(-1, value.shape[-2]*value.shape[-1])
            return torch.sparse.mm(matrix, flat.T).T.reshape(value.shape[0], *shape)
        operation = astra.create_backprojection if adjoint else astra.create_sino
        output = torch.empty((value.shape[0], *shape), device=value.device, dtype=value.dtype)
        for batch in range(value.shape[0]):
            for channel in range(value.shape[1]):
                data_id = None
                try:
                    data_id, array = operation(value[batch, channel].detach().float().cpu().numpy(), self.projector)
                    output[batch, channel] = torch.as_tensor(array, device=value.device, dtype=value.dtype)
                finally:
                    if data_id is not None:
                        astra.data2d.delete(data_id)
        return output

    def forward(self, x):
        return _ASTRA2DFunction.apply(x, self, False)

    def adjoint(self, y):
        return _ASTRA2DFunction.apply(y, self, True)

    def pseudo_inverse(self, y, **kwargs):
        from inv_framework.solvers.analytic import fbp
        return fbp(self, y)

    def subset(self, indices):
        key = tuple(indices.cpu().tolist())
        if key not in self.subsets:
            self.subsets[key] = ASTRAOperator2D(self.domain_shape[-1], self.angles[list(key)],
                                               self.device, self.domain_shape[0])
        return self.subsets[key]

    def close(self):
        for child in self.subsets.values():
            child.close()
        self.subsets.clear()
        self.matrix = None
        if self.projector is not None:
            astra.projector.delete(self.projector)
            self.projector = None

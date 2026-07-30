import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inv_framework.operators.ct import ASTRAFDKOperator3D, LEAPOperator3D
from inv_framework.operators.ct import astra_adapter as astra_backend
from inv_framework.operators.ct import leap_adapter
from inv_framework.solvers.classical import FDKSolver


def _volume_geometry(size=2, voxel_sizes=(1.0, 1.0, 1.0)):
    x_size, y_size, z_size = voxel_sizes
    return {
        "GridColCount": size,
        "GridRowCount": size,
        "GridSliceCount": size,
        "option": {
            "WindowMinX": -0.5 * size * x_size,
            "WindowMaxX": 0.5 * size * x_size,
            "WindowMinY": -0.5 * size * y_size,
            "WindowMaxY": 0.5 * size * y_size,
            "WindowMinZ": -0.5 * size * z_size,
            "WindowMaxZ": 0.5 * size * z_size,
        },
    }


def _cone_geometry(size=2, num_angles=3, geometry_type="cone"):
    geometry = {
        "type": geometry_type,
        "DetectorRowCount": size,
        "DetectorColCount": size,
    }
    if geometry_type == "cone":
        geometry["ProjectionAngles"] = np.linspace(0.0, np.pi, num_angles).tolist()
    else:
        geometry["Vectors"] = np.zeros((num_angles, 12), dtype=np.float32)
    return geometry


class _FakeData3D:
    class GPULink:
        def __init__(self, ptr, x, y, z, pitch):
            self.ptr = ptr
            self.shape = (z, y, x)
            self.pitch = pitch

    def __init__(self, backend):
        self.backend = backend
        self.objects = {}
        self.deleted = []
        self.next_id = 1

    def create(self, datatype, geometry, data=None):
        object_id = self.next_id
        self.next_id += 1
        shape = self.backend.geom_size(geometry)
        if np.isscalar(data) or data is None:
            array = np.full(shape, 0.0 if data is None else data, dtype=np.float32)
        else:
            array = np.array(data, dtype=np.float32, copy=True)
        self.objects[object_id] = array
        return object_id

    def link(self, datatype, geometry, data):
        raise AssertionError("CPU adapter test must not use ASTRA GPU linking")

    def get(self, object_id):
        return self.objects[object_id].copy()

    def delete(self, object_id):
        self.deleted.append(object_id)
        self.objects.pop(object_id, None)


class _FakeAlgorithm:
    def __init__(self, backend):
        self.backend = backend
        self.configs = []
        self.deleted = []

    def create(self, config):
        self.configs.append(config)
        return 100 + len(self.configs)

    def run(self, algorithm_id):
        config = self.configs[algorithm_id - 101]
        if self.backend.fail_run:
            raise RuntimeError("synthetic ASTRA failure")
        projection = self.backend.data3d.objects[config["ProjectionDataId"]]
        reconstruction_id = config["ReconstructionDataId"]
        self.backend.data3d.objects[reconstruction_id].fill(float(projection.mean()))

    def delete(self, algorithm_id):
        self.deleted.append(algorithm_id)


class _FakeASTRA:
    def __init__(self):
        self.fail_run = False
        self.data3d = _FakeData3D(self)
        self.algorithm = _FakeAlgorithm(self)

    @staticmethod
    def create_projector(kind, projection_geometry, volume_geometry):
        assert kind == "cuda3d"
        return 77

    @staticmethod
    def geom_size(geometry):
        if "GridSliceCount" in geometry:
            return (
                geometry["GridSliceCount"],
                geometry["GridRowCount"],
                geometry["GridColCount"],
            )
        angles = geometry.get("ProjectionAngles", geometry.get("Vectors"))
        return (
            geometry["DetectorRowCount"],
            len(angles),
            geometry["DetectorColCount"],
        )

    @staticmethod
    def astra_dict(algorithm_type):
        assert algorithm_type == "FDK_CUDA"
        return {"type": algorithm_type}


@pytest.fixture
def fake_astra(monkeypatch):
    backend = _FakeASTRA()
    monkeypatch.setattr(astra_backend, "astra", backend, raising=False)
    monkeypatch.setattr(astra_backend, "_HAS_ASTRA", True)
    return backend


def test_astra_fdk_rejects_unsupported_geometry_before_backend_creation():
    with pytest.raises(ValueError, match="regular ASTRA 'cone'"):
        ASTRAFDKOperator3D(_volume_geometry(), _cone_geometry(geometry_type="cone_vec"))
    with pytest.raises(ValueError, match="cubic voxels"):
        ASTRAFDKOperator3D(
            _volume_geometry(voxel_sizes=(1.0, 1.0, 2.0)), _cone_geometry()
        )


def test_astra_fdk_solver_options_batch_dtype_and_cleanup(fake_astra):
    operator = ASTRAFDKOperator3D(_volume_geometry(), _cone_geometry())
    measurements = torch.stack(
        [torch.ones(operator.range_shape), torch.full(operator.range_shape, 2.0)]
    ).to(torch.float64)

    reconstruction = FDKSolver(
        filter="ramp", short_scan=True, voxel_supersampling=2
    ).solve(measurements, operator)

    assert reconstruction.shape == (2, *operator.domain_shape)
    assert reconstruction.dtype == torch.float64
    assert torch.allclose(reconstruction[0], torch.ones_like(reconstruction[0]))
    assert torch.allclose(reconstruction[1], torch.full_like(reconstruction[1], 2.0))
    assert all(
        config["option"]
        == {"ShortScan": True, "VoxelSuperSampling": 2}
        for config in fake_astra.algorithm.configs
    )
    assert len(fake_astra.algorithm.deleted) == 2
    assert len(fake_astra.data3d.deleted) == 4
    assert not fake_astra.data3d.objects


def test_astra_fdk_filter_alias_validation_and_failure_cleanup(fake_astra):
    operator = ASTRAFDKOperator3D(_volume_geometry(), _cone_geometry())
    measurements = torch.ones(1, *operator.range_shape)
    with pytest.raises(ValueError, match="different ASTRA filters"):
        operator.fdk(measurements, filter_type="hann", filter="cosine")
    with pytest.raises(NotImplementedError, match="would be ignored"):
        operator.fdk(measurements, filter_type="hann")
    with pytest.raises(TypeError, match="positive integer"):
        operator.fdk(measurements, voxel_supersampling=True)

    fake_astra.fail_run = True
    with pytest.raises(RuntimeError, match="synthetic ASTRA failure"):
        operator.fdk(measurements, filter_type="ramp")
    assert "FilterType" not in fake_astra.algorithm.configs[-1]["option"]
    assert len(fake_astra.algorithm.deleted) == 1
    assert len(fake_astra.data3d.deleted) == 2
    assert not fake_astra.data3d.objects


class _FakeLEAPModel:
    def __init__(self):
        self.project_calls = 0
        self.backproject_calls = 0
        self.fbp_calls = 0

    def get_numZ(self):
        return 2

    def get_numY(self):
        return 2

    def get_numX(self):
        return 3

    def get_numAngles(self):
        return 3

    def get_numRows(self):
        return 2

    def get_numCols(self):
        return 2

    def get_geometry(self):
        return "CONE"

    def project(self, g, f):
        self.project_calls += 1
        assert g.shape == (3, 2, 2)
        assert f.shape == (2, 2, 3)
        g.copy_(2.0 * f.reshape_as(g))
        return True

    def backproject(self, g, f):
        self.backproject_calls += 1
        assert g.shape == (3, 2, 2)
        assert f.shape == (2, 2, 3)
        f.copy_(2.0 * g.reshape_as(f))
        return True

    def FBP(self, g, f, inplace=False):
        self.fbp_calls += 1
        assert inplace is False
        f.copy_(g.reshape_as(f))
        return True


def test_leap_adapter_batch_dtype_forward_adjoint_and_fdk():
    model = _FakeLEAPModel()
    operator = LEAPOperator3D(model)
    volume = torch.arange(24, dtype=torch.float64).reshape(2, *operator.domain_shape)

    projection = operator.forward(volume)
    backprojection = operator.adjoint(projection)
    reconstruction = FDKSolver().solve(projection, operator)

    assert projection.shape == (2, *operator.range_shape)
    assert backprojection.shape == volume.shape
    assert reconstruction.shape == volume.shape
    assert projection.dtype == backprojection.dtype == reconstruction.dtype == torch.float64
    assert torch.allclose(projection.reshape_as(volume), 2.0 * volume)
    assert torch.allclose(backprojection, 4.0 * volume)
    assert torch.allclose(reconstruction, projection.reshape_as(volume))
    assert (model.project_calls, model.backproject_calls, model.fbp_calls) == (2, 2, 2)


def test_leap_forward_autograd_uses_backprojection():
    model = _FakeLEAPModel()
    operator = LEAPOperator3D(model)
    volume = torch.randn(2, *operator.domain_shape, dtype=torch.float64, requires_grad=True)
    weights = torch.randn(2, *operator.range_shape, dtype=torch.float64)

    loss = (operator.forward(volume) * weights).sum()
    loss.backward()

    assert torch.allclose(volume.grad, 2.0 * weights.reshape_as(volume))
    assert model.project_calls == 2
    assert model.backproject_calls == 2


def test_leap_dependency_and_geometry_errors_are_explicit(monkeypatch):
    def missing_model():
        raise ImportError("synthetic missing LEAP")

    monkeypatch.setattr(leap_adapter, "_new_leap_model", missing_model)
    with pytest.raises(ImportError, match="synthetic missing LEAP"):
        LEAPOperator3D()

    model = _FakeLEAPModel()
    model.get_numAngles = lambda: 0
    with pytest.raises(ValueError, match="geometry is not configured"):
        LEAPOperator3D(model)

    model = _FakeLEAPModel()
    model.get_geometry = lambda: "PARALLEL"
    operator = LEAPOperator3D(model)
    with pytest.raises(NotImplementedError, match="requires cone geometry"):
        operator.fdk(torch.ones(1, *operator.range_shape))

    operator = LEAPOperator3D(_FakeLEAPModel())
    with pytest.raises(TypeError, match="accepts no adapter options"):
        operator.fdk(torch.ones(1, *operator.range_shape), filter_type="hann")


def test_real_astra_cuda_fdk_reconstructs_small_cone_volume():
    if not astra_backend._HAS_ASTRA:
        pytest.skip("astra-toolbox is not installed")
    astra = astra_backend.astra
    if not astra.use_cuda() or not torch.cuda.is_available():
        pytest.skip("ASTRA CUDA and PyTorch CUDA are required")

    size = 16
    angles = np.linspace(0.0, 2.0 * np.pi, 60, endpoint=False)
    volume_geometry = astra.create_vol_geom(size, size, size)
    projection_geometry = astra.create_proj_geom(
        "cone", 1.0, 1.0, 32, 32, angles, 100.0, 100.0
    )
    operator = ASTRAFDKOperator3D(volume_geometry, projection_geometry)
    coordinates = torch.arange(size, device="cuda", dtype=torch.float32)
    zz, yy, xx = torch.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    center = 0.5 * (size - 1)
    phantom = (((xx - center) ** 2 + (yy - center) ** 2 + (zz - center) ** 2) <= 16.0)
    phantom = phantom.to(torch.float32).unsqueeze(0).requires_grad_(True)

    projection = operator.forward(phantom)
    reconstruction = operator.fdk(projection)
    weights = torch.ones_like(projection) / projection.numel()
    (projection * weights).sum().backward()
    explicit_adjoint = operator.adjoint(weights)

    assert reconstruction.shape == phantom.shape
    assert torch.isfinite(reconstruction).all()
    assert torch.allclose(phantom.grad, explicit_adjoint)
    inside = reconstruction[phantom.bool()].mean()
    outside = reconstruction[:, :2, :2, :2].abs().mean()
    assert inside > outside + 1e-3

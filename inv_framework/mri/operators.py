"""Centered orthonormal Fourier encoding; maps retain their supplied scale."""
import numpy as np


def fft(x, dimensions, inverse=False):
    axes = tuple(range(x.ndim - dimensions, x.ndim))
    transform = np.fft.ifftn if inverse else np.fft.fftn
    return np.fft.fftshift(transform(np.fft.ifftshift(x, axes=axes), axes=axes,
                                    norm="ortho"), axes=axes)


class SenseEncoding:
    def __init__(self, maps, mask=None, trajectory=None):
        self.maps, self.mask = maps, mask
        self.shape = maps.shape[2:]
        self.trajectory = trajectory
        if trajectory is not None:
            import torch
            import torchkbnufft as tkbn
            self.omega = torch.as_tensor(trajectory.T.copy(), dtype=torch.float64)
            self.nufft = tkbn.KbNufft(self.shape, dtype=torch.float64)
            self.nufft_adjoint = tkbn.KbNufftAdjoint(self.shape, dtype=torch.float64)
            self.scale = np.sqrt(np.prod(self.shape))

    def fourier(self, x, inverse=False):
        if self.trajectory is None:
            return fft(x, len(self.shape), inverse)
        import torch
        transform = self.nufft_adjoint if inverse else self.nufft
        return transform(torch.as_tensor(np.ascontiguousarray(x), dtype=torch.complex128),
                         self.omega).numpy() / self.scale

    def forward(self, x):
        y = self.fourier(self.maps * x[:, None])
        return y if self.mask is None else self.mask * y

    def adjoint(self, y):
        if self.mask is not None:
            y = self.mask * y
        return np.sum(self.maps.conj() * self.fourier(y, inverse=True), axis=1)

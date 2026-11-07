"""Same-size convolution with explicit PSF index origin and matched transpose."""
import numpy as np


class Convolution:
    def __init__(self, shape, psf, origin, axes, boundary):
        self.shape, self.psf, self.origin = tuple(shape), psf, tuple(origin)
        self.axes, self.boundary = tuple(axes), boundary
        if boundary == "periodic":
            kernel = np.zeros([shape[a] for a in axes])
            for index in np.ndindex(psf.shape):
                target = tuple((i-o) % n for i, o, n in zip(index, origin, kernel.shape))
                kernel[target] += psf[index]
            otf = np.fft.fftn(kernel)
            expanded = [1] * len(shape)
            for a in axes:
                expanded[a] = shape[a]
            self.otf = otf.reshape(expanded)

    def _apply(self, x, adjoint):
        if self.boundary == "periodic":
            multiplier = self.otf.conj() if adjoint else self.otf
            return np.fft.ifftn(np.fft.fftn(x, axes=self.axes) * multiplier,
                                axes=self.axes).real
        result = np.zeros_like(x, dtype=float)
        for index in np.ndindex(self.psf.shape):
            source, target = [slice(None)] * x.ndim, [slice(None)] * x.ndim
            for a, i, o in zip(self.axes, index, self.origin):
                delta = i-o
                target[a] = slice(max(delta, 0), min(x.shape[a], x.shape[a]+delta))
                source[a] = slice(max(-delta, 0), min(x.shape[a], x.shape[a]-delta))
            if adjoint:
                source, target = target, source
            result[tuple(target)] += self.psf[index] * x[tuple(source)]
        return result

    def forward(self, x):
        return self._apply(x, False)

    def adjoint(self, y):
        return self._apply(y, True)

"""Small resident-array operations shared by NumPy and Torch numerical kernels."""
import numpy as np
import torch


def clone(x):
    return x.clone() if torch.is_tensor(x) else x.copy()


def zeros_like(x):
    return torch.zeros_like(x) if torch.is_tensor(x) else np.zeros_like(x)


def inner(x, y):
    return (x.conj()*y).sum()


def norm(x):
    return float(inner(x, x).real**0.5)


def shrink(x, threshold, axis=None):
    """Modulus soft threshold; grouped axes leave all other channels independent."""
    if torch.is_tensor(x):
        magnitude = x.abs() if axis is None else x.abs().square().sum(axis, keepdim=True).sqrt()
        return x*(1-threshold/magnitude.clamp_min(torch.finfo(x.real.dtype).tiny)).clamp_min(0)
    magnitude = abs(x) if axis is None else np.sqrt(np.sum(abs(x)**2, axis=axis, keepdims=True))
    return x*np.maximum(1-threshold/np.maximum(magnitude, np.finfo(x.real.dtype).tiny), 0)


class Differences:
    """Forward spatial differences, periodic or zero terminal (Neumann).

    Axes index the input array; component axis is 0. Other axes are independent,
    including channels. Complex inputs retain native phase and Hermitian adjoints.
    """
    def __init__(self, axes, spacing=None, boundary='neumann'):
        self.axes = tuple(axes)
        self.spacing = tuple(spacing) if spacing is not None else (1.,)*len(self.axes)
        if boundary not in ('periodic', 'neumann') or len(self.spacing) != len(self.axes) or any(h <= 0 for h in self.spacing):
            raise ValueError('differences require positive spacing and periodic/neumann boundary')
        self.boundary = boundary
        self.bound = 4*sum(h**-2 for h in self.spacing)

    def forward(self, x):
        components = []
        for a, h in zip(self.axes, self.spacing):
            d = (torch.roll(x, -1, a) if torch.is_tensor(x) else np.roll(x, -1, a))-x
            if self.boundary == 'neumann':
                last = [slice(None)]*x.ndim; last[a] = -1
                d[tuple(last)] = 0
            components.append(d/h)
        return torch.stack(components) if torch.is_tensor(x) else np.stack(components)

    def adjoint(self, z):
        out = zeros_like(z[0])
        for d, a, h in zip(z, self.axes, self.spacing):
            d = clone(d)
            if self.boundary == 'neumann':
                last = [slice(None)]*d.ndim; last[a] = -1
                d[tuple(last)] = 0
            out += ((torch.roll(d, 1, a) if torch.is_tensor(d) else np.roll(d, 1, a))-d)/h
        return out

class OrthogonalHaar:
    """Orthonormal dyadic Haar on selected axes; low-frequency block retained."""
    def __init__(self, shape, axes, levels):
        self.shape, self.axes, self.levels = tuple(shape), tuple(axes), levels
        if levels <= 0 or any(shape[a] % 2**levels for a in axes):
            raise ValueError('Haar spatial sizes must be divisible by 2**levels')

    def forward(self, x):
        out = x.clone() if torch.is_tensor(x) else x.copy()
        for level in range(self.levels):
            region = [slice(None)]*x.ndim
            for a in self.axes:
                region[a] = slice(0, self.shape[a]//2**level)
            block = out[tuple(region)].clone() if torch.is_tensor(x) else out[tuple(region)].copy()
            for a in self.axes:
                even, odd = [slice(None)]*x.ndim, [slice(None)]*x.ndim
                even[a], odd[a] = slice(0,None,2), slice(1,None,2)
                lo, hi = block[tuple(even)], block[tuple(odd)]
                parts = [(lo+hi)/2**.5, (lo-hi)/2**.5]
                block = torch.cat(parts,dim=a) if torch.is_tensor(x) else np.concatenate(parts,axis=a)
            out[tuple(region)] = block
        return out

    def adjoint(self, x):
        out = x.clone() if torch.is_tensor(x) else x.copy()
        for level in reversed(range(self.levels)):
            region = [slice(None)]*x.ndim
            for a in self.axes:
                region[a] = slice(0, self.shape[a]//2**level)
            block = out[tuple(region)].clone() if torch.is_tensor(x) else out[tuple(region)].copy()
            for a in reversed(self.axes):
                lo, hi = torch.chunk(block,2,dim=a) if torch.is_tensor(x) else np.split(block,2,axis=a)
                merged = torch.empty_like(block) if torch.is_tensor(x) else np.empty_like(block)
                even, odd = [slice(None)]*x.ndim, [slice(None)]*x.ndim
                even[a], odd[a] = slice(0,None,2), slice(1,None,2)
                merged[tuple(even)], merged[tuple(odd)] = (lo+hi)/2**.5, (lo-hi)/2**.5
                block = merged
            out[tuple(region)] = block
        return out

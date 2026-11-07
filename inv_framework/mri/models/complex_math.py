"""Centered orthonormal 2D FFT and real-pair complex arithmetic for VarNet."""
import torch


def _fft(x, inverse):
    z = torch.view_as_complex(x.contiguous())
    transform = torch.fft.ifft2 if inverse else torch.fft.fft2
    z = torch.fft.fftshift(transform(torch.fft.ifftshift(z, dim=(-2, -1)),
                                    norm='ortho'), dim=(-2, -1))
    return torch.view_as_real(z)


def fft2c(x):
    return _fft(x, False)


def ifft2c(x):
    return _fft(x, True)


def complex_mul(x, y):
    return torch.stack((x[..., 0]*y[..., 0]-x[..., 1]*y[..., 1],
                        x[..., 0]*y[..., 1]+x[..., 1]*y[..., 0]), -1)


def complex_conj(x):
    return torch.stack((x[..., 0], -x[..., 1]), -1)


def complex_abs(x):
    return (x.square().sum(-1)).sqrt()


def rss(x, dim):
    return x.square().sum(dim).sqrt()


def rss_complex(x, dim):
    return x.square().sum(-1).sum(dim).sqrt()


def batched_mask_center(x, start, end):
    columns = torch.arange(x.shape[-2], device=x.device)[None, :]
    mask = (columns >= start[:, None]) & (columns < end[:, None])
    return x * mask[:, None, None, :, None]

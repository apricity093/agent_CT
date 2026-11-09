"""Autograd wiring and dense-view FBP regression.

Independent discrete-transpose checks live in test_operator_references.py.
"""

import math
import os
import sys
import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inv_framework.ct.backends.radon_torch import ParallelBeamRadon2D
from inv_framework.solvers.analytic import fbp


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_double_stencil_matches_streaming_projection_and_transpose(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from inv_framework.ct.backends.radon_torch import _radon_forward_impl, _radon_adjoint_impl
    generator = torch.Generator().manual_seed(51)
    op = ParallelBeamRadon2D(8, num_angles=13, in_channels=2, device=device)
    x = torch.randn((2, 2, 8, 8), generator=generator, dtype=torch.float64).to(device).requires_grad_()
    y = torch.randn((2, 2, 13, 8), generator=generator, dtype=torch.float64).to(device)
    actual = op.forward(x)
    torch.testing.assert_close(actual, _radon_forward_impl(x, op.angles), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(op.adjoint(y), _radon_adjoint_impl(y, op.angles, 8), rtol=1e-12, atol=1e-12)
    left = (actual*y).sum(); right = (x*op.adjoint(y)).sum()
    assert float(abs(left-right)/abs(left)) < 1e-12
    gradient, = torch.autograd.grad(left, x)
    torch.testing.assert_close(gradient, op.adjoint(y), rtol=1e-12, atol=1e-12)


def test_autograd_matches_adjoint(image_size=32, num_angles=24, tol=1e-5,
                                  seed=1):
    torch.manual_seed(seed)
    op = ParallelBeamRadon2D(image_size=image_size,
                             num_angles=num_angles, device='cpu')
    x = torch.randn(1, 1, image_size, image_size, requires_grad=True)
    y = torch.randn(1, 1, num_angles, image_size)
    s = (op.forward(x) * y).sum()
    g_auto, = torch.autograd.grad(s, x)
    g_exact = op.adjoint(y)
    rel = (g_auto - g_exact).norm() / (g_exact.norm() + 1e-12)
    print(f"autograd vs adjoint rel_err = {rel.item():.3e} (tol {tol:.1e})")
    assert rel < tol, f"Autograd != adjoint: rel_err={rel.item():.3e}"
    print("PASS: backward(forward) == adjoint.")


def _disk_phantom(N: int, device='cpu') -> torch.Tensor:
    grid = torch.linspace(-1, 1, N, device=device)
    yy, xx = torch.meshgrid(grid, grid, indexing='ij')
    img = torch.zeros(N, N, device=device)
    img = torch.where(xx ** 2 + yy ** 2 <= 0.7 ** 2, img + 1.0, img)
    img = torch.where((xx - 0.2) ** 2 + yy ** 2 <= 0.15 ** 2,
                      img - 0.5, img)
    return img.clamp(0, 1)


def test_fbp_recovers_disk(image_size=128, num_angles=180, tol_psnr=20.0,
                           seed=0):
    torch.manual_seed(seed)
    x_true = _disk_phantom(image_size)[None, None]
    op = ParallelBeamRadon2D(image_size=image_size, num_angles=num_angles,
                             device='cpu')
    y = op.forward(x_true)
    x_rec = fbp(op, y).clamp(0, 1)
    mse = ((x_rec - x_true) ** 2).mean()
    psnr = 10.0 * math.log10(1.0 / mse.item())
    print(f"FBP PSNR on disk phantom: {psnr:.2f} dB (require >{tol_psnr})")
    assert psnr > tol_psnr, f"FBP failed: PSNR={psnr:.2f} dB"
    print("PASS: FBP recovers disk phantom.")


if __name__ == '__main__':
    test_autograd_matches_adjoint()
    test_fbp_recovers_disk()
    print("\nAll Radon operator tests passed.")

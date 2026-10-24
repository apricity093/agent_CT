"""Frozen pre-P0 forward and independent small dense references.

Forward copied from 011d48d's discretization; no production operator imports.
The optional double grid is only an isolated float64 reference.
"""

import torch
import torch.nn.functional as F


def rotation_grid(theta, shape, dtype=torch.float32):
    c, s = theta.cos(), theta.sin()
    z = torch.zeros_like(c)
    rotation = torch.stack((torch.stack((c, -s, z)), torch.stack((s, c, z))))
    rotation = rotation.to(dtype).unsqueeze(0).expand(shape[0], -1, -1)
    return F.affine_grid(rotation, list(shape), align_corners=False)


def frozen_forward(x, angles, dtype=torch.float32):
    return torch.stack([
        F.grid_sample(x, rotation_grid(theta, x.shape, dtype),
                      align_corners=False, padding_mode="zeros").sum(-2)
        for theta in angles
    ], dim=-2)


def explicit_matrix(size, angles, dtype=torch.float32):
    basis = torch.eye(size * size, device=angles.device, dtype=dtype)
    return frozen_forward(basis.reshape(-1, 1, size, size), angles, dtype).reshape(size * size, -1).T

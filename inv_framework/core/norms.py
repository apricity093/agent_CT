"""Per-solve squared norm estimates; estimates are never certified bounds."""
import time
import torch
from inv_framework.core.array_math import norm, inner


@torch.no_grad()
def power_norm(operator, reference, max_iterations=20, tolerance=1e-3):
    started = time.perf_counter()
    generator = torch.Generator(device=reference.device).manual_seed(0)
    x = torch.randn(reference.shape, dtype=reference.dtype, device=reference.device, generator=generator)
    x /= norm(x)
    previous = None
    for iteration in range(1, max_iterations+1):
        y = operator.adjoint(operator.forward(x))
        magnitude = norm(y)
        if magnitude == 0:
            value = 0.; break
        x = y/magnitude
        value = float(inner(x, operator.adjoint(operator.forward(x))).real)
        if previous is not None and abs(value-previous) <= tolerance*max(abs(value), 1e-30):
            break
        previous = value
    return dict(value=value, type='estimate', source='power_iteration_seed_0',
                iterations=iteration, initialization_sec=time.perf_counter()-started)

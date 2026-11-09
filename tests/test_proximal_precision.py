"""Precision-limited proximal certificates and their FISTA stop propagation."""
import json
from pathlib import Path
import torch
import pytest
from inv_framework.core.regularizer import TVRegularizer, joint_prox, ProximalAccuracyError
from inv_framework.core.operator import LinearOperator
from inv_framework.solvers import FISTASolver


class Identity(LinearOperator):
    domain_shape = (1, 1, 2)
    range_shape = domain_shape
    def forward(self, x):
        return x
    def adjoint(self, x):
        return x
    def squared_norm_upper_bound(self, x):
        return 1.


@pytest.mark.parametrize('mode', ['anisotropic', 'isotropic'])
def test_two_pixel_precision_and_attainable_target(mode, tmp_path):
    v = torch.tensor([[[[0., 1.]]]], dtype=torch.float32)
    reg = TVRegularizer(mode)
    attainable = joint_prox(reg, v, .2, tolerance=1e-5, iteration=1)
    # Reference is float64, so float32 endpoint rounding remains observable.
    exact = torch.tensor([[[[.2, .8]]]], dtype=torch.float64)
    actual_error = float((attainable.double()-exact).norm())
    assert attainable.dtype == v.dtype and attainable.device == v.device
    assert actual_error <= reg.last_prox_error_bound <= reg.last_prox_target
    attainable_bound = reg.last_prox_error_bound
    with pytest.raises(ProximalAccuracyError, match='^joint_prox_returned_precision$'):
        joint_prox(reg, v, .2, tolerance=1e-5, max_iterations=1000, iteration=50)
    assert actual_error > reg.last_prox_target
    assert reg.last_prox_error_bound > reg.last_prox_target
    assert reg.last_prox_iterations < 1000
    # A single unsuccessful iteration with a reachable target is a budget stop.
    harder = torch.tensor([[[[0., .2], [1., -.7]]]], dtype=torch.float64)
    with pytest.raises(ProximalAccuracyError, match='^joint_prox_inner_max_iterations$'):
        joint_prox(TVRegularizer(mode), harder, .2, tolerance=1e-8, max_iterations=1)
    values = dict(mode=mode, target=reg.last_prox_target, actual_error=actual_error,
                  bound=reg.last_prox_error_bound, rounding_bound=reg.last_prox_rounding_bound,
                  attainable_bound=attainable_bound, iterations=reg.last_prox_iterations,
                  reason='joint_prox_returned_precision')
    print(json.dumps(values))
    (tmp_path/'numerics.json').write_text(json.dumps(values, indent=2))


def test_fista_precision_stop_retains_completed_image():
    op = Identity()
    y = torch.tensor([[[[0., 1.]]]], dtype=torch.float32)
    initial = torch.zeros_like(y)
    reg = TVRegularizer('anisotropic', num_iterations=1000)
    # First target is attainable; k^-2.1 subsequently reaches the dtype floor.
    solver = FISTASolver(reg, strength=.2, step_size=1., num_iterations=100,
                         tolerance=0.)
    result = solver.solve_detailed(y, op, x_init=initial)
    one = FISTASolver(TVRegularizer('anisotropic', num_iterations=1000), strength=.2,
                      step_size=1., num_iterations=1, tolerance=0.).solve_detailed(y, op)
    assert result.status == 'stalled'
    assert result.stopping_reason == 'joint_prox_returned_precision'
    assert result.actual_iterations > 0 and result.status != 'converged'
    assert torch.equal(result.reconstruction, one.reconstruction)
    assert torch.isfinite(result.reconstruction).all()
    assert result.metadata['joint_prox_error_bound'] > result.metadata['joint_prox_target']
    print(json.dumps(dict(status=result.status, reason=result.stopping_reason,
                         iterations=result.actual_iterations,
                         reconstruction=result.reconstruction.flatten().tolist(),
                         target=reg.last_prox_target, bound=reg.last_prox_error_bound)))


def test_haar_zero_and_small_details_with_box():
    from inv_framework.core.regularizer import HaarL1Regularizer
    class ImageIdentity(Identity):
        domain_shape = (1, 2, 2)
        range_shape = domain_shape
    for perturbation in (0., 1e-6):
        y = torch.ones(1, 1, 2, 2, dtype=torch.float64)
        y[0, 0, 0, 0] += perturbation
        reg = HaarL1Regularizer(1)
        result = FISTASolver(reg, strength=.2, step_size=1., min_value=0.,
                             max_value=2., num_iterations=10).solve_detailed(y, ImageIdentity())
        expected = y.mean().expand_as(y)
        error = float((result.reconstruction-expected).norm())
        assert result.status == 'converged'
        assert result.actual_iterations > 0
        assert error < 1e-12
        assert error <= reg.last_prox_error_bound <= reg.last_prox_target
        print(json.dumps(dict(haar_perturbation=perturbation, error=error,
                             bound=reg.last_prox_error_bound, target=reg.last_prox_target,
                             status=result.status, iterations=result.actual_iterations)))


def _decimal_tv_certificate(snapshot):
    """Independent unit-spacing Neumann primal-dual bound, reference only.

    Decimal dual components are rounded to 80 digits, then treated as exact
    numbers. 256-digit evaluation and an outward-rounded square root bound
    the gap; a squared-norm check verifies every resulting dual is feasible.
    """
    from decimal import Decimal, localcontext, ROUND_CEILING
    x, v, p, weight = snapshot
    def image(a):
        return [[Decimal.from_float(float(q)) for q in row] for row in a.cpu().double()[0, 0].tolist()]
    xx, vv = image(x), image(v)
    pp = [image(p[:, k]) for k in range(2)]
    height, width = len(xx), len(xx[0])
    lam = Decimal.from_float(weight)
    with localcontext() as ctx:
        ctx.prec = 80
        # A strict radial margin dominates 80-digit normalization roundoff.
        margin = Decimal(1)-Decimal('1e-60')
        for i in range(height):
            for j in range(width):
                size = (pp[0][i][j]**2+pp[1][i][j]**2).sqrt()
                scale = min(Decimal(1), lam/size)*margin if size else Decimal(1)
                for k in range(2):
                    pp[k][i][j] *= scale
    with localcontext() as ctx:
        ctx.prec = 256
        ctx.rounding = ROUND_CEILING
        max_squared_excess = Decimal('-Infinity')
        gap = Decimal(0)
        for i in range(height):
            for j in range(width):
                a, b = pp[0][i][j], pp[1][i][j]
                excess = a*a+b*b-lam*lam
                max_squared_excess = max(max_squared_excess, excess)
                assert excess <= 0
                dz0 = xx[i+1][j]-xx[i][j] if i+1 < height else Decimal(0)
                dz1 = xx[i][j+1]-xx[i][j] if j+1 < width else Decimal(0)
                size_upper = ctx.next_plus((dz0*dz0+dz1*dz1).sqrt())
                defect = lam*size_upper-a*dz0-b*dz1
                adjoint = ((pp[0][i-1][j] if i else 0)
                           -(a if i+1 < height else 0)
                           +(pp[1][i][j-1] if j else 0)
                           -(b if j+1 < width else 0))
                residual = xx[i][j]-vv[i][j]+adjoint
                gap += defect+residual*residual/2
        distance = ctx.next_plus((2*gap).sqrt())
    return dict(distance_bound=float(distance), gap_upper=float(gap),
                max_dual_squared_excess=float(max_squared_excess), decimal_precision=256)


def test_public_ct_tv_candidate_and_completed_update(monkeypatch, tmp_path):
    from inv_framework.api import CTTaskRequest, run_ct_task
    import inv_framework.core.regularizer as module
    original = module._proximal_residual_certificate
    accepted = {}
    def capture(*args):
        result = original(*args)
        if not accepted and result[0] <= 1e-5:
            accepted.update(snapshot=tuple(args[k].detach().clone() for k in (2, 3, 4))+(args[6],),
                            bound=result[0])
        return result
    monkeypatch.setattr(module, '_proximal_residual_certificate', capture)
    result = run_ct_task(CTTaskRequest('parallel_2d__ellipses_sparse_gaussian', 'fista',
                                      backend='torch', regularizer='tv', params={'max_iter': 1}))
    assert result.status == 'success', result.error_message
    assert result.iterations == 1
    assert torch.as_tensor(result.reconstruction).norm() > 0
    assert accepted
    reference = _decimal_tv_certificate(accepted['snapshot'])
    assert reference['distance_bound'] <= accepted['bound'] <= 1e-5
    # Replay exactly the original 50-step budget, withholding acceptance only
    # in this evidence run, to assess the candidate rejected by the old rule.
    last = {}
    def withhold(*args):
        last['snapshot'] = tuple(args[k].detach().clone() for k in (2, 3, 4))+(args[6],)
        return float('inf'), 0.
    monkeypatch.setattr(module, '_proximal_residual_certificate', withhold)
    _, v, _, weight = accepted['snapshot']
    with pytest.raises(ProximalAccuracyError, match='joint_prox_inner_max_iterations'):
        joint_prox(TVRegularizer(), v, weight, tolerance=1e-5, max_iterations=50)
    rejected_reference = _decimal_tv_certificate(last['snapshot'])
    assert rejected_reference['distance_bound'] < 1e-5
    values = dict(target=1e-5, accepted_bound=accepted['bound'], accepted_reference=reference,
                  old_50_step_candidate=rejected_reference, iterations=result.iterations,
                  reconstruction_norm=float(torch.as_tensor(result.reconstruction).norm()),
                  reason=result.stopping_reason)
    print(json.dumps(values))
    (tmp_path/'ct_certificate.json').write_text(json.dumps(values, indent=2))

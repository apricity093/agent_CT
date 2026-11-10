"""Deterministic CT adjoints, automatic steps and catalog FDK checks."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
from inv_framework.api import CTTaskRequest, run_ct_task
from inv_framework.benchmarks import load_ct_case
from inv_framework.core.operator import LinearOperator, GradientOperator, HaarOperator
from inv_framework.core.regularizer import TikhonovRegularizer
from inv_framework.ct.backends.radon_torch import ParallelBeamRadon2D
from inv_framework.ct.backends.astra_adapter import ASTRAOperator2D
from inv_framework.solvers import FISTASolver, PDHGSolver
from inv_framework.utils.metrics import orthogonal_metrics
from tools.check_gpu_lifecycle import objects


class Diagonal(LinearOperator):
    domain_shape = (1, 2, 2)
    range_shape = (1, 2, 2)

    def __init__(self):
        self.diagonal = torch.tensor([1., 2., 3., 4.], dtype=torch.float64).reshape(1, 1, 2, 2)

    def forward(self, x):
        return self.diagonal * x

    def adjoint(self, y):
        return self.diagonal * y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(71)
    report = {'adjoints': {}, 'steps': {}}
    for device in ['cpu', 'cuda']:
        for name, op in [
            ('radon', ParallelBeamRadon2D(16, num_angles=12, device=device)),
            ('limited', ParallelBeamRadon2D(16, angles=torch.arange(12)*2./12, device=device)),
            ('astra2d', ASTRAOperator2D(16, torch.arange(12)*torch.pi/12, device=device)),
            ('tv', GradientOperator((1,16,16))), ('haar', HaarOperator((1,16,16), level=2)),
        ]:
            x = torch.randn(1,1,16,16, device=device)
            ax = op.forward(x)
            y = torch.randn_like(ax)
            left, right = (ax*y).sum(), (x*op.adjoint(y)).sum()
            error = float(abs(left-right)/abs(left))
            report['adjoints'][name+'_'+device] = dict(error=error, lhs=float(left), rhs=float(right), dtype=str(x.dtype))
            close = getattr(op, 'close', None)
            if close: close()
    op = Diagonal()
    y = torch.tensor([.4,.8,1.2,1.6],dtype=torch.float64).reshape(1,1,2,2)
    weight = .1
    exact = op.diagonal*y/(op.diagonal.square()+weight)
    for solver in [FISTASolver, PDHGSolver]:
        result = solver(regularizer=TikhonovRegularizer(), strength=weight,
                        num_iterations=2000, tolerance=1e-9).solve_detailed(y,op)
        error = float((result.reconstruction-exact).norm())
        report['steps'][solver.__name__] = dict(result=result.to_dict(), solution_error=error,
            finite=bool(torch.isfinite(result.reconstruction).all()))
        assert error < 1e-6, (solver.__name__, error)
        m = result.metadata
        if solver is PDHGSolver:
            assert m['tau']*m['sigma']*m['norm_bound_squared'] < 1
        else:
            assert 0 < m['step_size'] <= 1/(16+weight)
    case = load_ct_case('cone_3d/spheres_astra_12',device='cuda')
    from inv_framework.api.execution import build_operator
    cone = build_operator(case,'cuda','astra')
    x = torch.rand_like(case.truth)
    ax = cone.forward(x)
    y_cone = torch.rand_like(ax)
    left, right = (ax*y_cone).sum(), (x*cone.adjoint(y_cone)).sum()
    report['adjoints']['cone_cuda'] = dict(error=float(abs(left-right)/abs(left)),
        lhs=float(left),rhs=float(right),dtype=str(x.dtype),backend='matched cone ray integral CUDA')
    cone.close()
    baseline = objects()
    request = CTTaskRequest(case.case_id,'fdk',backend='astra',device='cuda',inputs=dict(
        observation=case.measurement.cpu().numpy(), geometry=case.geometry,
        measurement=case.metadata['measurement'], truth=case.truth.cpu().numpy(),
        data_range=case.metadata['ground_truth']['data_range']))
    before = request.inputs['observation'].copy()
    result = run_ct_task(request)
    torch.cuda.synchronize()
    record = asdict(result)
    x = record.pop('reconstruction')
    record.update(shape=list(x.shape),dtype=str(x.dtype),finite=bool(np.isfinite(x).all()),
        input_unchanged=bool(np.array_equal(before,request.inputs['observation'])),
        objects_released=objects()==baseline,
        slices=orthogonal_metrics(torch.as_tensor(x),case.truth.cpu(),request.inputs['data_range']))
    report['fdk'] = record
    np.save(args.output/'fdk.npy',x)
    (args.output/'mathematics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    assert result.status == 'success' and record['finite'] and record['input_unchanged'] and record['objects_released']
    assert x.shape == tuple(case.truth.shape) and x.dtype == np.float32
    assert all(v['error']<1e-5 for v in report['adjoints'].values())


if __name__ == '__main__':
    main()

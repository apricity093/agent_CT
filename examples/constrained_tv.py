"""Bounded P4 library example: synthetic observations never enter parameter selection."""
import argparse
import json
from pathlib import Path
from dataclasses import replace
import torch
from inv_framework.benchmarks.ct_cases import load_ct_case, write_ct_case
from inv_framework.ct_runtime import run_case
from inv_framework.operators.ct.radon_torch import ParallelBeamRadon2D


def make_case(root, limited=False):
    template = load_ct_case('parallel_2d/disk_analytic_32')
    angles = torch.arange(7)*(2. if limited else torch.pi)/7
    op = ParallelBeamRadon2D(8, angles=angles)
    phantom = torch.zeros(1,1,8,8)
    phantom[...,2:6,2:6] = .3
    phantom[...,3:5,3:5] = .5
    y = op.forward(phantom)
    mask = torch.ones_like(y, dtype=torch.bool)
    mask[...,0,0] = False
    case = replace(template, case_id='p4b_limited' if limited else 'p4b_sparse',
        truth=phantom, measurement=y, measurement_clean=y,
        roi_mask=None, valid_measurement_mask=mask,
        geometry={**template.geometry, 'domain_shape':list(op.domain_shape),
                  'range_shape':list(op.range_shape), 'angles_rad':angles.tolist()},
        metadata={**template.metadata, 'dimension':2, 'noise_model':'none'})
    record = write_ct_case(case, root/'case')
    record['path'] = 'case'
    (root/'catalog.json').write_text(json.dumps({'schema_version':'1.0','cases':[record]}), encoding='utf-8')
    return case, op


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limited', action='store_true')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    case, _ = make_case(args.out, args.limited)
    for name in ('constrained_pdhg','asd_pocs_tv'):
        result = run_case(name, case.case_id,
            Path(__file__).resolve().parents[1]/f'configs/algorithms/{name}.yaml',
            args.out/name, data_root=args.out, device=args.device,
            max_iterations=20, max_forward_calls=1500, max_adjoint_calls=1500)
        print(name, result['status'], result.get('convergence_status'), flush=True)

if __name__ == '__main__':
    main()

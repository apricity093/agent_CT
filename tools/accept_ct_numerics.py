"""Run the catalog CT numerical acceptance without regenerating data."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
from inv_framework.api import CTTaskRequest, run_ct_task
from inv_framework.benchmarks import load_ct_case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--only', nargs='*')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(71)
    tasks = [
        ('dense_fbp', 'dense_clean', 'fbp', None, None, {}),
        ('sparse_sart', 'sparse_gaussian', 'sart', None, None, {}),
        ('sparse_os_sart', 'sparse_gaussian', 'os_sart', None, None, {}),
        ('poisson_pdhg', 'sparse_poisson', 'pdhg', 'transmission_poisson_nll', 'tv', {'strength': .01, 'min_value': 0}),
        ('limited_pdhg', 'limited_angle', 'pdhg', 'l2', 'tv', {'strength': .01}),
    ]
    for label, suffix, algorithm, fidelity, regularizer, params in tasks:
        if args.only and label not in args.only:
            continue
        request = CTTaskRequest('parallel_2d__ellipses_' + suffix, algorithm,
                                device='cuda', fidelity=fidelity, regularizer=regularizer, params=params)
        result = run_ct_task(request)
        record = asdict(result)
        x = record.pop('reconstruction')
        if x is not None:
            np.save(args.output / (label + '.npy'), x)
            record.update(shape=list(x.shape), dtype=str(x.dtype), finite=bool(np.isfinite(x).all()))
        record['request'] = asdict(request)
        (args.output / (label + '.json')).write_text(json.dumps(record, indent=2), encoding='utf-8')
        print(label, result.status, result.iterations, result.stopping_reason, result.metrics, flush=True)
        assert result.status == 'success', result.error_message
        assert x.shape == tuple(load_ct_case(request.case_id).truth.shape) and np.isfinite(x).all(), x.shape


if __name__ == '__main__':
    main()

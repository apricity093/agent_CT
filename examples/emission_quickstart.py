"""Small nonnegative emission example for the MLEM/OSEM public CT entries."""
import argparse
import numpy as np
import torch
from inv_framework.ct.backends import ParallelBeamRadon2D
from inv_framework.runner import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--algorithm', choices=('mlem', 'osem'), default='mlem')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    angles = torch.arange(8)*torch.pi/8
    operator = ParallelBeamRadon2D(8, angles=angles)
    truth = torch.zeros(1, 1, 8, 8)
    truth[..., 2:6, 2:6] = 2.
    counts = np.random.default_rng(7).poisson(operator.forward(truth).numpy()).astype(np.float32)
    params = dict(num_iterations=2, initial_value=1.)
    if args.algorithm == 'osem':
        params['subset_count'] = 2
    result = run_experiment(dict(modality='ct', algorithm=args.algorithm,
        backend='torch', case_id='emission_toy', model={}, params=params,
        inputs=dict(observation=counts,
            geometry=dict(type='parallel_2d', domain_shape=[1,8,8], range_shape=[1,8,8], angles_rad=angles.tolist()),
            measurement=dict(observation_domain='nonnegative_counts', observation_model='poisson_emission', noise_model='poisson'))), args.output)
    print(result.status, result.stopping_reason, result.iterations)
    if result.status != 'success':
        raise RuntimeError(result.error)


if __name__ == '__main__':
    main()

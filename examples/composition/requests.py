"""Expand the fixed development candidates or select a quadratic method; never search."""
import argparse
from copy import deepcopy
from pathlib import Path
import os
import yaml

DIRECTORY = Path(__file__).resolve().parent
METHODS = ('admm', 'pdhg', 'fista', 'landweber', 'cgls', 'lsqr', 'cg')
DOMAINS = ('ct', 'mri', 'deconvolution')


def paths(value, convert):
    if isinstance(value, dict):
        return {k: convert(v) if k == 'path' else paths(v, convert) for k, v in value.items()}
    if isinstance(value, list):
        return [paths(v, convert) for v in value]
    return value


def load(name):
    return paths(yaml.safe_load((DIRECTORY / name).read_text(encoding='utf-8')),
                 lambda p: str((DIRECTORY / p).resolve()))


def candidates(domain):
    definition = yaml.safe_load((DIRECTORY / 'candidates.yaml').read_text())
    base = load(f'{domain}_tv_haar.yaml')
    refs = definition['references'][domain]
    for regularizer, reference in refs.items():
        for index, exponent in enumerate(definition['single_exponents']):
            config = deepcopy(base)
            config['params'].update(regularizer=regularizer, strength=reference * 10**exponent)
            if regularizer in ('huber', 'gradient_huber'):
                config['params']['boundary'] = 'neumann'
            yield f'{domain}_{regularizer}_{index}', config
    for q in definition['joint_q']:
        for a in definition['joint_a']:
            config = deepcopy(base)
            config['params'].update(strength=q*a*refs['tv'], haar_strength=q*(1-a)*refs['haar_l1'])
            yield f'{domain}_joint_q{q}_a{a}', config
    config = deepcopy(base)
    config['params'].update(regularizer='none', strength=0., haar_strength=0.)
    yield f'{domain}_unregularized', config


def quadratic(domain, method):
    config = load(f'{domain}_quadratic.yaml')
    config['algorithm'] = method
    # The common target lives in the base request; only ADMM needs these controls.
    if method == 'admm':
        config['params'].update(rho=1., rho_data=1., rho_regularizer=1.,
            rho_constraint=1., cg_max_iterations=100, cg_tolerance=1e-6)
    return config


def write(config, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    config = paths(config, lambda p: Path(os.path.relpath(p, destination.parent)).as_posix())
    with destination.open('x', encoding='utf-8') as stream:
        yaml.safe_dump(config, stream, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    expand = sub.add_parser('expand')
    expand.add_argument('--output', type=Path, required=True)
    choose = sub.add_parser('quadratic')
    choose.add_argument('--domain', choices=DOMAINS, required=True)
    choose.add_argument('--method', choices=METHODS, required=True)
    choose.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'expand':
        for domain in DOMAINS:
            for name, config in candidates(domain):
                write(config, args.output / 'development' / f'{name}.yaml')
    else:
        write(quadratic(args.domain, args.method), args.output)


if __name__ == '__main__':
    main()

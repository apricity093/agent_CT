"""Independent P5 pilot, frozen fixed-default batch and evaluation (CPU)."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity

from examples.ct_demo import shepp_logan
from inv_framework.ct_runtime import build_solver
from inv_framework.measurement import simulate_observation
from inv_framework.instrumentation import CountingLinearOperator, OperatorBudgetExceeded
from inv_framework.operators.ct.radon_torch import ParallelBeamRadon2D
from inv_framework.solvers.base import SolveControl


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


class TimedOperator(CountingLinearOperator):
    """Enforce wall and operator limits at the existing operator seam."""
    def __init__(self, operator, budget):
        super().__init__(operator, max_forward_calls=budget['forward']-1,
                         max_adjoint_calls=budget['adjoint'])
        self.deadline = time.perf_counter()+budget['wall_seconds']

    def _before(self, kind):
        if time.perf_counter() >= self.deadline:
            raise OperatorBudgetExceeded('wall budget exhausted', kind='wall', limit=600)
        super()._before(kind)


def phantom(pilot):
    if not pilot:
        return shepp_logan(32)[None, None]
    # Independent non-Shepp phantom, same unit-grid attenuation convention.
    grid = torch.linspace(-1, 1, 32)
    y, x = torch.meshgrid(grid, grid, indexing='ij')
    return (.2*((x/.7)**2+(y/.85)**2 < 1).float()
            +.3*(((x-.2)/.18)**2+((y+.1)/.25)**2 < 1).float())[None, None]


def make_case(case_id, pilot, settings):
    angles = torch.arange(48)*((2*torch.pi/3) if case_id == 'C3' else torch.pi)/48
    op = ParallelBeamRadon2D(32, angles=angles)
    truth = phantom(pilot)
    clean = op.forward(truth)
    seed = (29000 if pilot else 17)+int(case_id[1:])-2
    model = 'gaussian' if case_id in ('C2','C3') else 'poisson_gaussian' if case_id == 'C6' else 'poisson'
    observed, public, meta = simulate_observation(clean, model=model, seed=seed,
        sigma=settings['sigma'], I0=settings['I0'][case_id] if case_id in settings['I0'] else None,
        sigma_e=settings['sigma_e'] if case_id == 'C6' else 0., floor=1., scale=1.)
    case = SimpleNamespace(measurement=observed, public_measurement=public,
        valid_measurement_mask=torch.ones_like(observed, dtype=torch.bool),
        geometry={'domain_shape': [1,32,32]}, metadata={'measurement':meta})
    return case, op, truth, clean, angles


def noise_statistics(case, clean):
    public = case.public_measurement
    out = {'seed':case.metadata['measurement']['seed'],
           'projection_quantiles':torch.quantile(clean, torch.tensor([0.,.1,.5,.9,1.])).tolist()}
    if 'I0' in public:
        mu = public['I0']*torch.exp(-clean)
        raw = public.get('counts', public.get('c_obs'))
        electronic = public.get('sigma_e', torch.zeros_like(mu))**2
        out.update(expected_count_quantiles=torch.quantile(mu, torch.tensor([0.,.1,.5,.9,1.])).tolist(),
            zero_fraction=float((raw==0).float().mean()), negative_fraction=float((raw<0).float().mean()),
            clipped_fraction=float((raw<1).float().mean()), expected_zero_fraction=float(torch.exp(-mu).mean()),
            electronic_to_photon_variance_quantiles=torch.quantile(electronic/mu, torch.tensor([.1,.5,.9])).tolist(),
            standardized_readout_squared_error=float(((raw-mu)**2/(mu+electronic)).mean()))
    else:
        out.update(noise_mean=float((case.measurement-clean).mean()), noise_std=float((case.measurement-clean).std()))
    return out


def jobs(baselines=True):
    result = [dict(case=c, solver='pdhg', prior=p, fidelity='l2') for c in ('C2','C3')
              for p in ('tikhonov','tv','h1','huber','haar_l1')]
    result += [dict(case=c, solver='pdhg', prior='tv', fidelity=f) for c in ('C4','C5')
               for f in ('l2','wls','transmission_poisson_nll')]
    result += [dict(case='C6', solver='pdhg', prior='tv', fidelity=f) for f in ('l2','wls')]
    if baselines:
        result += [dict(case=c, solver=s, prior=None, fidelity='l2') for c in ('C2','C3') for s in ('fbp','cgls','sart')]
    return result


def parameters(job, manifest):
    solver = job['solver']
    if solver == 'fbp': return {}
    if solver == 'cgls': return dict(num_iterations=25, tol=0.)
    if solver == 'sart': return dict(num_iterations=5, block_size=1, order_strategy='ordered', relaxation=.1)
    result = dict(regularizer=job['prior'], reg_strength=1536*manifest['alpha_reg'],
        num_iterations=manifest['budget']['iterations'], tolerance=1e-5,
        delta=.01, level=2, tv_mode='isotropic', data_fidelity=job['fidelity'],
        min_value=0 if job['case'] in ('C4','C5','C6') else None)
    if job['fidelity']=='wls':
        result['weight_source'] = 'post_log_poisson_gaussian' if job['case']=='C6' else 'post_log_poisson'
    return result


def evaluate(x, truth, predicted, case):
    error = x-truth
    rmse = float(error.square().mean().sqrt())
    out = dict(rmse=rmse, psnr=20*np.log10(1/max(rmse,1e-12)),
        relative_error=float(error.norm()/truth.norm()),
        ssim=float(structural_similarity(truth[0,0].numpy(), x[0,0].numpy(), data_range=1.)),
        log_projection_rmse=float((predicted-case.measurement).square().mean().sqrt()),
        normalized_residual=float((predicted-case.measurement).norm()/case.measurement.norm()),
        minimum=float(x.min()), maximum=float(x.max()))
    public = case.public_measurement
    if 'I0' in public:
        mu = public['I0']*torch.exp(-predicted)
        raw = public.get('counts', public.get('c_obs'))
        out['raw_mean_prediction_mse'] = float((mu-raw).square().mean())
        if 'counts' in public:
            term = torch.where(raw>0, raw*torch.log(raw.clamp_min(1)/mu.clamp_min(1e-30)), torch.zeros_like(raw))
            out['mean_poisson_deviance'] = float((2*(term-raw+mu)).mean())
    return out


def run_job(root, job, case, op, truth, manifest):
    name = '_'.join(str(job[k]) for k in ('case','solver','prior','fidelity'))
    dest = root/name
    dest.mkdir()
    params = parameters(job, manifest)
    write(dest/'config.json', params)
    started = time.perf_counter()
    counted = TimedOperator(op, manifest['budget'])
    solver = build_solver(job['solver'], params, case)
    control = SolveControl(max_iterations=params.get('num_iterations',1), tolerance=1e-5,
        min_iterations=5, patience=5, stop_on_convergence=job['solver']=='pdhg',
        max_trajectory_points=2000)
    result = solver.solve_detailed(case.measurement, counted, control=control,
        **({'fidelity':solver.fidelity} if getattr(solver,'fidelity',None) is not None else {}))
    solve_seconds = time.perf_counter()-started
    x = result.reconstruction.detach()
    assert x.shape == truth.shape and x.dtype == torch.float32 and x.device.type == 'cpu'
    assert torch.isfinite(x).all()
    solve_cost = counted.stats()
    # One full projection reserved within the total budget for independent evaluation.
    counted.max_forward_calls = manifest['budget']['forward']
    predicted = counted.forward(x)
    metrics = evaluate(x, truth, predicted, case)
    out = dict(job=job, parameters=params, solve=result.to_dict(), metrics=metrics,
        solve_cost=solve_cost, total_cost=counted.stats(), solve_seconds=solve_seconds,
        evaluation_seconds=time.perf_counter()-started-solve_seconds, tuning_cost=0)
    write(dest/'result.json', out)
    torch.save({'reconstruction':x,'error':x-truth,'prediction':predicted}, dest/'arrays.pt')
    fig, axes = plt.subplots(1,3, figsize=(10,3))
    axes[0].imshow(truth[0,0],vmin=0,vmax=1,cmap='gray')
    axes[1].imshow(x[0,0],vmin=0,vmax=1,cmap='gray')
    axes[2].imshow((x-truth)[0,0],vmin=-1,vmax=1,cmap='coolwarm')
    for ax, title in zip(axes, ['truth','reconstruction','error [-1,1]']): ax.set_title(title); ax.axis('off')
    fig.tight_layout(); fig.savefig(dest/'images.png'); plt.close(fig)
    fig, axes = plt.subplots(1,2, figsize=(8,3))
    for ax, key in zip(axes, ['objective','algorithm_residual']):
        rows = [r for r in out['solve']['trajectory'] if r.get(key) is not None]
        ax.plot([r['iteration'] for r in rows], [r[key] for r in rows]); ax.set_title(key)
    fig.tight_layout(); fig.savefig(dest/'trajectory.png'); plt.close(fig)
    print(name, result.status, result.actual_iterations, metrics['psnr'], solve_seconds, flush=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['pilot','freeze','formal'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume', action='store_true', help='continue only unsaved formal jobs')
    args = parser.parse_args()
    root = args.out
    root.mkdir(parents=True, exist_ok=True)
    if args.phase == 'freeze':
        pilot = json.loads((root/'pilot/summary.json').read_text())
        assert len(pilot)==8 and all(r['solve']['status'] not in ('invalid_parameters','numerical_error') for r in pilot)
        manifest = json.loads((root/'pilot/proposal.json').read_text())
        manifest.update(jobs=jobs(), frozen=True, pilot_evidence='pilot/summary.json',
            budget_decision='Retain initial shared cap; finite endpoints, compare bounded fixed defaults; no optimal-prior ranking.',
            optional_constrained_runs=[], tuning=False, tuning_cost=0)
        stats=json.loads((root/'pilot/noise_statistics.json').read_text())
        assert stats['C5']['expected_count_quantiles'][2]<1 and stats['C5']['zero_fraction']>.1
        assert stats['C6']['negative_fraction']>0 and stats['C6']['electronic_to_photon_variance_quantiles'][1]>1
        manifest['frozen_configs']=[dict(job=j, parameters=parameters(j,manifest)) for j in manifest['jobs']]
        write(root/'manifest.json',manifest)
        return
    pilot = args.phase=='pilot'
    manifest = dict(settings=dict(sigma=.05, I0={'C4':10000.,'C5':20.,'C6':20.},sigma_e=3.),
        budget=dict(forward=300,adjoint=300,iterations=2000,wall_seconds=600),
        alpha_reg=.01/1536, data_range=1., roi='full 32x32', device='cpu', dtype='float32',
        initialization='zeros', spacing='unit-grid; projector sums pixels; attenuation unchanged',
        parameter_source='P2 split default reg_strength=.01 at m=1536; alpha=lambda/m. Huber=.01 and Haar level=2 library defaults. Fixed engineering defaults, no quality selection.',
        stopping='native PDHG fixed point <=1e-5, five consecutive; CGLS 25 and SART 5 fixed sweeps, FBP direct; shared hard caps include evaluation projection',
        scoring='full observation deviance for C4/C5; raw mean prediction MSE for C6; no training-objective cross-model ranking; no heldout selection or ranking',
        truth_policy='formal truth used only by generator and independent evaluation; solver and parameter selection never receive truth')
    if not pilot: manifest=json.loads((root/'manifest.json').read_text()); assert manifest['frozen']
    dest=root/args.phase
    dest.mkdir(exist_ok=args.resume and not pilot)
    write(dest/'proposal.json' if pilot else dest/'frozen_config.json',manifest)
    import inv_framework
    write(dest/'environment.json',dict(python=sys.executable, ct_import=inv_framework.__file__, torch=torch.__version__,
        numpy=np.__version__, environment={k:os.environ.get(k) for k in ('PYTHONPATH','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}))
    cases={}
    stats={}
    for c in ('C2','C3','C4','C5','C6'):
        case,op,truth,clean,angles=make_case(c,pilot,manifest['settings'])
        cases[c]=(case,op,truth)
        stats[c]=noise_statistics(case,clean)
        torch.save(dict(observation=case.measurement, public=case.public_measurement, truth=truth,
            clean=clean, angles=angles, mask=case.valid_measurement_mask, initialization=torch.zeros_like(truth)),dest/(c+'.pt'))
    write(dest/'noise_statistics.json',stats)
    work = jobs() if not pilot else jobs(False)[:5]+[jobs(False)[9], jobs(False)[15],jobs(False)[17]]
    results=json.loads((dest/'summary.json').read_text()) if args.resume else []
    for job in work:
        if any(r['job']==job for r in results): continue
        failed_dir=dest/'_'.join(str(job[k]) for k in ('case','solver','prior','fidelity'))
        if failed_dir.exists():
            failed_dir.rename(failed_dir.with_name(failed_dir.name+'_interrupted'))
        results.append(run_job(dest,job,*cases[job['case']],manifest))
        write(dest/'summary.json',results)


if __name__ == '__main__':
    main()

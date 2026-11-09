# Minimal public calls

From the algorithm-library root, use the existing numerical environment. No Agent SDK is imported. For other methods replace the algorithm ID and supply the required inputs/task type listed on its [algorithm page](algorithms/README.md); domain mathematics and existing calibration/blind/VarNet examples remain in docs/mri/algorithms.md and docs/deconvolution/algorithms.md.

```python
from inv_framework.api import CTTaskRequest, run_ct_task, list_capabilities
print([(r['name'], r['fidelity']) for r in list_capabilities()['algorithms']])
ct = run_ct_task(CTTaskRequest('parallel_2d__ellipses_sparse_gaussian',
    'fista', fidelity='gaussian_nll', regularizer='tikhonov',
    params={'max_iter': 2, 'lam': 0.01}))
print(ct.status, ct.effective_params, ct.converged)
# Same method, different implemented objective on the same observation:
l2 = run_ct_task(CTTaskRequest('parallel_2d__ellipses_sparse_gaussian',
    'fista', fidelity='l2', params={'max_iter': 2, 'lam': 0.01}))
```

```python
import numpy as np
from inv_framework.mri.api import MRITaskRequest, run_mri_task
inputs = dict(kspace=np.ones((1,1,4,4), complex),
    sampling_mask=np.ones((1,1,4,4)), sensitivity_maps=np.ones((1,1,4,4)))
model = dict(sampling='cartesian', spatial_shape=[4,4],
    axes=['batch','coil','y','x'], fft_norm='ortho', fft_centered=True)
mri = run_mri_task(MRITaskRequest('cg_sense', inputs, model,
    params={'lam': 0.1, 'max_iter': 2}, fidelity='l2'))
inputs['data_weights'] = np.arange(1,17).reshape(1,1,4,4)
weighted = run_mri_task(MRITaskRequest('cg_sense', inputs, model,
    params={'lam': 0.1, 'max_iter': 2}, fidelity='wls'))
# Gaussian NLL with these same weights is the same complex quadratic.
```

```python
from inv_framework.deconvolution.api import DeconvolutionTaskRequest, run_deconvolution_task
inputs = dict(observation=np.arange(16).reshape(4,4)/10, psf=np.ones((1,1)))
model = dict(axes=['y','x'], boundary='periodic', psf_origin=[0,0], noise_model='poisson')
poisson = run_deconvolution_task(DeconvolutionTaskRequest('tv_admm', inputs,
    model, params={'max_iter': 2}, fidelity='poisson_nll'))
quadratic = run_deconvolution_task(DeconvolutionTaskRequest('tv_admm', inputs,
    model, params={'max_iter': 2}, fidelity='gaussian_nll'))
rejected = run_deconvolution_task(DeconvolutionTaskRequest('richardson_lucy',
    inputs, model, fidelity='gaussian_nll'))
print(rejected.error) # fixed Poisson update rejects the switch
```

For file experiments save arrays to NPY and replace arrays by `{path: array.npy}`. Paths resolve relative to the experiment file. Supply modality, algorithm, inputs, model, params and optional fidelity. Execute `python -m inv_framework.runner --config task.yaml --output NEW_OUTPUT_DIR`, or `run_experiment('task.yaml', output_dir=NEW_OUTPUT_DIR, params={'max_iter': 1})`. Explicit params override file params before defaults are resolved. Use a new output directory to retain earlier experiments. Read status, stopping_reason, converged and effective_params separately; success does not assert convergence.

Shared ADMM/PDHG/FISTA three-domain calls, five regularizer definitions and fixed-block joint ADMM: [composable solvers](composable_solvers.md). A runnable small example is `python examples/composition_quickstart.py --domain mri --algorithm admm --regularizer tv_haar` (also ct/deconvolution).

See [quadratic baselines](quadratic_baselines.md) for all four linear methods and three domain task examples.

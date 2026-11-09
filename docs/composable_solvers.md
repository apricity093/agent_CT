# Shared ADMM, PDHG and FISTA

The public task APIs and `python -m inv_framework.runner` accept `admm`, `pdhg`
and `fista` for 2D parallel CT, native complex 2D Cartesian SENSE, and scalar
2D deconvolution with a known PSF. MRI and deconvolution use CPU arrays;
CT retains its Torch, ASTRA and LEAP operators and resource lifecycle.

Choose `params.regularizer`: `tv`, `haar_l1`, `tikhonov`, `h1`, or
`gradient_huber`. The older `huber` still means componentwise gradient Huber.
ADMM additionally accepts `tv_haar`, with `strength` for TV and `haar_strength`
for Haar. These are summed in one objective, with one x block and a joint
auxiliary block. Zero-weight branches are removed.

All penalties are unnormalized sums. Spatial axes are the last two, spacing is
one, and channels are independent. `boundary` is `neumann` (zero terminal
forward difference) or `periodic` for TV/H1. Huber currently requires Neumann.
`tv_mode` selects isotropic gradient-group modulus or anisotropic component
moduli. Complex moduli preserve phase. Tikhonov is 0.5 ||x||²; H1 is
0.5 ||Dx||². Gradient Huber sums h_delta(||Dx||₂), where h_delta(r) is
r²/(2 delta) below delta and r-delta/2 above it; `delta` defaults to .01.
Haar is orthonormal dyadic 2D analysis: `level` must divide both spatial sizes
by 2**level. `penalize_lowpass` defaults false for CT/deconvolution and true
for MRI. Low-frequency conventions are part of the objective.

L2 is 0.5 ||Ax-y||². Gaussian/WLS is 0.5 sum W |Ax-y|², with real,
nonnegative inverse-variance weights; no extra factor is inferred for complex
MRI. MRI/deconvolution use `inputs.data_weights`, required for WLS; Gaussian
without weights uses identity. CT retains public sigma/weights and
`weight_source`. Generation metadata does not override an explicit fidelity.
The new imaging quadratic path requires gain=1 and background=0. Existing CT
transmission-count and deconvolution Poisson methods retain their data prox
and dedicated paths; the new imaging combinations do not advertise Poisson.

ADMM `rho` defaults to 1. `rho_data`, `rho_regularizer`, `rho_haar`, and
`rho_constraint` independently override actual branches; each is positive and
fixed throughout a solve, unrelated to lambda. Thresholds use lambda/rho.
The identity split has an identity prox without bounds; it makes the system
positive definite without adding a ridge objective. Real bounds use
`min_value`/`max_value`; complex MRI rejects them. `subproblem_solver` is
auto, cg, fft, or cholesky. FFT requires periodic convolution and simultaneously
Fourier-diagonalizable splits (periodic D, orthogonal Haar, identity).
Otherwise auto uses warm-started Hermitian CG for the same system. Explicit
cholesky assembles and factors a small fixed system, including the CT example;
storage grows quadratically with the number of unknowns. See the
[verified starting points](../examples/composition/README.md) for precision,
resource limits and unchanged true-residual certification on all three paths.
Inner targets use cg_tolerance * max(1, ||b_1||) / k²; true residuals,
precision failure and budget exhaustion are returned with the last completed
outer iterate. No adaptive rho is implemented.

PDHG uses the actual stacked data/regularizer transform and conjugate prox,
with tau*sigma*(bound_A_squared+bound_K_squared)<1. Finite power iterations
are estimates, never certified upper bounds. FISTA uses checked smooth
majorization, including H1/Huber curvature; TV retains certified iterative
proximal stopping. Successful execution, iteration limit, inner precision,
and convergence are separate outcomes.

Results include effective parameters, objective decomposition, regularizer
definition, constraints, steps, fixed block rho, block residuals, weighted
adjoint dual residual, inner residual reports and stopping reason. For MRI
and deconvolution these are in diagnostics; CT uses
`diagnostics.convergence.metadata`. Defaults < file params < explicit runner
params. Algorithms do not depend on an Agent SDK or model service.

```python
from inv_framework.runner import run_experiment
result = run_experiment({
    "modality": "deconvolution", "algorithm": "admm", "fidelity": "wls",
    "model": {"axes": ["y", "x"], "boundary": "periodic", "psf_origin": [0, 0]},
    "inputs": {"observation": image, "psf": psf, "data_weights": weights},
    "params": {"regularizer": "tv_haar", "strength": .08, "haar_strength": .06,
               "level": 1, "boundary": "periodic", "rho_data": 1.1,
               "rho_regularizer": .7, "rho_haar": 1.3, "rho_constraint": .9}
})
```

Use `inv_framework.mri.api.run_mri_task(MRITaskRequest(...))`,
`inv_framework.deconvolution.api.run_deconvolution_task(DeconvolutionTaskRequest(...))`,
or `inv_framework.ct.api.run_ct_task(CTTaskRequest(...))` for in-memory tasks.
MRI inputs are complex kspace, sensitivity_maps and binary sampling_mask,
all (batch,coil,y,x); model declares axes, spatial_shape and cartesian sampling.
CT input geometry and measurement conventions remain documented in
[CT data format](ct/data_format.md). Each domain's list_capabilities() reads
the same actual records used for parameter resolution.

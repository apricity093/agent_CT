# Quadratic linear baselines

`landweber`, `cgls`, `lsqr` and `cg` are independently discoverable through
each domain's `list_capabilities()` and selectable in Python tasks and runner
configurations. All minimize the unnormalized sum

    0.5 sum Q abs(Ax-y)^2 + 0.5 strength ||Lx||^2.

Q is real nonnegative diagonal weighting, not a general noise covariance.
L2 uses identity Q; WLS requires public weights. CT Gaussian requires public
sigma. Complex MRI uses modulus squared and Hermitian adjoints, preserving
phase. Reconstruction assumptions do not change data generation metadata.
Poisson, TV, Haar-L1, Huber and constraints are not applicable to these updates.

L is identity (`tikhonov`) or forward spatial differences (`h1`) on the final
two axes, unit spacing, independent batches/channels. `boundary` selects zero
terminal differences (`neumann`) or `periodic`. H1 sums squared complex gradient
moduli. Arbitrary physical spacing is not exposed by this task interface.
`none` or zero strength omits the regularizer branch entirely.

Landweber, CGLS and LSQR share B = (sqrt(Q) A, sqrt(strength) L) and right hand
side (sqrt(Q) y, 0). CGLS performs forward/adjoint least-squares conjugate
directions. LSQR uses Golub-Kahan bidiagonalization with real plane rotations,
including for complex inputs. LSQR `damping=d` contributes identity strength
d squared, through augmentation. Simultaneously active damping and explicit
regularization is rejected. Initial estimates do not change the regularizer's
center.

Landweber uses step 1/U automatically, where U is an upper bound on the squared
augmented norm: max(Q) times the physical squared norm bound plus strength times
the squared norm bound of L. Explicit steps require 0 < step_size * U < 2;
invalid values are rejected without clipping. This conservative condition does
not treat a finite power estimate as a certified bound.

Independent CG applies Hx = A*QAx + strength L*Lx and b = A*Qy without building a
matrix. Its public entry requires positive identity Tikhonov, proving Hermitian
positive definiteness even under undersampling or zero data weights. H1 and
unregularized CG are rejected because public operator metadata does not certify
the common nullspace. No ridge is added. The numerical CG implementation is
shared with ADMM and MRI `cg_sense`. That dedicated NumPy entry retains `lam`;
CT `tikhonov` retains its specialized interface. These are not extra algorithms
in the adaptation count. Legacy bounded CT Landweber remains projected
unweighted L2 gradient; weighted or regularized bounded requests are rejected.

Common tolerance uses the explicitly recomputed true normal residual

    ||A*Q(Ax-y) + strength L*Lx|| / max(1, ||A*Qy||).

Outputs include raw and weighted data residual norms, objective and regularizer
definitions, effective parameters, iterations, status and stopping reason.
Budget exhaustion and precision limit do not imply convergence.

CG's numerical kernel certifies the normal system; the task monitor separately
checks the requested stopping policy. Under default stopping, an already solved
initial point converges with zero CG updates and a checkpoint at iteration zero.
If CG finishes numerically while explicit policy requirements remain unmet,
the task returns `stalled` with `cg_numerical_target_reached_policy_unmet`.
It retains the actual updates and checkpoints without manufacturing additional
observations to satisfy patience or iteration requirements. `cg_numerical`
diagnostics expose the kernel's residual certification separately. Cancellation
remains cancellation even when the numerical target is met.

When CT `tolerance` is absent, legacy CGLS `tol` or LSQR `atol + btol` selects this
normalized normal threshold; these are not SciPy backward-error criteria.
Set `tolerance` explicitly for comparisons. Generated algorithm pages separate
defaults, legal domains and recommended searches. Search strength logarithmically
at fixed Q/discretization; compare numerical accuracy before quality. Search
suggestions never clip legal parameters.

Defaults < experiment parameters < explicit runner parameters. Minimal calls
with prepared inputs from the domain data-format guides:

A runnable small example is `python examples/quadratic_quickstart.py --domain mri
--method lsqr`; choose any of the three domains and four methods with the same
options. The CT example uses the existing ellipse catalog case.

```python
from inv_framework.ct.api import CTTaskRequest, run_ct_task
from inv_framework.mri.api import MRITaskRequest, run_mri_task
from inv_framework.deconvolution.api import DeconvolutionTaskRequest, run_deconvolution_task
from inv_framework.runner import run_experiment

for method in ("landweber", "cgls", "lsqr", "cg"):
    p = dict(regularizer="tikhonov", strength=0.08,
             num_iterations=200, tolerance=1e-6)
    ct = run_ct_task(CTTaskRequest("parallel_2d__ellipses_dense_clean",
                                 method, backend="torch", params=p))
    mri = run_mri_task(MRITaskRequest(method, mri_inputs, mri_model, params=p))
    deconv = run_deconvolution_task(DeconvolutionTaskRequest(
        method, deconv_inputs, deconv_model, params=p))

# The same fields can be saved as a YAML/JSON runner configuration.
result = run_experiment(dict(modality="mri", algorithm="lsqr",
    inputs=mri_inputs, model=mri_model, fidelity="wls", params=p),
    params={"num_iterations": 300})
```

MRI requires native complex (batch, coil, y, x), centered orthonormal Cartesian
FFT, sensitivity maps and binary mask. Deconvolution requires real scalar 2D,
known PSF with explicit origin, periodic or zero convolution boundary, gain=1,
background=0. These CPU paths convert inputs to Torch float64/complex128 once;
CT retains its float32 backend/device. Iterations stay resident under no_grad;
output conversion occurs after solving. ASTRA lifetime retains CT finally cleanup.

# Constrained TV library usage

当前支持范围以[组件组合支持表](supported_combinations.md)为准；下文保留对应实现阶段的接口与数学说明。

The library exposes `ConstrainedTVSolver`, `constrained_pdhg`, and `asd_pocs_tv`
from `inv_framework.solvers`. The detailed API returns `SolveResult`;
convenience functions return the image. The registry/CLI IDs are
`constrained_pdhg` and `asd_pocs_tv`. Existing `pdhg` remains penalized.

The objective is `min TV(x)` subject to `||M(Ax-y)||_2 <= epsilon, x>=0`.
Supported input is one batch, one channel, square 2D parallel-beam line-integral
observations, including sparse and limited angles, with a binary nonempty mask.
TV uses unit-grid forward differences with zero terminal differences, isotropic
or anisotropic. No whitening, WLS/NLL constraint, upper image bound, alternate
prior or automatic radius estimation is supported. Counts must first be
converted to line integrals; epsilon must then be supplied in those units.

`epsilon` is finite and nonnegative and `epsilon_source` is a nonempty explicit
provenance string. The repository example's .2 radius is a numerical demonstration,
not a noise estimator or tuned value. `num_iterations` is a positive integer;
`tolerance` and `feasibility_tolerance` are positive finite values (defaults 1e-5).
The latter is an absolute residual-ball violation tolerance. Nonnegativity is
mandatory (`min_value=0`, `max_value=null`). PDHG derives a conservative joint
A/D step bound internally; no free step or penalty strength is accepted.
ASD accepts relaxation in (0,1], relaxation_decay and tv_step_decay in (0,1),
positive tv_step/update_ratio and positive integer tv_updates/block_size.
Defaults are 1/.99/.95/.2/.95/20/1 respectively. Parameters affect the existing
SART2D feasibility-backtracking variant, not an exact TIGRE reproduction.

```python
from inv_framework.solvers import ConstrainedTVSolver, SolveControl
solver = ConstrainedTVSolver(epsilon=0.2, epsilon_source="explicit observation radius",
                             algorithm="pdhg", tv_mode="isotropic", num_iterations=20)
r = solver.solve_detailed(y, operator, mask=mask,
                         control=SolveControl(stop_on_convergence=False))
print(r.status, r.stopping_reason, r.metadata['feasibility_status'])
```

Use `algorithm="asd_pocs_tv"` for the heuristic variant. `x_init` defaults to
zero; explicit initialization must match image shape, is copied to the observation
dtype/device, and is projected nonnegative. The solver never receives ground truth. The synthetic example below
uses a phantom only to generate observations and to evaluate the saved output.

Run from the main project root in PowerShell with the existing environment:

```powershell
$env:PYTHONPATH="$PWD/external/ct-benchlab;$PWD/src"
$env:PATH="D:/anaconda3/envs/inverse-agent-round1/Library/bin;D:/anaconda3/envs/inverse-agent-round1;"+$env:PATH
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'
D:/anaconda3/envs/inverse-agent-round1/python.exe external/ct-benchlab/examples/constrained_tv.py --out external/ct-benchlab/artifacts/constrained-tv-example-sparse
D:/anaconda3/envs/inverse-agent-round1/python.exe external/ct-benchlab/examples/constrained_tv.py --limited --device cuda --out external/ct-benchlab/artifacts/constrained-tv-example-limited-cuda
```

Each invocation creates a fresh case/catalog and runs both registered algorithms
with 20 iterations and 1500 forward/adjoint call caps. Choose a new output path for
each invocation. CUDA must actually be available; there is no CPU fallback.
For an existing catalog:

```powershell
D:/anaconda3/envs/inverse-agent-round1/python.exe -m inv_framework.cli run constrained_pdhg --case p4b_sparse --data-root external/ct-benchlab/artifacts/constrained-tv-example-sparse --config external/ct-benchlab/configs/algorithms/constrained_pdhg.yaml --out external/ct-benchlab/artifacts/constrained-tv-example-cli --max-forward-calls 1500 --max-adjoint-calls 1500
```

Outputs retain reconstruction.pt, metrics.json, diagnostics.json and manifest.json.
`diagnostics.convergence.endpoint_confirmation` contains independently recomputed
TV, raw residual, epsilon/source and feasible/violated status. A confirmed endpoint
means its reported quantities agree, not that it is feasible or optimal.
Native PDHG convergence requires feasibility plus the primal-dual fixed-point
criterion under SolveControl min/check/patience. ASD stagnation is `stalled`, with
no global optimality certificate. Budget exhaustion returns the last complete
outer image as partial_reconstruction.pt, preserves `resource_exhausted` and
clears unconfirmed terminal quantities. NaN/Inf propagate as numerical_error.
`status=success` is the existing process-level API; inspect convergence_status.

External stopping policies must disable discrepancy (`discrepancy.enabled=false`);
epsilon is the constraint radius. `--fixed-compute` disables early stopping.
Neither setting changes the native optimality test into a generic residual test.
Agent invocation and installation are described in [Agent usage](agent_usage.md).

## Validation and remaining precision limits

Bounded sparse/limited-angle CPU/CUDA checks cover shape, dtype/device, finite
nonnegative output, independent TV/residuals, invalid parameters, native stopping,
cancellation and interrupted endpoints. Their saved runtime checks are in
[the library evidence](../artifacts/ct_composable/p4/p4b-20261023/).
A confirmed endpoint can still violate epsilon; feasibility alone is not optimality.

The original [high-precision failure evidence](../artifacts/ct_composable/p4/acceptance-20261023/report.md)
and strict `tests/test_p4_ct_acceptance.py::test_nonconstant_ct` remain:

- Sparse CPU PDHG reached 20000 iterations with epsilon=.2, residual .2000011504,
  fixed-point residual 6.46e-5 and relative TV gap 1.88e-4, above the unchanged
  1e-5 optimization/objective thresholds. This is not a proof of infeasibility.
- Sparse CPU ASD was feasible but stalled after 320 sweeps, with TV gap about
  8.16%; the heuristic has no global optimality guarantee.
- Sparse CUDA SLSQP failed in its LSQ subproblem. Small constraint error does
  not make that reference successful.
- The limited-angle CPU reference succeeded but its production run was
  interrupted. Remaining long-run CPU/CUDA acceptance and CUDA costs are incomplete.

No reference-success, convergence or TV-gap assertion has been relaxed. Fixed-source
Agent delivery remains incomplete; P7/P8 have not started. Source, variant and
licensing details are in [Constrained TV sources](constrained_tv_sources.md).

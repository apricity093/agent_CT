# CT acceptance record

Date: 2026-11-02. Source: `D:/Pythonprojects/inverse_problem_agent_agentv1/external/ct-benchlab`.
Base commit: `44a66943eb8a4f05e2ab659c1328287cc5e0303f`; implementation includes uncommitted work.

## Handoff status

The CT public API, Agent migration and eight fixed-parameter demonstrations are complete.
All eight saved executions succeeded; seven 2D optimizations remain unconverged and
historical tissue FDK reports `completed_valid`. The user-authorized FDK device
validation and proximal stopping semantics are included in the current working tree.
This does not mark the default numerical regression or LEAP acceptance as passed.
See the [Agent handoff](../../../docs/ct_agent_api.md#源码与交接) for both source trees,
HEADs and uncommitted state, and [PLAN](../PLAN.md#11-执行交付状态) for completion status.

## Implemented structure and interface

- Removed `inv_framework/ct_runtime.py`; migrated internal callers without a compatibility module.
- Moved `ct/execution.py` to `api/execution.py`, `api/legacy.py` to `api/outputs.py`,
  and `convergence.py` to `solvers/convergence.py`. Dispatcher and CLI share one
  numerical execution pipeline; file output is separate.
- Added device, external catalog, fit/heldout indices, stopping policy, physical
  constraints and cancellation to public requests. Added detailed statuses,
  reason codes, history and partial reconstructions to results.
- Enabled actual missing-truth HDF5 input; added validation and orthogonal-slice
  helpers. SSIM now evaluates original values rather than display-clipped images.
- Removed gradient timing/CUDA synchronization, operator call counters, redundant
  solver preassembly, obsolete accounting budget arguments and FDK fairness records.
- Added parallel ASTRA and LEAP task bindings. ASTRA CUDA uses the exported linear
  matrix and its transpose because native CUDA FP/BP failed strict adjoint checking.
  LEAP binding is implemented but not exercised against an installed library.
- Preserved 17 registered algorithm entries and library DIP/INR/DPS. The existing
  core/CT/solver reorganization, four ellipses cases and single tests directory
  were retained; those earlier changes were not recreated.

## Executed checks

Interpreter: `D:/anaconda3/envs/inverse-agent-round1/python.exe`, PyTorch 2.5.1,
CUDA available, ASTRA 2.2.0. Small-workload runs used OMP/MKL thread counts of one.

| Check | Actual result |
| --- | --- |
| Parameter, public API and contract tests | 74 passed after dispatcher simplification |
| API/CLI/Krylov targeted regression | 40 passed |
| Stopping/convergence/FDK adapter tests | 40 passed; affected test after removal of obsolete accounting assertion: 1 passed |
| Physical bounds, missing truth, view isolation, orthogonal metrics, API evaluation | 31 passed |
| Real ASTRA cancellation/exception release | 2 passed; projector and cached subset projectors released |
| Required Radon/weighted/gradient/Haar inner products | 8 passed, CPU/CUDA; prescribed denominator, maximum error 7.08e-7 |
| ASTRA parallel strict adjoint | CPU 2.34e-7, CUDA 3.56e-8; both pass 1e-5; corresponding autograd checks also pass |
| ASTRA affected backend tests after matrix correction | 7 passed |
| ASTRA CUDA lifecycle | 50 short tasks after warmup; WDDM process dedicated GPU bytes at calls 0/10/20/30/40/50 were all 191508480 |
| Forbidden generated audit terms in inv_framework and tests | No source/data matches after renaming the metadata-shape test |
| Directory checks | No test/, specs.py, detailed.py or ct_runtime.py; four ellipses cases and spheres retained |
| Default regression | Not passed; interrupted after confirmed numerical failures described below |

Targeted counts overlap and are not summed. Interrupted or skipped checks are not
counted as passes. These are existing verification records, not reruns for this handoff.

### Representative short tasks

These are finite-output smoke checks, not the eight formal quality experiments.
All 2D tasks use the new ellipses data; FDK uses the protected spheres regression.

| Task | Iterations | State | PSNR | SSIM | RMSE |
| --- | ---: | --- | ---: | ---: | ---: |
| dense FBP | 0 | completed_valid | 30.0548 | 0.740862 | 0.000628477 |
| sparse Gaussian SART | 2 | max_iterations | 23.5795 | 0.433830 | 0.00132451 |
| sparse Gaussian OS-SART (4 subsets) | 2 | max_iterations | 16.3268 | 0.229261 | 0.00305275 |
| sparse Poisson PDHG + transmission NLL + TV | 2 | max_iterations | 20.3444 | 0.354759 | 0.00192225 |
| spheres FDK | 0 | completed_valid | 24.2288 | 0.862695 | 0.00122911 |

### Unresolved default-regression failures

`tests/test_constrained_ct_reference.py::test_nonconstant_ct[False-cpu]`:
constrained PDHG reached its existing 20000-iteration cap. Its normalized fixed
point residual was 6.46218e-5, above the required 1e-5; the independently evaluated
TV relative gap was 1.88140e-4, also above 1e-5. The measurement residual was
0.20000115 for radius 0.2. The test requires convergence and a smaller TV gap.
The status was correctly retained as `max_iterations`.

The sparse CUDA independent SLSQP reference returned `success=false`, message
`More than 3*n iterations in LSQ subproblem` after 392 iterations. Its feasibility
and objective values do not turn that unsuccessful reference run into a pass.
The remaining default suite was interrupted once these failures were confirmed;
its unexecuted portion remains unaccepted. No tolerance was relaxed and no
iteration budget or parameter search was added to conceal these failures.

The numerical/reference failures, remaining unexecuted default regression and LEAP
real-backend acceptance remain deferred. LEAP's dependency is absent.

## Fixed-parameter demonstration results

The historical input is present at the main project's
`artifacts/ct-agent-tissue-showcase-20261024/data/cone_fdk/arrays.h5`, with `case.json`
and the parent `data/catalog.json`. The case is `cone_3d/tissue_volume_clean_64`;
it was reused without regenerating data. The four ellipses cases remain registered,
and `cone_3d/spheres_astra_12` remains the separate protected regression case.

All eight tasks use float32 on the GTX 1650: Torch CUDA for 2D, ASTRA CUDA for FDK.
2D initialization is zero, with nonnegativity and `strength=0.01`. Non-L2 discrepancy
control is disabled; other native stopping controls remain active. ADMM has an outer
cap of 20, wall budget 900 s and timeout_s 1020 s; other 2D caps are 200, 300 s and
420 s. The Agent uses the minimum specified time limit. FDK has a 420 s deadline.
ADMM schedules public cancellation at 780 s; it completed in 391.08 s without cancellation.

| 实验 | 迭代 | 优化状态 | PSNR | SSIM | CT 秒 |
| --- | ---: | --- | ---: | ---: | ---: |
| gaussian_fista_tikhonov | 200 | `max_iterations` | 32.3192 | 0.886033 | 56.842 |
| gaussian_admm_tikhonov | 20 | `max_iterations` | 20.0575 | 0.467411 | 391.082 |
| gaussian_pdhg_tikhonov | 200 | `max_iterations` | 24.8367 | 0.770358 | 48.518 |
| gaussian_fista_tv | 66 | `stalled` | 36.0947 | 0.933266 | 80.442 |
| gaussian_fista_haar | 68 | `max_iterations` | 36.0679 | 0.933021 | 98.914 |
| poisson_pdhg_wls_tv | 200 | `max_iterations` | 32.0050 | 0.886940 | 53.644 |
| poisson_pdhg_nll_tv | 200 | `max_iterations` | 32.0049 | 0.886940 | 77.767 |
| cone_fdk | 0 | `completed_valid` | 30.6813 | 0.817536 | 27.414 |

TV stops with `joint_prox_returned_precision` (float32 returned-point precision),
Haar with `joint_prox_inner_max_iterations`; neither is convergence. Other 2D tasks
stop at their outer caps. ADMM retains visible artifacts. Different budgets and
fidelity objective scales do not support an equal-budget quality ranking.

Final [configuration](../../../outputs/ct_showcase/experiment.json),
[metrics and reconstruction paths](../../../outputs/ct_showcase/summary.json), and
[figures](../../../outputs/ct_showcase/README.md) are in the main project. FDK includes
three orthogonal center slices. Prior failure evidence remains in
[prior_attempts](../../../outputs/ct_showcase/prior_attempts/).
The existing five targeted checks cover FDK device validation, proximal stop reason
and returned-point propagation, and joint-prox certificate/budget behavior.

No remaining issue blocks this fixed-parameter demonstration handoff. Deferred
numerical and backend acceptance remains explicitly incomplete; the working-tree
results do not establish installation from a committed release.

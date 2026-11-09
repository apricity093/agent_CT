# Framework-independent experiments

See [domain boundaries](#domain-boundaries), [CT data](ct/data_format.md),
[MRI tasks and data](mri/tasks.md), and [deconvolution tasks and data](deconvolution/tasks.md).
Numerical code, configurations and tests install independently of Agent SDKs.

```python
from inv_framework.ct.api import CTTaskRequest, run_ct_task
from inv_framework.mri.api import MRITaskRequest, list_capabilities as mri_capabilities
from inv_framework.deconvolution.api import DeconvolutionTaskRequest
from inv_framework.runner import run_experiment
result = run_experiment("experiment.yaml", output_dir="run")
```

`inv-run --config experiment.yaml --output run` (or
`python -m inv_framework.runner`) invokes the same function once. Exit codes:
0 successful execution, 1 structured task failure, 2 invalid file/configuration.
No search, retries, history loading or implicit algorithm substitution occurs.
Legacy `invct` and `python -m inv_framework` remain CT commands.

Task keys: `modality`, `algorithm`, `task_type` (default reconstruction), `inputs`,
`model`, `params`, `device`; CT additionally accepts `backend`, `case_id`,
`data_root`, `stopping_policy`, `problem_constraints`. Catalog CT tasks omit
inputs/model and specify case_id; external CT tasks use inputs/model and need no
catalog. External case_id is only a label and defaults to `external` in the runner.
`evaluation: {reference: {path: truth.npy}, data_range: 1.0}` is optional for CT
external tasks. CT evaluation uses PSNR/SSIM/RMSE; missing truth omits them.
MRI/deconvolution evaluation runs only after successful reconstruction; see their domain documents.

Array input values are NumPy-compatible arrays or `{path: file.npy}` /
`{path: file.npz, key: array_name}`. Object arrays are disallowed. File-task paths
resolve relative to the task file; mapping/direct API paths use the working
directory. No implicit normalization, clipping or complex-to-real conversion is
performed by the shared loader. CT converts to float32; MRI preserves native
complex representation in its declared contract. Checkpoints must be supplied
explicitly; no downloads occur.

Precedence: explicit `run_experiment(..., params={...})` overrides task params,
which override the algorithm YAML defaults. CT public aliases and conflict rules
below still apply. CT defaults have one source in `configs/algorithms`; MRI/deconvolution descriptions live under `configs/<domain>/algorithms` and
are read by discovery directly. Executable defaults are read from these configuration files.

MRI and deconvolution request dataclasses contain `algorithm`, `inputs`, `model`,
`params`, `task_type`, `device`; they have no case/catalog requirement. Each
exports a domain TaskResult name for the shared `ExperimentResult` dataclass.
Discovery returns seven MRI and seven deconvolution executable records. VarNet requires a local matching checkpoint; blind methods require task_type=blind_deconvolution and model.kernel_shape.
Calling a planned method returns failed/not_implemented with empty outputs,
metrics and effective_params; an unknown ID returns invalid_parameters. Executable methods validate external inputs and mathematical parameters at the domain API boundary.

The runner returns `ExperimentResult`: `status`, `modality`, `algorithm`,
`effective_params`, `metrics`, `runtime_sec`, `iterations`, `converged`,
`stopping_reason`, `outputs`, `diagnostics`, `error` (code/message or null).
`effective_params` describes resolved solver parameters; it is empty when no
solver resolved them. CT diagnostics include actual device, dtype, convergence
trajectory and residuals. Execution success does not imply convergence:
`max_iterations` and `completed_valid` can be successful. Numerical divergence,
cancellation, resource exhaustion, unavailable backend and invalid parameters
are distinct CT reasons. Unexpected programming exceptions propagate.
The output directory contains `result.json` and one NPY per returned array.
Direct domain APIs return arrays in memory and do not write files.

Installation uses `python -m pip install .`; development checks use `.[dev]`.
Install `.[mri]` for non-Cartesian MRI (torchkbnufft 1.5.2). Classical Cartesian MRI and deconvolution use base dependencies.
VarNet additionally needs an explicit matching checkpoint; CUDA is optional for
VarNet, while the other current MRI methods and all deconvolution tasks use CPU.
FDK requires both PyTorch CUDA and ASTRA CUDA. ASTRA and LEAP install separately.
See [dependencies and installation](../DEPENDENCIES.md) for all three domains,
optional extras, backend setup and verification commands.

# CT public task API

`inv_framework.api` exports `CTTaskRequest`, `CTTaskResult`, `list_capabilities`,
`validate_ct_request`, `run_ct_task`, `orthogonal_slices`, and `orthogonal_metrics`.
The 17 algorithm IDs and their parameter descriptions come from one CT registry.
DIP, INR and DPS remain library solvers under `solvers/deep`, outside this task API.

## Discovery and validation

`list_capabilities(data_root=None)` returns algorithms, YAML defaults, parameter
rules, fidelity requirements, regularizers, backends and catalog records without
loading projection arrays. Each algorithm lists supported backends. Optional backend dependencies must be installed separately.

`validate_ct_request(request)` returns resolved mathematical parameters or raises
`ValueError` for invalid fields, incompatible geometry, data or combinations. It
uses the same registry validation as execution and does not run a solver or estimate
an operator norm. Device/backend dependencies are checked at execution.

## Request

| Field | Type and default | Meaning |
| --- | --- | --- |
| case_id | str, required | Catalog ID, or arbitrary label when inputs is supplied |
| algorithm | str, required | One of the 17 discovered IDs |
| backend | str, `torch` | `torch`, `astra`, `leap`; never silently substituted |
| device | str or None, None | `cpu`, `cuda`, `cuda:N`; default CPU for 2D, CUDA for cone FDK |
| data_root | str or None, None | Directory containing `catalog.json`; default `tests/data` |
| fidelity | str or None, None | Overrides the algorithm default |
| regularizer | str or None, None | Overrides the algorithm default; canonical Haar ID is `haar_l1`, no prior is `none` |
| params | dict, empty | Flat mathematical parameters merged over `configs/algorithms/<id>.yaml` |
| inputs | dict or None, None | External observation, geometry and measurement metadata; bypasses catalog |
| fit_view_indices | list[int] or None, None | Unique, nonempty, in-range view indices used for fitting |
| heldout_view_indices | list[int] or None, None | Supplied with fit indices, disjoint; evaluated only after fitting |
| stopping_policy | dict or None, None | Native stopping controls described below |
| problem_constraints | dict or None, None | `min_value`, `max_value`, `value_range=[lower,upper]`, `data_range` |
| callback | callable or None, None | Python callback receiving an IterationRecord; returning False cancels |

`max_iter`, `num_subsets`, `lam` map to `num_iterations`, `subset_count`, `strength`.
Conflicting spellings are rejected. Controls are separate from mathematical params.

Torch handles parallel 2D on CPU/CUDA. ASTRA handles parallel 2D on CPU/CUDA and
cone FDK on CUDA. Parallel CUDA applies the ASTRA linear sparse matrix and its
exact transpose with GPU sparse multiplication; its discretization is the same as
the ASTRA CPU linear projector. LEAP's parallel task binding uses a single-row parallel geometry;
its configuration follows [LEAP geometry](https://leapct.readthedocs.io/en/stable/ctgeometries.html)
and [volume](https://leapct.readthedocs.io/en/stable/ctvolume.html) definitions.
Missing optional dependencies return `unavailable`.

### Stopping and cancellation

Without a stopping policy, native solver tolerances and the iteration cap apply.
A policy may override `min_iterations=1`, `check_every=1`, `patience=5`,
`stop_on_convergence=true`, and the four tolerances (each default `1e-5`):
`relative_iterate_tolerance`, `normalized_normal_residual_tolerance`,
`relative_objective_tolerance`, `prox_gradient_mapping_tolerance`.
`stalled` and `divergence` accept `enabled`, `patience` and respectively
`relative_iterate_tolerance` / `relative_increase_tolerance`.
`discrepancy` selects the existing measurement-domain discrepancy rule; set
`{"enabled": false}` for non-L2 fidelity and constrained TV. Constrained TV uses
its mathematical epsilon. Optional policy identity is `schema_version="1.0"`,
`mode="solver_native_and_endpoint"`.

Cancellation occurs at solver checkpoints, after a completed iteration. Direct
backend calls cannot be interrupted mid-kernel. The Agent owns process hard
timeouts and total budgets. ASTRA resources are released by their owning finally
blocks on ordinary completion, cancellation and exceptions.

## Result

Required fields: `status` (`success`/`failed`), `case_id`, `algorithm_summary`,
`metrics: dict[str,float]`, `runtime_sec: float`, `iterations: int`,
`converged: bool`. Optional payloads are `reconstruction` (NumPy array or None)
and `error_message` (str or None).

`convergence_status`, `stopping_reason`, `reason_code`, and `history` carry native
solver evidence. History is a list of scalar iteration records, without retained
images or projection-call accounting. Termination states are `converged`,
`stalled`, `diverged`, `max_iterations`, `numerical_error`, `invalid_parameters`,
`completed_valid`, `unavailable`, `cancelled`, `resource_exhausted`.
An iteration cap is not convergence. Direct FBP/FDK use `completed_valid` with zero
iterative steps. Failed solver results preserve actual iterations and available
partial reconstruction. Failure before a solver returns has no partial image.
Unexpected implementation exceptions propagate; ordinary request/data errors,
missing backend, numerical failures and CUDA OOM have structured results.

2D arrays have shape `(B,C,H,W)`; 3D arrays `(B,D,H,W)`, float32. Reconstruction
values are not display-normalized. New ellipses use data range 0.02. Image metrics
use original values and the case's declared data range (or explicit constraint).
3D PSNR/RMSE cover the volume; SSIM averages axial 2D slices. `orthogonal_slices`
returns three central planes and `orthogonal_metrics(pred, truth, data_range)`
evaluates them with the common range.

Public HDF5 cases may omit `truth/x` and `measurement/y_clean`. Public measurement
arrays, geometry, noise metadata and masks remain required as applicable. Missing
truth disables PSNR/SSIM/RMSE, not reconstruction. Fit/heldout splitting selects
angles, measurements, counts, I0, masks and per-ray noise arrays together. Heldout
measurements never enter the solver; their normalized residual is returned as
`held_out_projection_residual`. No truth metrics are returned for split execution.

## Examples

```python
from inv_framework.api import CTTaskRequest, list_capabilities, run_ct_task, validate_ct_request

capabilities = list_capabilities()
request = CTTaskRequest("parallel_2d__ellipses_sparse_poisson", "pdhg",
    fidelity="transmission_poisson_nll", regularizer="tv",
    params={"max_iter": 2, "min_value": 0})
validate_ct_request(request)
result = run_ct_task(request)
print(result.convergence_status, result.iterations, result.metrics)

# Transmission data cannot be passed to emission MLEM.
rejected = run_ct_task(CTTaskRequest(request.case_id, "mlem"))
assert rejected.convergence_status == "invalid_parameters"

# An external catalog with observed data and no truth/x.
public = run_ct_task(CTTaskRequest("my_public_case", "sirt",
    data_root="D:/ct-public", device="cpu", params={"max_iter": 20},
    fit_view_indices=list(range(24)), heldout_view_indices=list(range(24, 32)),
    callback=lambda record: record.iteration < 10))
```

CLI: `python -m inv_framework task request.json --out output` calls the same
public dispatcher and writes `reconstruction.npy` and `diagnostics.json`.
Callbacks are Python-only. Existing benchmark/file-output commands share the
in-memory execution implementation under `api/execution.py`; file handling lives
in `api/outputs.py`.

## Dataset selection

The four registered 2D ellipses cases come from `ct-ellipses-new-20261101`.
The small `cone_3d/spheres_astra_12` case supports FDK regression. Other 3D
catalogs can be supplied through `data_root`; see [CT data](datasets.md).

## Domain boundaries

`inv-framework` / `inv_framework` is the numerical distribution, independently
usable without an Agent SDK. CT, MRI and deconvolution do not import one another.
Blind deconvolution belongs to deconvolution, ESPIRiT calibration belongs to MRI.

- `ct.api` exports the existing CT API implementation in `api`. Top-level CT
  names resolve lazily. `invct` and all existing Python imports retain meaning.
- `mri.api` and `deconvolution.api` own their request contracts and discovery.
  MRI and deconvolution each expose seven executable methods. VarNet needs explicit matching weights.
- `experiments.py` holds only shared file references, result fields and static
  resource reading; `runner.py` dispatches one explicit request and saves outputs.
- `core` contains existing tensor operators, fidelities and regularizers. It does
  not import a domain. Measurement parameter conversion lives with the fidelity
  constructors that use it; CT observation simulation imports that same code.
  Existing transmission fidelity remains here to preserve its public import;
  it is CT physics and is not a proposed MRI or deconvolution fidelity.
- Existing flat `solvers`, `models`, `operators`, `regularizers`, and CT
  `configs/algorithms` are retained. In particular the flat solver registry is CT
  coupled, not a shared-domain entry point. MRI and deconvolution use their own discovery interfaces.
  Shared mathematical code requires matching definitions and actual use.

Domain-specific tests are grouped under `tests/core`, `tests/ct`, `tests/mri`,
`tests/deconvolution`. Existing CT tests/data and catalog paths remain unchanged.
The parent Agent repository owns adapters and request/result conversion only.

Shared ADMM/PDHG/FISTA three-domain calls, five regularizer definitions and fixed-block joint ADMM: [composable solvers](composable_solvers.md). A runnable small example is `python examples/composition_quickstart.py --domain mri --algorithm admm --regularizer tv_haar` (also ct/deconvolution).

# Framework-independent experiments

## User data configurations

For your own data, copy an independent annotated configuration and follow its
field definitions: [CT template](ct/request_template.yaml) / [CT format](ct/data_format.md),
[MRI template](mri/request_template.yaml) / [MRI format](mri/data_format.md), and
[deconvolution template](deconvolution/request_template.yaml) / [deconvolution format](deconvolution/data_format.md).
Supply your own arrays and replace acquisition dimensions and model values together.
These templates do not depend on the bundled minimal CPU examples. The examples
provide separate, prefilled small-data instances of the same public request format.


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
Discovery reads every public algorithm from its domain registry/configuration, including shared composition solvers. VarNet requires a local matching checkpoint; blind methods require task_type=blind_deconvolution and model.kernel_shape.
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

# Minimal CPU tasks and data contracts

Ready-to-run external inputs, complete configs, commands and measured results:
[CT FBP](../examples/minimal_cpu/ct/README.md),
[MRI CG-SENSE](../examples/minimal_cpu/mri/README.md),
[Poisson RL](../examples/minimal_cpu/deconvolution/README.md).
Field, axis, unit, precision and evaluation contracts:
[CT](ct/data_format.md), [MRI](mri/data_format.md),
[deconvolution](deconvolution/data_format.md).

In unified runner requests, inputs, model, params and evaluation must be mappings
when present. Explicit null is invalid_request, including fields that may be
omitted. Omit evaluation to disable reference metrics; omit params or use `{}`
for defaults. Omitting CT inputs selects the catalog route; external CT requires
inputs and model. This applies equally to files and mapping requests; it does
not change nullable fields of direct domain APIs. Writable output directories
receive failed result.json and CLI exits nonzero for these request errors,
without an AttributeError traceback.

# CT public task API

`inv_framework.api` exports `CTTaskRequest`, `CTTaskResult`, `list_capabilities`,
`validate_ct_request`, `run_ct_task`, `orthogonal_slices`, and `orthogonal_metrics`.
The 17 algorithm IDs and their parameter descriptions come from one CT registry.
DIP, INR and DPS remain library solvers under `solvers/deep`, outside this task API.

## Discovery and validation

`list_capabilities(context=None)` (keyword-only context for CT) returns algorithms,
YAML defaults, parameter rules, fidelity requirements, regularizers and backends.
Algorithm discovery never reads the example case catalog. Each algorithm lists
supported backends. Optional backend dependencies must be installed separately.

```python
from inv_framework.capabilities import list_capabilities
from inv_framework.ct.cases import list_ct_cases

domains = list_capabilities()  # keys: ct, mri, deconvolution
mri = list_capabilities("mri")  # same result as mri.api.list_capabilities()
print({domain: len(caps["algorithms"]) for domain, caps in domains.items()})
cases = list_ct_cases()  # repository example data, explicitly queried
print([case["case_id"] for case in cases])
# Custom catalog: list_ct_cases(data_root="/path/to/case-root")
```

`inv_framework.list_capabilities()` retains its CT meaning. The aggregate entry
point is explicitly imported from `inv_framework.capabilities`; unknown modalities
raise `ValueError`. Single-domain queries import only that domain.

**Field change:** CT capability results no longer contain `cases`, and the CT
algorithm query no longer takes `data_root`. Replace `caps["cases"]` with
`list_ct_cases(data_root=...)`. Explicit case queries still raise
`FileNotFoundError` when the catalog is absent. Installed algorithm discovery
does not require `tests/data`. CT case execution keeps its existing data lookup.

All algorithm records include `modality`, `name`, `task_types`, `input_specs`
(name, required, axes, dtype and conditional requirements), `outputs`,
`model_requirements`, `runtime_requirements`, `controls`, `dependencies`, `devices`
and `weights`. Existing CT `parameter_specs`/`default_parameters` and the other
domains' `parameter_schema`/`parameters` remain authoritative for parameter types,
defaults, bounds and choices. Omitted bounds mean no independent bound is declared;
coupled conditions remain in applicability, objective and runtime requirements.
MRI calibration exposes sensitivity maps and eigenvalues; GRAPPA and VarNet
expose reconstruction and k-space. Blind deconvolution exposes reconstruction
and estimated PSF, with an optional initial PSF and required kernel shape.

`controls.native` describes actual iteration names, threshold names and stopping
tests. `inner_parameters` describes separate subproblem/final recovery budgets;
these are not outer iteration aliases. For example CG-SENSE and Richardson–Lucy
use `max_iter`, while shared FISTA/ADMM/PDHG use `num_iterations`. CG-SENSE tests a
normal-equation residual; RL tests relative iterate change. Their thresholds are
not interchangeable.

`controls.unified_entry.field_mappings` is the source for MRI/deconvolution
execution mapping. `stopping_policy.max_iterations` sets native `max_iter` or
`num_iterations`; `tolerance` retains the algorithm's criterion described above.
LSQR/CGLS also expose their named native threshold fields where registered.
Inner CG, TV prox and final blind recovery budgets remain separate native params.
Blind budgets apply per pyramid scale; they do not implement threshold stopping.
Direct methods and fixed VarNet inference reject nonempty stopping policies.
Unknown fields and unsupported constraints fail explicitly.

MRI complex objects reject real Box constraints. Deconvolution shared
FISTA/ADMM/PDHG accept `problem_constraints.min_value/max_value` within the solver;
TV-ADMM accepts `nonnegative`. RL/RL-TV accept only intrinsic `nonnegative=true`.
Quadratic shared methods, blind methods and direct methods accept no top-level
physical constraints. Native parameters retain their existing behavior.

MRI/deconvolution precedence is defaults < config params < call-time params <
explicit top-level policies/constraints. `effective_params` records final values;
`diagnostics.parameter_overrides` records top-level overrides of explicit native values;
`call_parameter_overrides` records config values replaced by call-time params.
`effective_stopping_policy` and `effective_problem_constraints` describe what ran.
CT retains its existing algorithm-specific control semantics.

`success` means a valid completed execution, independently of `converged`.
An exhausted budget can return success with converged=false; direct completion
uses `completed_valid` without claiming threshold convergence. Solver iteration
counts and stopping reasons are retained.

`output_artifacts` describes every actual output alongside unchanged `outputs`
keys: path, shape, dtype, axes, representation, units and scale. In-memory paths
are null. Persisted `outputs` and artifact paths are relative to `result.json`,
with arrays saved as `.npy`. Output dtype is taken from each array; computation
precision is separately reported. Unspecified physical units remain unknown or
arbitrary. RL returns object x in `gain*H(x)+background`, without post-rescaling.
Nonfinite numerical diagnostics are null in strict JSON failure records.

File-config inputs, evaluation references, `params.checkpoint` and `data_root`
resolve relative to the configuration file directory, without cwd fallback.
Mapping-request relative paths and CLI output paths use the calling directory.
Only declared file fields are resolved; ordinary model strings are unchanged.
External reference metrics require explicit `evaluation.reference`; the reference
is used after reconstruction. Catalog CT retains optional dataset evaluation.
Once a request file has been read, request/execution failures are saved to a
writable output directory and CLI exits nonzero. Unreadable configs and output
I/O errors retain their original error.

`implemented` describes implementation status. `dependency_installed` only tests
module discoverability; conditional dependencies (such as non-Cartesian MRI's
`torchkbnufft`) are listed separately. `runtime_status.state` is `unavailable`
when a known prerequisite is false, or `pending` when conditions remain unknown.
`reasons` lists known blockers and `unresolved_conditions` lists outstanding
inputs, model, device, weight compatibility and request validation. `runnable`
is `false` for known blockers and `null` for unverified requests. Discovery does
not certify runnable array contents. Backend `runnable` describes only backend
availability, not a complete task; CUDA availability is `null` without probing.
No query runs a solver, probes hardware or downloads/loads weights.

Domain queries accept optional declared context for cheap prerequisite screening:

```python
from inv_framework.mri.api import list_capabilities as mri_capabilities

caps = mri_capabilities(context={
    "available_inputs": ["kspace", "sampling_mask", "sensitivity_maps"],
    "model": {"sampling": "cartesian", "spatial_shape": [24, 28],
              "axes": ["batch", "coil", "y", "x"]},
    "device": "cpu",
})
cg = next(r for r in caps["algorithms"] if r["name"] == "cg_sense")
print(cg["controls"], cg["runtime_status"])
```

Context is advisory prerequisite screening, not array validation. Absence of
known blockers still leaves execution-time shape, dtype and coupled-parameter
checks pending. Recommendations do not change legal execution parameters.

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
An iteration cap is not convergence. Direct CT FBP/FDK use `direct_reconstruction_valid` with zero
iterative steps. Failed solver results preserve actual iterations and available
partial reconstruction. Failure before a solver returns has no partial image.
Unexpected implementation exceptions propagate; ordinary request/data errors,
missing backend, numerical failures and CUDA OOM have structured results.

2D arrays have shape `(B,C,H,W)`; 3D arrays `(B,D,H,W)`. External torch inputs
retain float64 when supplied as float64; otherwise CT inputs convert to float32.
Backend paths and inner computations can differ; read the actual output dtype
and computation_precision diagnostics. Reconstruction
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

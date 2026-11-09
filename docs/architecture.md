# Architecture and terminology

## Execution and module boundaries

`inv-framework` executes explicitly specified CT, MRI and deconvolution tasks.
The shared `run_experiment` entry point resolves a file or mapping request and
dispatches it to the domain API. Each domain validates its inputs, assembles
the physical model and solver, and returns reconstruction diagnostics.
An optional evaluation reference is used only after reconstruction.

- `inv_framework/api/`: CT request validation, dispatch, execution and CLI.
- `inv_framework/core/`: shared operators, fidelities, regularizers and numerical operations.
- `inv_framework/ct/`: CT geometries, measurement domains, cases and backends.
- `inv_framework/mri/`: MRI encoding, calibration and domain-specific solvers.
- `inv_framework/deconvolution/`: known-kernel and blind image reconstruction.
- `inv_framework/solvers/`: iterative, composable and deep-prior solvers.
- `inv_framework/runner.py`: shared file requests, result storage and independent evaluation.

The numerical package runs independently of Agent SDKs. Algorithm selection,
parameter search and scheduling subsequent experiments belong to the caller.
See the [task API](agent_algorithm_api.md) for request and result contracts and
[shared mathematics](shared_mathematics.md) for operator and solver conventions.

## Domain terminology

**Observation**: Measured data available to reconstruction: CT line integrals/counts,
MRI complex k-space, or a blurred intensity/count image.

**Physical model**: Geometry and acquisition conventions defining how an unknown
object produces the observation. Sampling masks and PSF origins are explicit.

**Calibration**: Estimation of MRI coil sensitivities from actual calibration
observations; its output is maps rather than a reconstructed object.

**Blind deconvolution**: Joint estimation of an image and a spatially invariant
PSF from a blurred observation and declared kernel constraints.

**Evaluation reference**: Optional truth used after reconstruction for quality
measurement. It is not an observation, initialization or calibration input.

**Experiment**: One explicitly specified reconstruction, calibration or blind
estimation task. Selection of subsequent experiments belongs to its caller.

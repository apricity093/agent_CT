# MRI tasks

Primary showcase/quality inputs: [standard lightweight benchmarks](../public_benchmarks.md).
Run tools/run_public_benchmarks.py; the analytic MR phantom is separate from
the authorized local fastMRI slice needed for clinical-distribution VarNet evaluation.

Use MRITaskRequest and run_mri_task from inv_framework.mri.api, or run_experiment
and inv-run. zero_filled and cg_sense accept external observations, without a
catalog or reference image. GRAPPA, CS, ESPIRiT calibration and checkpoint-backed
VarNet also execute. VarNet supports CPU/CUDA; other MRI methods use CPU.
Non-Cartesian sampling needs `.[mri]`; VarNet needs an explicit matching checkpoint.
See [dependencies and installation](../../DEPENDENCIES.md).

See data_format.md and algorithms.md for mathematical conventions and shapes.
For runnable generated examples see ../public_benchmarks.md. Missing sensitivity
maps returns invalid_parameters; it never triggers another algorithm.

See ../public_benchmarks.md for external-input API/CLI examples and editable benchmarks.

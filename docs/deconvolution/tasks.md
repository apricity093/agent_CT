# Deconvolution tasks

Primary showcase/quality inputs: [standard public-image benchmarks](../public_benchmarks.md)
with Levin09 and ideal Airy PSFs. Run tools/run_public_benchmarks.py.
The final algorithm/input mapping is in ../imaging_catalog.md.

DeconvolutionTaskRequest and run_deconvolution_task accept external arrays/files.
wiener_hunt and richardson_lucy execute reconstruction without a catalog or truth.
TV ADMM, RL-TV and Hyper-Laplacian also execute. blind_dcp and blind_l0 execute task_type=blind_deconvolution; see ../public_benchmarks.md.
No algorithm substitution, search or retry occurs. See algorithms.md,
data_format.md and ../public_benchmarks.md for executable calls.

See ../public_benchmarks.md for external-input API/CLI examples and editable benchmarks.

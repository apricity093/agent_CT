# inv_framework

For CT, MRI and deconvolution dependencies and device requirements, see
[dependencies and installation](../DEPENDENCIES.md).

The public CT entry point is `inv_framework.api`. See
[the task contract](../docs/agent_algorithm_api.md).

`api` owns requests and execution, `core` owns mathematical abstractions,
`ct` owns measurement data and backends, and `solvers` owns numerical iteration.
Deep priors remain available as library solvers under `solvers/deep`.

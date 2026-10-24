# Data fidelity execution paths

The default remains `parameters.data_fidelity: l2`. FISTA, ADMM and PDHG accept `gaussian_nll` and `wls`; only PDHG accepts `transmission_poisson_nll`. Existing prior/constraint restrictions still apply. FISTA does not acquire H1, Huber or constrained TV/Haar proximal capabilities through this extension.

Registry parameter definitions drive configuration loading and Agent allowed parameters. Adapter forwards scalar parameters through the existing CLI overrides. Runtime `build_solver` constructs a fidelity using public measurement arrays; the existing detailed solve route passes this object to the numerical core and independent endpoint confirmation. No truth or clean projection enters this construction.

Gaussian NLL requires public sigma and uses weights 1/sigma squared. Equivalent L2 strength is lambda_NLL times sigma squared for constant sigma. WLS requires public weights and explicit weight_source: known_variance, provided, post_log_poisson or post_log_poisson_gaussian. Post-log weights are approximations; reconstruction objectives across likelihood models are not comparable for ranking. Masks multiply quadratic weights. Exact masked transmission NLL is currently rejected.

Pure transmission NLL requires declared pure Poisson noise, raw counts and positive I0; mixed readout and y-only cases are rejected. Scale follows the observation metadata. The objective omits count-only constants and can be negative. The named P3 transmission NLL configuration requires min_value=0. NLL tolerance/max iterations are exposed as nll_tolerance/nll_max_iterations; tolerance must be achievable for the observation dtype.

FISTA uses the fidelity gradient, value and checked smooth majorizer. ADMM uses the quadratic weighted normal equation and existing summable CG tolerance. PDHG keeps weights inside f, uses prox_(sigma f*) and the unweighted A norm bound. Endpoint objective and fixed-point checks consume the same fidelity. The numerical step size constraint is tau*sigma*bound([A;T])<1. Explicit non-L2 discrepancy stopping is rejected; use discrepancy.enabled=false until model-specific calibration is available.

Python detailed solvers also accept `fidelity=` directly. YAML examples are in configs/examples/p3_*.yaml.


Float32 NLL proximal inputs use float64 working roots and return the original dtype/device. Diagnostics distinguish the working optimality residual from the residual after casting the root back. The latter can be amplified by local curvature and is not advertised as float64 accuracy. Float64 scalar-reference tolerances are unchanged. Every vector bisection iteration, failed call and synchronized proximal time is accounted once.

Weighted 32² ADMM acceptance sets cg_max_iterations=500 explicitly, while retaining cg_tolerance=1e-6 and its summable outer schedule. The legacy default of 100 remains unchanged; an insufficient inner budget reports cg_inner_max_iterations, without a complete endpoint certificate. Missing split state now makes independent endpoint confirmation fail.

The parameter recommendation converts a known scalar Gaussian sigma by lambda_NLL=lambda_L2/sigma² and records the equivalent L2 value. Explicit strength overrides keep their requested model units. Heterogeneous variances have no single scalar L2 conversion; use the recorded weighted objective.

The quadratic path supports I/H1 CG without box constraints. ADMM requires a positive-definite inner normal system and a finite minimizer. For gradient priors with zero data weights, a common data/transform null space is rejected. All-zero effective counts with an unpenalized constant mode may have no finite minimizer; an iteration limit is not a convergence certificate.

## Numerical evidence

The saved [accepted summary](../artifacts/ct_composable/p3/acceptance-20261023/accepted_summary.json)
and [per-configuration resources](../artifacts/ct_composable/p3/acceptance-20261023/accepted_resources.json)
index the actual parameters, complete outer iterations, endpoints, devices and costs.
[Excluded records](../artifacts/ct_composable/p3/acceptance-20261023/excluded_resources.json)
include early zero-outer-iteration results; they are not accepted reconstructions.
[Negative records](../artifacts/ct_composable/p3/acceptance-20261023/negative_resources.json)
preserve rejected and interrupted attempts. Raw referenced paths remain available.
Independent scalar, Hessian/conjugate and small-CT references remain in the same
result directory and in permanent tests. Historical wall times include concurrent
checks and do not support speed rankings. TV uses finite FGP, not an exact gap certificate.

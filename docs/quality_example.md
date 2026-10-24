# Independent quality example

`examples/p5_quality.py` provides a bounded CPU example with independent pilot
and formal phantoms and seeded observations. From the CT repository root:

```powershell
python examples/p5_quality.py pilot --out artifacts/quality-example
python examples/p5_quality.py freeze --out artifacts/quality-example
python examples/p5_quality.py formal --out artifacts/quality-example
```

Use a fresh output directory. `formal --resume` continues unsaved jobs with the
same frozen configuration. The script uses `build_solver`, `solve_detailed`,
`simulate_observation`, `SolveControl` and counted operators. It does not invoke
Agent selection or automatic tuning. Truth generates observations and supplies
independent image metrics; the solver-visible case has no truth or clean data.

The example uses 32² matched parallel-beam geometry, CPU float32, zero initialization,
unit-grid priors and full-image evaluation with data_range=1. Pilot and formal
phantoms and seeds differ. Fixed settings are sigma=.05, I0=10000 for C4 and
I0=20 for C5/C6, sigma_e=3 for C6, scale=1 and log floor=1. C6 retains negative
raw c_obs; its WLS weights are an observation-based delta-method approximation.

The mean-loss coefficient alpha_reg=.01/1536 corresponds to summed-objective
strength=m*alpha_reg, where m is the number of measurements. Huber delta=.01,
Haar level=2 without LL penalty, and isotropic TV are engineering defaults.
No cross-validation or quality tuning is performed. Reconstruction budgets are
F/A each 300, outer iterations 2000 and wall 600 seconds, including initialization
and native diagnostics. One full-observation endpoint forward is reserved.
PDHG uses the native 1e-5 criterion with patience 5; CGLS uses 25 iterations,
SART 5 sweeps with relaxation=.1 and block_size=1, and FBP is direct.

Compare image quality, data consistency, optimization state and cost separately.
Training objectives from different likelihood models are not a common ranking
scale. The script evaluates full-observation fit; it does not implement independent
held-out selection. An independent evaluation protocol must freeze its angles,
seeds and extra operator budget before data generation and score all candidates.

Budget exhaustion and max_iterations do not establish optimal quality. Single
phantom/seed and one resolution do not establish general method rankings. The
previous SART attempt interrupted before structured cost output has no recoverable
exact cost; accepted-endpoint costs do not represent all historical attempts.


Saved results: [frozen protocol](../artifacts/ct_composable/p5/acceptance-20261023/manifest.json),
[formal endpoints](../artifacts/ct_composable/p5/acceptance-20261023/formal/summary.json),
[independent held-out scores](../artifacts/ct_composable/p5/acceptance-20261023/heldout/scores.json)
and [verification](../artifacts/ct_composable/p5/acceptance-20261023/final_verification.json).
The held-out protocol was frozen as a later supplement before its data generation;
it was not in the original reconstruction protocol and did not select or refit outputs.
Its extra 24 forward calls are separate from reconstruction budgets. The early
SART interruption and missing exact cost remain recorded in formal.log.

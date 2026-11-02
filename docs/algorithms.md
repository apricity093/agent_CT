# CT algorithms

The registry retains 17 entries: fbp, fdk, sirt, landweber, cgls, lsqr, sart,
os_sart, mlem, osem, tikhonov, tv_fista, fista, admm, pdhg,
constrained_pdhg and asd_pocs_tv.

`core/operator.py` contains linear transforms and adjoints; `core/fidelity.py`
contains measurement objectives; `core/regularizer.py` contains priors and proximal
maps. Solvers are grouped into analytic, algebraic, subset, statistical and
splitting modules. DIP, INR and DPS remain in `solvers/deep`.

- [Composable reconstruction](composable_reconstruction.md)
- [Constrained TV](constrained_tv.md)
- [Numerical validation](numerical_validation.md)

FISTA and PDHG retain automatic operator-norm-based steps. Transmission likelihoods
retain logarithm/exponential stability bounds. Emission MLEM/OSEM do not consume
transmission photon counts. Ordered subsets retain projection-angle scheduling.
The public registry is the authoritative compatibility matrix and parameter source.

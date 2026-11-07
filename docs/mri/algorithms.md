# MRI algorithms

`zero_filled` and `cg_sense` execute on CPU for 2D/3D spatial arrays, with
arbitrary batch and coil counts. Defaults and parameter rules have one source:
`configs/mri/algorithms/*.yaml`, read directly by discovery and execution.

- `zero_filled`: mask observed k-space, centered orthonormal IFFT per coil.
  Default `coil_combination=sense` returns sum(conj(S)*coil_image)/sum(abs(S)^2),
  zero where map power is zero. It is a complex sensitivity-combined baseline,
  not an inverse for undersampling. Explicit maps are required even for one coil.
  `rss` returns sqrt(sum(abs(coil_image)^2)), a real magnitude image with coil
  sensitivity dependent scale. RSS needs no maps and has no complex phase.
- `cg_sense`: solves (A* W A + lam I)x = A* W y with complex Hermitian CG,
  A=PFS. Maps are supplied without renormalization or implicit calibration.
  Single coil is exactly the same model with one explicit map (ones is valid).
  Data weights are W, not sqrt(W); they enter the equation once. Zero default
  initialization; optional complex initial_estimate. Relative normal residual
  is measured against ||A*Wy||; tolerance and max_iter control stopping.
  Cartesian FFT and non-Cartesian NUFFT share centered coordinates and 1/sqrt(N)
  scale. NUFFT uses torchkbnufft 1.5.2 CPU double precision and its default
  interpolation parameters (2x oversampling, six neighbors, table 1024).

Direct zero-filled execution reports completed_valid, zero iterations and
converged=false. CG distinguishes converged, max_iterations and numerical_error.
Diagnostics contain weighted/unweighted data residuals, relative normal residual,
normal residual history, actual dtype/device and coil combination. For RSS, data
residual is computed from coil images, not by encoding the RSS magnitude as an
object. Data residuals apply the sampling mask. Neither method clips outputs.

VarNet executes with an explicitly prepared matching checkpoint; see public_benchmarks.md in the parent documentation directory.
No PnP, DPS, DiffPIR or DeepInv is invoked.

## Calibrated and regularized methods

All methods execute on CPU. Defaults/types/ranges are in algorithm YAML; explicit
params override them. No automatic calibration, PSF normalization or tuning occurs.

GRAPPA (`grappa`) accepts 2D Cartesian (batch,coil,y,x), at least two coils,
identical binary masks across coils and explicit fully acquired calibration_data
(batch,coil,acs_y,acs_x). Each missing target groups its local acquired-neighbor
pattern within odd kernel_width >=3. Valid ACS patches form S, center coil samples
form T; solve min ||SK-T||_F² + ridge*||S* S||_F/ncols*||K||_F².
Augmented least squares handles zero ridge/rank deficiency. No synthesized sample
is reused as a source. Outer-grid kernels use zero padding and only acquired
neighbors; edge fits can be weaker. Any local pattern with no acquired neighbor
is rejected (increase kernel width or supply a different acquisition). There is
no acceleration-factor whitelist or non-Cartesian GRAPPA. Output kspace retains
all acquired samples, including measured zeros, and fills missing samples.
reconstruction is magnitude RSS of centered ortho per-coil IFFTs. Calibration
relative residuals report fits for each batch/pattern, not reconstruction error.
Fully sampled ACS is a caller input assertion; no implicit ACS extraction.

ESPIRiT (`espirit`) requires task_type=calibration, only calibration_data, and a
2D Cartesian output spatial_shape. ACS dimensions fit that grid and the odd square
kernel_width. SVD of valid ACS patch rows retains right singular vectors above
svd_threshold times the largest singular value (0<threshold<1). Nonzero ACS is
required. Centered zero-padded kernels are inverse FFT transformed. Sum of their
coil outer products, scaled by prod(spatial_shape)/kernel_width², defines the
image-domain Hermitian calibration operator. Dense eigh gives the largest
eigenvalue and unit-norm eigenvector at every pixel. Crop where eigenvalue <=crop
(0<=crop<=1). The strongest coil fixes display phase; compare map projectors or
phase-aligned maps, not raw vectors. Outputs sensitivity_maps (batch,coil,y,x),
eigenvalues (batch,y,x), retained ranks. One map set only; no full soft-SENSE.
Pass these maps explicitly to a subsequent SENSE/CS request. Image truth evaluation
is not defined for a calibration task. Dense SVD/eigh can be expensive for large
coil/kernel dimensions; no hidden coil compression is performed.

CS (`cs_wavelet`, `cs_tv`) minimizes .5 sum W|Ax-y|² + lam R(x), with supplied
complex maps, the existing Cartesian or NUFFT encoding, nonnegative real weights
W (not sqrt W), arbitrary batches and 2D/3D spatial arrays. Complex x is
unconstrained. ADMM splits z=Tx, solves (A*WA+rho T*T)x=A*Wy+rho T*(z-u) with
Hermitian CG, shrinks Tx+u, and updates u. rho>0 is a splitting penalty; no explicit
gradient step size or operator norm estimate is needed. cg_max_iter caps each
inner solve; cg_tolerance/iteration² is its summably decreasing relative target.
Convergence requires primal, dual and inner normal residuals <= tolerance times
max(1,||x||). The inner stopping reason remains visible if CG reaches its cap.

Wavelet uses the orthogonal separable Haar basis, levels>=1, periodized dyadic
blocks, all spatial dimensions divisible by 2**levels. The analysis penalty sums
complex coefficient magnitudes including low-frequency coefficients. Orthogonality
makes the equivalent synthesis model possible, but the implementation splits the
analysis transform. Complex soft shrinkage preserves phase. No PyWavelets dependency.
TV uses forward differences x[i+1]-x[i] on spatial axes only, periodic unit spacing,
and R=sum_pixels sqrt(sum_spatial_axes |Dx|²); this is complex isotropic TV, not
TV of magnitude and not SigPy's anisotropic TV default. Its shrinkage groups the
spatial directions at each complex voxel; batches remain independent.

max_iter is an outer iteration cap; tolerance is a nonnegative relative stopping
threshold. lam>=0 weights unnormalized sums (no division by pixel/sample count).
Diagnostics include data_fit, regularizer_value, regularizer_contribution,
primal/dual/inner residual history and weighted/unweighted fit residuals. These
are numerical evidence and do not diagnose image quality or choose parameters.

## End-to-end VarNet

`varnet` implements fastMRI learned sensitivity estimation, normalized complex
U-Nets and soft k-space data consistency: k <- k - w P(k-y) - FS U(S*F*k).
The final output is magnitude RSS of reconstructed coil images. No supplied
ESPIRiT maps replace the learned sensitivity model. The acquired-sample residual
is computed on predicted coil k-space, not by encoding magnitude RSS.

Native complex inputs convert to real/imaginary pairs; real and imaginary groups
are independently normalized with sample standard deviations. Zero variance and
zero coil power use 1e-12 floors. Symmetric zero padding reaches multiples of
2**pools and is removed after each U-Net; default four-pool behavior matches
upstream. Output retains the full acquisition grid. Caller-selected evaluation
cropping must be performed explicitly on both reference and output.

Structure controls num_cascades, chans, pools, sens_chans, sens_pools must match
checkpoint. Raw state dictionaries and Lightning state_dict dictionaries whose
keys all start with varnet. are accepted; that prefix alone is removed. Strict
loading rejects missing, unexpected or mismatched parameters. Inference uses
eval and inference_mode; no random-weight execution occurs when weights are missing.
Only CPU/CUDA, 2D Cartesian shared column sampling is supported. Spatial grids
are not fixed; InstanceNorm needs more than one bottleneck spatial element.
This is neither non-Cartesian MRI nor a 3D network. Batch can contain independent
slices. Official knee/brain weights are distribution-specific, independently of
the supported array sizes. Synthetic inference is not clinical quality validation.
Small synthetic inputs can differ substantially from the checkpoint's training
distribution. Matrix size alone is not an applicability threshold or a guarantee
of quality. The current MR phantom validates inference and task integration;
it does not establish clinical image quality. See ../imaging_catalog.md.

## Calibration parameter interpretation

Actual continuous acquired ACS is required; random sampling outside ACS does
not imply that calibration samples are absent. Noise, calibration extent and
object/coil support affect the singular spectrum. svd_threshold and crop are
caller controls; inspect retained rank, eigenvalues and map support, then compare
explicit downstream reconstructions on development acquisitions. A larger SVD
threshold is not universally better. Do not silently raise defaults or attribute
all calibration error to noise. Current examples use explicitly configured ACS
and report calibration outputs separately from image reconstruction.

# MRI data conventions

inputs: kspace, sampling_mask (Cartesian), trajectory (non-Cartesian),
sensitivity_maps, optional data_weights and initial_estimate. Values are native
arrays or explicit NPY/NPZ path/key references. All observations are finite.
Model axes must be [batch,coil,y,x] or [batch,coil,z,y,x]; spatial_shape declares
the corresponding positive dimensions. Cartesian kspace/maps are
(batch,coil,*spatial_shape), non-Cartesian kspace is (batch,coil,sample).
Reconstruction and initial_estimate are (batch,*spatial_shape).

model.sampling is cartesian or noncartesian. fft_centered=true and fft_norm=ortho
are the only conventions. FFT = fftshift(fftn(ifftshift(x),norm='ortho')) over
spatial axes only. Cartesian mask has exactly kspace shape and binary entries;
broadcast explicitly before calling. Observed zeros remain sampled; unobserved
entries are ignored regardless of their value. data_weights have kspace shape,
are real nonnegative inverse-variance weights, and default to ones.

Trajectory is (sample,spatial_dimension), shared across batches and coils.
trajectory_units is radians/voxel, cycles/voxel or grid_indices. At entry cycles
multiply by 2*pi, centered grid indices multiply by 2*pi/spatial_shape. Converted
coordinates must be in [-pi,pi). Spatial coordinates are n-floor(N/2), with
negative exponential forward and conjugate positive exponential adjoint.
Physical inverse-length coordinates require caller conversion with voxel size.
Install the mri extra for NUFFT. There is no second NUFFT backend.
Density compensation is not implicitly applied: explicitly pass it as data_weights
if the intended objective uses it. No implicit calibration or gridding baseline.

Evaluation via runner is optional: evaluation: {reference: {path: truth.npy},
data_range: 1.0, mode: magnitude}. Magnitude mode compares abs(reconstruction)
with abs(reference), without scaling or phase alignment; complex mode returns
complex RMSE and PSNR without SSIM. Use a matching RSS or object reference for
the selected coil combination. Real SSIM uses local population moments with
window up to seven spatial pixels/voxels and averages valid centers, independently
across batch. Exact zero RMSE omits infinite PSNR from finite JSON. Truth is
passed exclusively to the evaluator after reconstruction.

Calibration/GRAPPA additionally accept calibration_data (batch,coil,acs_y,acs_x), a fully acquired Cartesian ACS rectangle with native complex values. ESPIRiT accepts only that input, with task_type=calibration. GRAPPA accepts kspace, sampling_mask and calibration_data only. CS uses the same encoding inputs as cg_sense; no implicit calibration occurs.

VarNet accepts only kspace and sampling_mask, native complex (batch,coil,y,x),
at least two coils. Masks have exactly this shape and must be constant across
coils and readout y; columns x are the phase encoding dimension. The explicit
params.num_low_frequencies declares a fully sampled centered ACS band starting
at (width-num_low_frequencies+1)//2. Default 24 is an example acquisition setting,
not automatic calibration inference; change it to the actual acquisition width.
params.checkpoint is a local file name; file-task relative checkpoint paths resolve
against that task file. No array reference or checkpoint is fetched at execution.
Outputs are reconstruction (batch,y,x) magnitude RSS and complex kspace on the
original grid. Missing checkpoint reports missing_checkpoint. Structure mismatches
report invalid_parameters. See ../public_benchmarks.md for preparation.

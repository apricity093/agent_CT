# Deconvolution inputs and evaluation

inputs contains finite real observation, psf, and optional initial_estimate (iterative methods), as arrays or NPY/NPZ path/key references. model.axes is [y,x], [z,y,x],
[y,x,channel] or [z,y,x,channel]. PSF contains spatial axes only. No channel mixing
or batch dimension is implicit. psf_origin explicitly indexes each PSF axis.
model.boundary is periodic or zero. noise_model is gaussian (Wiener/HQS), poisson (RL/RL-TV), or either for tv_admm; RL accepts scalar positive gain and nonnegative background in count units.
Neither PSF nor data is silently normalized. Pixel spacing in the Laplacian is one.

Evaluation in runner is optional: evaluation: {reference: {path: truth.npy},
data_range: 1.0}. Real RMSE, PSNR and local SSIM use original intensities, without
fitting to truth. SSIM uses population moments, window up to seven per spatial
axis, valid centers, independent channels. Exact zero RMSE omits infinite PSNR
from finite JSON. No truth reaches the solver. Without reference, execution still
returns reconstruction and data diagnostics. Direct domain APIs do not evaluate
or save; inv_framework.evaluation.array_metrics.evaluate_array is independently
callable on returned arrays with spatial_axes.

Blind methods require task_type=blind_deconvolution. inputs contains only
observation and optional initial_psf; true PSF/image are never solver inputs.
model.axes is [y,x] or [y,x,channel]; shared spatially invariant Gaussian blur,
periodic boundary, gain=1, background=0. model.kernel_shape contains two odd
sizes >=3 within the observation. The output origin is kernel_shape//2 (an
explicit psf_origin, if supplied, must match). Output psf is a nonnegative
unit-sum spatial array; reconstruction preserves the observation shape.
Kernel support, residual, mass/minimum, scale/iteration history, inner solver
termination and final recovery settings are diagnostics. Original outputs remain
unaligned; any shift-aligned scoring belongs exclusively to independent evaluation.

# Known-PSF algorithms

wiener_hunt and richardson_lucy execute on CPU, 2D/3D spatial arrays with optional
independent last-axis channels sharing the same spatial PSF. Defaults and rules
are read from configs/deconvolution/algorithms, also used by discovery.

Convolution is Hx[i]=sum_j h[j] x[i-(j-origin)]. No PSF normalization occurs.
Periodic boundary wraps; zero boundary sets outside values to zero and returns
a same-size crop. The adjoint transposes that exact crop, including even and
asymmetric PSFs. PSF support cannot exceed image dimensions.

Wiener-Hunt minimizes ||Hx-y||² + balance||Lx||² with periodic discrete Laplacian
L (unit spacing, 2d center and -1 neighbors). FFT closed form uses conj(H)Y /
(|H|²+balance|L|²); unsupported zero boundary is rejected. Unpenalized null modes
use the minimum-norm zero coefficient. Only Gaussian Hx, gain=1, background=0
is accepted. It reports completed_valid, zero iterations, converged=false.

RL uses Poisson mean gain*Hx+background and update
x <- x * [gain H*(y/max(mean,epsilon))] / [gain H*1]. Unsupported object pixels
with zero sensitivity are set to zero. The matched sensitivity denominator is
retained for both periodic and zero boundaries, nonunit PSFs and nonunit gain.
Input data, PSF, initialization and background must be nonnegative; gain positive.
Floating nonnegative count arrays are accepted for likelihood experiments.
Default initial image is constant initial_value=.5 in object units. Supplied
initialization zeros stay zero under multiplicative EM. epsilon protects the
Poisson denominator/log; tolerance uses relative iterate change. NLL history
omits the data-only log-factorial constant. A cap is max_iterations, not convergence.

Neither method clips input or output. Diagnostics include residual against the
actual model, PSF sum, boundary, origin, gain/background and output extrema.
RL additionally returns NLL history. Blind methods execute as specified below. No external complete reconstruction function is wrapped.

## Regularized reconstruction

`tv_admm`, `rl_tv`, `hyper_laplacian` execute CPU 2D/3D real reconstruction with
periodic convolution, unit-spacing periodic spatial differences and optional
independent last-axis channels. gain=1 is required. PSF remains unnormalized.
Zero convolution boundary is still supported by the baseline RL, but is not
claimed for these regularized methods. Parameters have a single default source
in configs/deconvolution/algorithms. No image clipping is applied except the
explicit nonnegative constraint in Poisson TV.

`tv_admm` minimizes f(Hx)+lam sum sqrt(sum_axes (Dx)²). Gaussian f(z)=.5||z-y||²,
background=0, unconstrained x. Poisson f(z)=sum(z+b-y log(z+b)), y>=0, b>=0,
x>=0 and nonnegative positive-sum PSF. This is emission Hx+b, not CT transmission.
ADMM splits z=Hx, d=Dx, v=x; all penalties rho>0. The x FFT system has denominator
|H|²+sum_axes |D|²+1. Gaussian z=(y+rho*(Hx+u_z))/(1+rho). Poisson computes the
nonnegative root for mean t=z+b of rho(t-(Hx+u_z+b))+1-y/t=0, including y=0;
this is a different subproblem. d is vector soft thresholding at lam/rho. v is
identity for Gaussian and projection to x>=0 for Poisson. Scaled duals accumulate
constraint errors. Returned v is feasible; primal residual includes its distance
from x. Dual residual is rho||H*(z-z_old)+D*(d-d_old)+v-v_old||. Both must be <=
tolerance*max(1,||x||) for converged; otherwise report max_iterations. Outputs
include fit, raw/weighted TV, constraint residuals, and final projection count.
The planned name tv_split_bregman was replaced with tv_admm to describe the actual
three-split ADMM update, rather than claim Getreuer's exact iteration sequence.

`rl_tv` implements the Dey multiplicative fixed-point form with a specified
smooth discrete TV variant R_e=sum sqrt(sum_axes(Dx)²+tv_epsilon²):
x_new=x * H*(y/max(Hx+b,epsilon)) / (H*1 + lam D*(Dx/sqrt(|Dx|²+tv_epsilon²))).
D* is minus divergence. This retains the likelihood sensitivity H*1; nonunit PSF
sum and constant b>=0 are allowed. The forward differences and their exact
transpose replace continuum derivatives; epsilon only floors the Poisson mean.
The positive tv_epsilon defines the smoothed prior, not arbitrary denominator
clipping. A nonpositive update denominator returns failed/numerical_error with
the last valid image and minimum denominator; decrease lam explicitly if desired.
There is no automatic retry or change of method. Nonnegative initial_estimate or
positive constant initial_value is used; initialization zeros remain locked.
Tiny negative FFT roundoff in the nonnegative numerator is floored at zero.
Stopping is relative iterate change <=tolerance; no monotonic objective guarantee
is asserted. This method is separate from Poisson-TV ADMM and from RL+denoising.

`hyper_laplacian` minimizes .5||Hx-y||² + lam sum_axes,pixels |Dx|^p, 0<p<1,
Gaussian only, background=0, unconstrained real image. This is an anisotropic
nonconvex gradient prior. HQS alternates global scalar minimization
.5(w-g)²+(lam/beta)|w|^p and FFT x update with denominator |H|²+beta sum |D|².
The paper puts lambda on the data term; project lam is its reciprocal and beta
is expressed in project objective units. One alternating pair is used at each
continuation value, followed by repeated pairs at beta_max.
The scalar optimum is either zero or the larger positive stationary root with
the sign of g. The derivative turning point brackets that root; 70 bisections
and objective comparison select the global minimum, including threshold ties
(choose zero). This replaces the author's LUT without changing the subproblem.
Unidentified Fourier modes use minimum norm zero. beta_initial>0 increases by
beta_factor>1 up to beta_max>=beta_initial. At beta_max, relative iterate change
<=tolerance means convergence of that finite-penalty alternating problem, not a
guarantee of the global optimum of the nonconvex unsplit model. max_iter caps all
HQS steps. Diagnostics retain the actual per-step beta, splitting residual and
relative change. No automatic continuation beyond the declared schedule.

lam weights unnormalized sums; max_iter and tolerance control the stated stopping
rules. All three return the actual data-fit value, raw regularizer, lam times
regularizer, model residual, extrema and algorithm-specific residual history.
Poisson fit omits the data-only factorial term and floors means at 1e-12 for
finite diagnostics. These quantities alone do not prove an image has ringing,
noise or excessive smoothing. Explicit initialization must match observation shape.

## Spatially invariant blind reconstruction

`blind_dcp` uses Pan et al. CVPR 2016: ||k*x-y||^2 +
mu ||Dx||_0 + lambda ||dark(x)||_0. The dark channel minimizes over all channels
and an odd periodic patch, storing original-image indices. Its gather transpose
is scatter-add, including repeated selection of the same pixel. HQS hard
thresholds scalar gradient and dark auxiliaries, freezes the current min selection,
and solves (H*H+alpha D*D+beta M*M)x=H*y+alpha D*g+beta M*u by CG.
Both penalties grow by penalty_factor each latent iteration. Scalar gradient
components are counted separately. Diagnostics record inner normal residuals and
termination; reaching an inner cap is not an exact solve or global convergence.
Kernel estimation uses gradient-domain ridge least squares as in equation 16.

`blind_l0` uses Xu, Zheng, Jia CVPR 2013, equations 5, 9--12 and 15:
||k*x-y||^2 + lam sum min((Dx/epsilon)^2,1) + gamma ||k||^2.
At each outer iteration epsilon starts at epsilon_initial and halves through
four default levels. Each level uses 2**level alternating updates: scalar
auxiliary l=Dx when |Dx|>epsilon, zero otherwise; x is the periodic FFT solution
with gradient weight lam/epsilon^2. Kernel estimation uses intensity-domain ridge
least squares. This is the paper's unnatural near-L0 representation, not the
2011 smoothing method nor Pan's 2014 text/intensity prior. Default outer cap is
five iterations per scale. lam is strictly positive; intensities and epsilon
are in caller-supplied units (the paper schedule assumes normalized images).

Both estimate a compact nonnegative kernel via L-BFGS-B ridge least squares
with analytic FFT correlation derivatives, then normalize its mass to one.
This directly enforces support and nonnegativity, replacing the papers' full-grid
FFT estimate followed by cropping/clipping. No exact simplex minimizer is claimed.
Kernels are centered by integer mass centroid with zero fill inside support;
the structural image is shifted in the opposite direction. Removed support-tail
mass is renormalized. This uses only the estimate, never a true kernel.

The Gaussian-antialiased image pyramid and bilinear kernel interpolation are
centered at integer array origins. Kernel displacements scale with the actual
image-grid size ratio, rather than the rounded kernel support ratio. Odd supports
and their centered origins are retained; the finest level returns kernel_shape.
Periodic boundaries apply to both images and dark patches; no edge taper or
MATLAB boundary extension is claimed. Optional initial_psf is normalized once.
Default initialization is a centered delta; local minima, including near-delta
solutions, are legitimate failure modes of blind estimation.

Both methods recover the final natural image using the existing Hyper-Laplacian
p=0.5 HQS with final_lam/final_max_iter; remaining controls come from that solver's
single YAML source. This agrees with Xu's final prior but is an explicit final
recovery variant of Pan's DCP, whose paper uses Pan 2014 artifact removal.
Structural latent images are never returned as final reconstructions. No clipping,
truth-based kernel alignment or automatic method switching occurs. Scale/outer
caps report max_iterations, converged=false; final recovery reports its own stop.
No universal restoration or global optimum guarantee is made.

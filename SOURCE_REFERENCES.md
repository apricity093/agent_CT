# Numerical sources

These are equation-based project implementations, not wrappers around complete
upstream reconstructions. No upstream source text has been copied or vendored.
Upstream authors retain copyright; package distributions retain their licenses.

- MRI SENSE: Pruessmann et al., MRM 42 (1999), 952–962, and MRM 46 (2001),
  638–651. Read [SigPy v0.1.27 mri/linop.py Sense](https://github.com/mikgroup/sigpy/blob/v0.1.27/sigpy/mri/linop.py),
  [mri/app.py SenseRecon](https://github.com/mikgroup/sigpy/blob/v0.1.27/sigpy/mri/app.py)
  and [alg.py ConjugateGradient](https://github.com/mikgroup/sigpy/blob/v0.1.27/sigpy/alg.py)
  (BSD-3-Clause). Same weighted quadratic model; this project explicitly requires
  masks independent of observed values, batch/coil axes and supplied maps. No
  calibration, implicit map scaling, coil batching or automatic weight inference.
  CG uses conjugate inner products and a relative normal residual. Zero-filled
  sensitivity combination and RSS are separately named conventions.
- NUFFT: [torchkbnufft 1.5.2 KbNufft/KbNufftAdjoint](https://torchkbnufft.readthedocs.io/en/stable/)
  (MIT), installed numerical dependency. Fessler & Sutton, IEEE TSP 51 (2003),
  560–574, min-max interpolation. Default kernel settings; centered coordinates,
  radians/voxel and symmetric 1/sqrt(prod(spatial_shape)) scaling outside both
  transforms. This is CPU complex128 adaptation, not SigPy kernel replication.
  Direct DFT tests use method-appropriate approximation tolerances (2e-3).
- Wiener–Hunt: Hunt, IEEE Trans. Audio Electroacoustics 19 (1971), 285–288;
  [scikit-image v0.25.2 restoration/deconvolution.py wiener](https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/restoration/deconvolution.py)
  and [uft.py ir2tf/laplacian](https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/restoration/uft.py)
  (BSD-3-Clause). Same periodic quadratic model, full NumPy FFT, explicit origin,
  unit-spacing Laplacian eigenvalues. Unlike its image pipeline defaults, no
  output clipping. Null unpenalized modes use the minimum-norm solution.
- Richardson (1972), JOSA 62, 55–59; Lucy (1974), Astron. J. 79, 745–754.
  Read scikit-image v0.25.2 richardson_lucy in the same source. Project EM retains
  H*1 sensitivity, handles explicit gain/background and periodic or cropped zero
  boundaries. It does not assume unit PSF sum or symmetric kernels. Default
  constant .5 initialization is explicit; no clipping. Epsilon stabilizes the
  mean in division and log; convergence uses relative object change.

Production execution uses local solvers plus NumPy/PyTorch/torchkbnufft, not
SigPy or scikit-image reconstruction calls. Independent tests use direct DFT,
spatial convolution sums, dense quadratic systems and EM formulas.

## Calibration and regularization

Equation-based Python implementations; no reference repository or upstream solver
is vendored or used by production. Inspected sources on 2026-11-06:

- Griswold et al., MRM 47 (2002), 1202-1210, GRAPPA;
  [pygrappa grappa.py, master](https://github.com/mckib2/pygrappa/blob/master/pygrappa/grappa.py)
  (MIT): sampling-pattern grouping and relative ridge equations. This project
  requires a separate acquired mask, uses valid ACS rows without padded fake
  calibration samples, augmented least squares and zero-padded outer-grid sources.
  The viewed master source is a reading reference, not a pinned runtime dependency.
- Uecker et al., MRM 71 (2014), 990-1001, ESPIRiT;
  [SigPy v0.1.27 EspiritCalib](https://github.com/mikgroup/sigpy/blob/v0.1.27/sigpy/mri/app.py)
  (BSD-3-Clause): kernel SVD, Fourier covariance and crop. Local code uses dense
  eigh instead of power iteration, one map set and strongest-coil phase gauge.
- Lustig et al., MRM 58 (2007), 1182-1195, sparse MRI; same SigPy version
  L1WaveletRecon/TotalVariationRecon for weighted encoding and objective reference.
  Local CS uses complex ADMM/CG, orthogonal Haar with all coefficients penalized,
  and isotropic periodic complex TV (different from SigPy's anisotropic TV).
- Getreuer, IPOL 2 (2012), 158-174,
  [TV deconvolution](https://www.ipol.im/pub/art/2012/g-tvdc/),
  DOI 10.5201/ipol.2012.g-tvdc. Local solver is explicitly tv_admm: three splits
  for convolution, gradient and object constraints; Gaussian and Poisson proximal
  equations differ. Not a claim of reproducing the exact IPOL C update sequence.
- Dey et al., Microscopy Research and Technique 69 (2006), 260-266,
  [DOI 10.1002/jemt.20294](https://doi.org/10.1002/jemt.20294),
  [author publication record](https://www-sop.inria.fr/ariana/en/publications.php?name=Dey).
  Local RL-TV uses the multiplicative denominator with minus divergence, a declared
  smooth discrete isotropic TV, sensitivity H*1 and constant background extension.
  The author's linked report could not be retrieved in this session; no claim of
  source-code or voxelwise reproduction is made. The update and derivative are
  independently checked numerically; no third-party Torch reconstruction is used.
- Krishnan & Fergus, NeurIPS 2009, Fast Image Deconvolution using Hyper-Laplacian
  Priors; [author project](https://dilipkay.wordpress.com/fast-deconvolution/),
  [paper equations 1-6 and Algorithm 1](https://proceedings.neurips.cc/paper_files/paper/2009/file/3dd48ab31d016ffcbf3314df2b3cb9ce-Paper.pdf).
  Local HQS uses exact global scalar selection via bracketed root plus zero
  comparison instead of MATLAB LUT; FFT periodic image update and explicit finite
  continuation schedule. MATLAB execution/trajectory equivalence is not claimed.

No new dependencies were needed: NumPy FFT, SVD/eigh and the existing NUFFT extra
suffice. Numerical tests use independent scalar optimization, closed-form full
sampling solutions, discrete optimality equations and phase-invariant projectors.

## Learned MRI and blind deconvolution (2026-11-06)

- Sriram et al., End-to-End Variational Networks for Accelerated MRI
  Reconstruction (MICCAI 2020); facebookresearch/fastMRI archived main,
  fastmri/models/varnet.py and unet.py, retrieved 2026-11-06. Only those necessary
  network files are adapted under MIT (THIRD_PARTY_NOTICES.md). Local complex
  arithmetic/centered FFT replaces fastmri imports; normalization/power zero
  floors and pooling-dependent padding are explicit numerical changes.
  Strict state layout and all four named network classes are retained.
  Official preparation source:
  https://github.com/facebookresearch/fastMRI/blob/main/fastmri_examples/varnet/run_pretrained_varnet_inference.py
  points to https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/ and
  knee_leaderboard_state_dict.pt / brain_leaderboard_state_dict.pt. Official
  structure: 12 cascades, chans=18, pools=4, sens_chans=8, sens_pools=4.
  Knee weights were explicitly prepared in outputs/weights; brain not downloaded.
- Pan, Sun, Pfister, Yang, Blind Image Deblurring Using Dark Channel Prior,
  CVPR 2016, equations 10--16:
  https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Pan_Blind_Image_Deblurring_CVPR_2016_paper.pdf
  Author page: https://jspan.github.io/projects/dark-channel-deblur/index.html
  Readable ancillary kernel/alternation reference:
  https://github.com/FWen/deblur-pmp/blob/master/blind_deconv_main.m
  (Pan-attributed code, modified PMP context; not treated as the DCP solver).
  Python independently implements min-index selection, its exact transpose,
  hard thresholds and frozen-selection quadratic CG. It does not wrap that script.
- Xu, Zheng, Jia, Unnatural L0 Sparse Representation for Natural Image
  Deblurring, CVPR 2013, equations 5, 9--12, 15 and Algorithm 1:
  https://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Xu_Unnatural_L0_Sparse_2013_CVPR_paper.pdf
  Author project: https://www.cse.cuhk.edu.hk/leojia/projects/l0deblur/
  Public software is executable; no claim to source-level comparison. The paper
  fully specifies the implemented uniform-blur near-L0 model and epsilon schedule.
- Both blind implementations use periodic boundaries, an antialiased centered
  bilinear pyramid and compact nonnegative ridge LS in place of full-grid FFT
  kernel fitting/cropping. Final recovery reuses local Hyper-Laplacian p=.5 HQS;
  this is explicitly a DCP final-recovery variant. See domain algorithm docs.

## Standard lightweight benchmark inputs

Prepared 2026-11-07. See docs/public_benchmarks.md for provenance, units,
licenses and limitations; outputs/public_benchmarks/ for measured outputs.
scikit-image 0.26.0 camera/astronaut/cell are public source images, while their
blur/noise observations are synthesized locally. Levin09.mat is obtained from
https://github.com/cszn/DPIR/tree/master/kernels, preserving eight measured PSFs.
The ideal Airy PSF is a local scalar diffraction model, not Gibson-Lanni.
phantominator 0.7.0 provides MR M0/T1/T2; the local Biot-Savart circular-loop
integration provides simulated receive maps. The analytic phantom is neither
BrainWeb nor a real fastMRI knee slice. Official fastMRI fixtures create random
mock acquisitions; authorized real slices remain external and local.

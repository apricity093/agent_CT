# 已支持的 CT 组件组合

本文是当前实现的组合支持表。算法入口保持 17 个；表中的支持不表示任意输入都能收敛，执行前仍须通过数据、参数和几何校验。

Q 表示 L2、已知方差 Gaussian NLL 或固定权重 WLS；T 表示零背景纯透射 Poisson NLL。

| Solver 入口 | 数据保真项 | Regularizer | 图像约束 | 适用噪声 |
| --- | --- | --- | --- | --- |
| `tikhonov`（CG） | Q | 恒等 Tikhonov、H1 | 无约束 | 无噪声、Gaussian、透射 Poisson（post-log WLS） |
| `cgls`、`lsqr` | Q | 无正则、恒等 Tikhonov、H1 | 扩展的加权/增广组合为无约束 | 无噪声、Gaussian、透射 Poisson（post-log WLS） |
| `fista`、`admm`、`pdhg` | Q | 无正则、恒等 Tikhonov、H1、TV isotropic/anisotropic、梯度 Huber、Haar detail-L1 | 无约束、非负、显式 box | 无噪声、Gaussian、透射 Poisson（post-log WLS） |
| `fista`、`admm`、`pdhg` | T | 同上 | 非负；可带公开的有限上界 | 纯透射 Poisson（原始 counts） |
| `tv_fista` | 既有 L2 路径 | TV | 保留历史 post-prox projection 语义 | 无噪声、Gaussian、透射 Poisson（log 域近似） |
| `constrained_pdhg`、`asd_pocs_tv` | 原始或 Gaussian 白化残差球 | TV | 非负 | 无噪声、Gaussian；透射 Poisson 使用 log 域残差球近似 |
| `mlem`、`osem` | 发射 Poisson | 无显式正则 | 原生非负域 | 发射 Poisson（原始 counts） |
| `fbp`、`fdk` | 合法 line/log 数据的直接重建 | 无显式正则 | 由直接方法定义 | FBP：无噪声、Gaussian、透射 Poisson（log 域）；FDK：当前登记无噪声 |
| `landweber`、`sirt`、`sart`、`os_sart` | 保留原有数据项与迭代 | 无新增通用正则组合 | 保留原有约束入口 | 无噪声、Gaussian、透射 Poisson（log 域近似） |

## 数据、算子与参数

- 二维 runtime 使用 `ParallelBeamRadon2D`；FDK 使用 regular circular cone 的 `ASTRAFDKOperator3D`，要求 cubic voxels。不能由上表推断三维通用迭代组合已接入。
- Gaussian NLL 需要公开 sigma；WLS 需要公开 weights 和 weight_source；T 需要原始整数 counts、正 I0、正 scale 及纯 Poisson 元数据。
- 零计数合法。若有限极小点尚未确立，透射问题需要正的恒等正则或公开有限上界；不从 truth 推断约束。
- 恒等 Tikhonov 与 H1 为二次惩罚；Huber 的 delta 使用图像差分单位；Haar 只惩罚细节，默认 level=2，尺寸须被 `2**level` 整除。
- 约束 TV/Haar 使用联合近端；不能把无约束 prox 后 clamp 视为通用联合 prox。
- 发射和透射 Poisson 分开。PL-EM、MM、增量 proximal SART 是理论可行但本轮未实现的路线。

## 实现与精度约定
The 17 algorithm IDs remain unchanged. `cgls` and `lsqr` support weighted L2,
known-variance Gaussian NLL and supplied WLS with no penalty, identity Tikhonov
or H1 through a weighted augmented system. These new Krylov combinations are
unconstrained. LSQR damping represents an identity strength of `damping**2`;
explicit augmentation and damping cannot both be active. Rank-deficient least
squares is allowed without a uniqueness claim. Regularized CG requires
`ker(B) intersect ker(L) = {0}` for a positive-definite normal equation.

`fista`, `admm` and `pdhg` support the same quadratic fidelities and zero-background
transmission Poisson NLL with `none`, `tikhonov`, `h1`, `tv`, `huber` or `haar_l1`.
TV has isotropic and anisotropic modes. A zero strength selects `none` and skips
regularizer transforms. Quadratic problems allow no constraint, nonnegativity or
an explicit box. Transmission NLL requires a nonnegative lower bound of zero,
raw integer `counts`, public `I0 > 0` and public `scale > 0`. Zero counts are legal;
when a finite minimizer is not established, a positive identity penalty or a
public finite upper bound is required. No upper bound is inferred from truth.

FISTA handles H1/Huber in the smooth gradient. Its accepted step must satisfy the
actual smooth majorizer; Poisson extrapolation restarts if it leaves the
nonnegative image domain. Identity and box have a closed-form joint prox.
Generic TV and constrained Haar use dual FGP denoising with a primal-dual gap,
including projection inside the dual iteration. Their distance error target is
`tv_tolerance * max(norm(v), 1) / k**2.1`. `tv_num_iterations` also caps this Haar
inner solve. Insufficient precision reports `joint_prox_inner_max_iterations`;
it is not convergence. The legacy `tv_fista` entry retains its post-prox box
projection semantics and is identified separately.

ADMM uses the two blocks `x` and `(u,z,q)` with `u=Ax`, `z=Tx`, `q=x`; the
observation prox covers quadratic and Poisson terms. The x step uses CG with
relative target `cg_tolerance / k**2`, capped by `cg_max_iterations`. PDHG uses
a Schur upper bound for `tau * sigma * norm(K)**2 < 1`, rather than treating a
finite power estimate as a proven upper bound. Poisson prox diagnostics check
the optimality residual after conversion to the returned dtype and include its
roundoff allowance.

`constrained_pdhg` and `asd_pocs_tv` accept `residual_metric: gaussian_whitened`
with public known `sigma`; both data updates and the residual ball then use
`Sigma**(-1/2) A`. Epsilon is dimensionless in this mode. The helper
`gaussian_whitened_radius(m, alpha=0.01)` supplies the conservative chi-square
radius. Raw residuals and whitened constraint diagnostics are recorded separately.
ASD remains a feasibility/backtracking heuristic; feasible stagnation is not a
global optimization certificate.

Projection arrays, counts, sigma and fixed weights stay in the case public
measurement fields. Gaussian NLL uses `1/sigma**2`; pure Poisson post-log WLS
uses the public fixed count-floor rule and remains an approximation. Mixed-noise
support is retained without a new likelihood. Penalized EM, MM and incremental
proximal SART are mathematically feasible extensions outside this implementation.

ASTRA FDK now owns a copy of its projection working buffer because `FDK_CUDA`
filters GPU-linked storage in place. It preserves the caller's observations.
GPU-link and CPU-copy reconstruction differences remain a separate backend
limitation; observation preservation does not assert equality of these paths.

The numerical references used for these interfaces are the
[LSQR formulation](https://web.stanford.edu/group/SOL/software/lsqr/) and
[inexact proximal-gradient analysis](https://arxiv.org/abs/1109.2415).

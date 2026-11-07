# CT algorithms

The registry retains 17 entries: fbp, fdk, sirt, landweber, cgls, lsqr, sart,
os_sart, mlem, osem, tikhonov, tv_fista, fista, admm, pdhg,
constrained_pdhg and asd_pocs_tv.

`core/operator.py` contains linear transforms and adjoints; `core/fidelity.py`
contains measurement objectives; `core/regularizer.py` contains priors and proximal
maps. Solvers are grouped into analytic, algebraic, subset, statistical and
splitting modules. DIP, INR and DPS remain in `solvers/deep`.

FISTA uses an operator-norm estimate to initialize its step and checks the smooth
majorizer by backtracking. PDHG uses a conservative joint-operator bound.
Transmission likelihoods retain logarithm/exponential stability bounds. Ordered
subsets retain projection-angle scheduling. The public registry and algorithm
YAML files define executable combinations and defaults.

## 组件组合

Q 表示 L2、已知方差 Gaussian NLL 或固定权重 WLS；T 表示零背景纯透射 Poisson NLL。

| Solver 入口 | 数据保真项 | Regularizer | 图像约束 | 适用噪声 |
| --- | --- | --- | --- | --- |
| `tikhonov`（CG） | Q | 恒等 Tikhonov、H1 | 无约束 | 无噪声、Gaussian、透射 Poisson（post-log WLS） |
| `cgls`、`lsqr` | Q | 无正则、恒等 Tikhonov、H1 | 扩展的加权/增广组合为无约束 | 无噪声、Gaussian、透射 Poisson（post-log WLS） |
| `fista`、`admm`、`pdhg` | Q | 无正则、恒等 Tikhonov、H1、TV isotropic/anisotropic、梯度 Huber、Haar detail-L1 | 无约束、非负、显式 box | 无噪声、Gaussian、透射 Poisson（post-log WLS） |
| `fista`、`admm`、`pdhg` | T | 同上 | 非负；可带公开的有限上界 | 纯透射 Poisson（原始 counts） |
| `tv_fista` | 既有 L2 路径 | TV | post-prox projection | 无噪声、Gaussian、透射 Poisson（log 域近似） |
| `constrained_pdhg`、`asd_pocs_tv` | 原始或 Gaussian 白化残差球 | TV | 非负 | 无噪声、Gaussian；透射 Poisson 使用 log 域残差球近似 |
| `mlem`、`osem` | 发射 Poisson | 无显式正则 | 原生非负域 | 发射 Poisson（原始 counts） |
| `fbp`、`fdk` | 合法 line/log 数据的直接重建 | 无显式正则 | 由直接方法定义 | FBP：无噪声、Gaussian、透射 Poisson（log 域）；FDK：当前登记无噪声 |
| `landweber`、`sirt`、`sart`、`os_sart` | 使用数据项与迭代 | 无通用正则组合 | 使用约束入口 | 无噪声、Gaussian、透射 Poisson（log 域近似） |

## 数据、算子与参数

- 二维 runtime 使用 `ParallelBeamRadon2D`；FDK 使用 regular circular cone 的 `ASTRAFDKOperator3D`，要求 cubic voxels。不能由上表推断三维通用迭代组合已接入。
- Gaussian NLL 需要公开 sigma；WLS 需要公开 weights 和 weight_source；T 需要原始整数 counts、正 I0、正 scale 及纯 Poisson 元数据。
- 零计数合法。若有限极小点尚未确立，透射问题需要正的恒等正则或公开有限上界；不从 truth 推断约束。
- 恒等 Tikhonov 与 H1 为二次惩罚；Huber 的 delta 使用图像差分单位；Haar 只惩罚细节，默认 level=2，尺寸须被 `2**level` 整除。
- 约束 TV/Haar 使用联合近端；不能把无约束 prox 后 clamp 视为通用联合 prox。

## 实现与精度约定
The 17 algorithm IDs remain unchanged. `cgls` and `lsqr` support weighted L2,
known-variance Gaussian NLL and supplied WLS with no penalty, identity Tikhonov
or H1 through a weighted augmented system. These Krylov combinations are
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
it is not convergence. The `tv_fista` entry uses post-prox box
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
support is retained without a new likelihood.

ASTRA FDK owns a copy of its projection working buffer because `FDK_CUDA`
filters GPU-linked storage in place. It preserves the caller's observations.
GPU-link and CPU-copy reconstruction differences remain a separate backend
limitation; observation preservation does not assert equality of these paths.

The numerical references used for these interfaces are the
[LSQR formulation](https://web.stanford.edu/group/SOL/software/lsqr/) and
[inexact proximal-gradient analysis](https://arxiv.org/abs/1109.2415).

## 保真项与离散变换

Gaussian NLL 使用公开 sigma 的 `1/sigma**2` 权重。常量 sigma 下，
`lambda_NLL=lambda_L2/sigma**2`；异方差没有统一标量换算。
WLS 使用公开固定权重，post-log Poisson 权重是近似。透射 NLL 使用原始
counts、I0 和 scale；发射 MLEM/OSEM 使用另一观测模型，不能混用。
不同保真项的目标值不能直接作为质量排名。

H1 为 `0.5||Dx||²`，梯度为 `D*Dx`。D 使用单位网格前向差分，
垂直末行、水平末列为零；伴随仅累加有效差分边。
Haar 为二维正交变换，各方向按 `1/sqrt(2)` 归一化，层内布局为
`[[LL,LH],[HL,HH]]`，只对 LL 递归，仅惩罚细节。图像尺寸须被
`2**level` 整除，变换不补零或裁剪。

参数 `strength` 只施加一次；API 支持的 `lam` 等别名见
[任务契约](agent_algorithm_api.md)。具体默认值从 `list_capabilities()` 读取。

## 约束 TV

`constrained_pdhg` 与 `asd_pocs_tv` 求解残差球内的非负 TV 重建：
`min TV(x), ||M(Ax-y)||₂ <= epsilon, x >= 0`。
输入为二维方形平行束线积分、单 batch/channel 和非空二值 mask。
`epsilon` 与 `epsilon_source` 明确给出半径和来源；Gaussian 白化模式使用
公开 sigma，半径无量纲。透射计数先转换到 log 域，此时残差球为近似模型。

```python
from inv_framework.api import CTTaskRequest, run_ct_task

result = run_ct_task(CTTaskRequest(
    case_id="parallel_2d__ellipses_sparse_gaussian",
    algorithm="constrained_pdhg",
    params={"epsilon": 0.2, "epsilon_source": "explicit observation-unit radius",
            "max_iter": 20},
))
print(result.convergence_status, result.stopping_reason)
```

示例半径用于演示接口，实际任务应按观测单位和噪声设置。
PDHG 收敛要求同时满足可行性与 primal-dual fixed-point 判据；ASD 是
SART2D 可行性回溯启发式，停滞不构成全局最优证书。
精度和后端限制见[验证说明](validation.md)。

## Constrained TV sources and variants

Mathematical source: Sidky and Pan, Physics in Medicine and Biology 53 (2008),
4777–4807, DOI [10.1088/0031-9155/53/17/021](https://doi.org/10.1088/0031-9155/53/17/021),
section 2.4.1 ([PMC article](https://pmc.ncbi.nlm.nih.gov/articles/PMC2630711/)).
The original uses ray-wise ART, positivity and adaptive normalized TV descent.
It solves a residual-tolerance constrained TV problem. Intermediate TV steps
can leave the feasible set; subsequent data updates restore consistency.
Local feasibility-preserving backtracking changes that trajectory, and therefore
has no inherited global optimality guarantee.

Engineering reference: CERN TIGRE **v2.6**, commit
`dfa6c6522aeaa0f2576e0c8500ad7deedfc1e64f`,
[Python/tigre/algorithms/pocs_algorithms.py](https://github.com/CERN/TIGRE/blob/dfa6c6522aeaa0f2576e0c8500ad7deedfc1e64f/Python/tigre/algorithms/pocs_algorithms.py),
`initASD_POCS.__init__`, `run_main_iter`, `ASD_POCS.__init__`.
The matching upstream MATLAB `ASD_POCS.m` and `LICENSE.txt` are available at the same pinned revision. TIGRE uses BSD-3-Clause, copyright University of Bath and CERN (2015).
Production code uses the existing local PyTorch operators; TIGRE is not a
runtime dependency and no TIGRE CUDA kernels are incorporated.

| Update | TIGRE v2.6 | Local `sart2d_feasibility_backtracking_v1` |
| --- | --- | --- |
| Data sweep | SART, ASD forces blocksize=1 | ordered angle blocks, reciprocal row/column normalization, nonnegativity after each block; block_size>1 is a block-SART variant |
| Relaxation | beta from lmbda, multiplied by lmbda_red | relaxation=1, relaxation_decay=.99 per complete sweep |
| Initial TV step | alpha times first data displacement, alpha=.002 | tv_step=.2 times max(first displacement, tolerance times image norm); explicit feasible-start fallback |
| TV iterations | tviter=20, minimizeTV | tv_updates=20 normalized local TV subgradient steps, each projected nonnegative |
| Adaptation | reduce dtvg by alpha_red=.95 when TV displacement exceeds rmax=.95 times data displacement and residual exceeds epsilon | bounded 40-trial backtracking by tv_step_decay=.95; enforce displacement ratio update_ratio=.95, TV nonincrease, and preserve feasibility once the data point is feasible |
| Radius | explicit maxl2err or FDK-derived default | mandatory epsilon and epsilon_source in raw line-integral units; no FDK calibration |
| Stop | displacement angle, beta<.005, or niter | resolved SolveControl min/check/patience; feasible iterate stagnation is stalled, never global convergence; iteration limits remain distinct stopping outcomes |

Local unit-grid TV uses forward differences with zero terminal differences,
with explicit isotropic/anisotropic modes. The implementation is a two-dimensional
SART-based ASD-POCS adaptation, not a numerically identical port of TIGRE or the
original ART algorithm. The variant identity and absence of a global optimality guarantee are preserved. Parameters above are engineering initializations, not quality-tuned defaults.

PDHG independently solves the convex problem using K=[A;D], residual-ball
conjugate proximal, TV dual-ball projection and nonnegative primal projection.
The Schur norm bound of A plus 8 bounds ||K||²; tau=sigma=.99/sqrt(bound)
satisfies tau*sigma*||K||²<1. Convergence requires feasible returned x and the
joint primal/dual fixed-point residual below resolved tolerance over the
requested checked patience window. This certificate is distinct from ASD's
feasible stagnation, and epsilon is distinct from optimization tolerance.

### TIGRE reference license

BSD 3-Clause License

Copyright (c) 2015, University of Bath and CERN- European Organization for  Nuclear Research
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

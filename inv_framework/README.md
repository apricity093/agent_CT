# inv_framework 目录说明

本文档说明当前 `inv_framework/` 目录结构、核心接口、算子、正则项，以及已经放入框架内的各类 CT solver。文中行号基于当前工作区代码版本；如果后续继续修改源码，请用 `rg -n "^(class|def) " inv_framework` 重新刷新行号。

## 目录结构

```text
inv_framework/
├── __init__.py
├── operators/
│   ├── __init__.py
│   ├── base.py
│   ├── noise.py
│   └── ct/
│       ├── __init__.py
│       ├── radon_torch.py
│       ├── astra_adapter.py      # upstream path when merged into original repo
│       └── astra_3d.py           # local fallback placeholder
├── regularizers/
│   ├── __init__.py
│   ├── base.py
│   ├── tikhonov.py
│   └── tv.py
└── solvers/
    ├── __init__.py
    ├── base.py
    ├── _utils.py
    ├── classical.py
    ├── subset.py
    ├── regularized.py
    └── statistical.py
```

## 模块职责

| 路径 | 作用 |
|---|---|
| `operators/base.py` | 定义 `ForwardOperator` 和 `LinearOperator`，是所有正向算子/线性算子的框架接口。 |
| `operators/noise.py` | 定义噪声模型，包括无噪声、高斯噪声、Poisson log-domain CT 噪声。 |
| `operators/ct/radon_torch.py` | 纯 PyTorch 2D parallel-beam Radon 算子，支持 forward、adjoint、subset。 |
| `operators/ct/astra_adapter.py` | 原仓库 ASTRA CUDA3D backend 路径；合入原仓库时应优先保留该文件。 |
| `operators/ct/astra_3d.py` | 当前本地 fallback 占位接口，用于未来 cone-beam / 3D CT backend。 |
| `regularizers/base.py` | 定义正则项和 proximal operator 抽象。 |
| `regularizers/tikhonov.py` | 定义恒等或广义线性正则算子下的二次 Tikhonov 正则项。 |
| `regularizers/tv.py` | 定义二维 isotropic/anisotropic TV 及纯 PyTorch 双变量 FGP proximal。 |
| `solvers/base.py` | 定义统一 solver 基类 `InverseProblemSolver`。 |
| `solvers/_utils.py` | solver 共用 shape、batch、device、dtype、`x_init` 校验工具。 |
| `solvers/classical.py` | FBP、SIRT、Landweber、CGLS、LSQR、FDK 等经典算法。 |
| `solvers/subset.py` | SART、OS-SART 等子集迭代算法。 |
| `solvers/regularized.py` | Tikhonov 正规方程共轭梯度和 TV-FISTA 正则化重建。 |
| `solvers/statistical.py` | MLEM、OSEM 等统计重建算法。 |
| `solvers/__init__.py` | 汇总导出公开 solver 函数和 solver 类。 |

## 核心接口位置

| 接口 | 文件 | 行号 | 说明 |
|---|---|---:|---|
| `ForwardOperator` | `operators/base.py` | 14 | 抽象正向模型，要求 `forward(x)`。 |
| `LinearOperator` | `operators/base.py` | 36 | 线性算子，额外要求 `adjoint(y)`；`pseudo_inverse()` 默认调用 Landweber。 |
| `InverseProblemSolver` | `solvers/base.py` | 10 | 统一 solver 基类，要求 `solve(measurement, operator, **kwargs)`。 |
| `Regularizer` | `regularizers/base.py` | 8 | 正则项抽象，要求 `value(x)`。 |
| `ProximalOperator` | `regularizers/base.py` | 16 | proximal 算子抽象，要求 `proximal(x, step_size)`。 |

## 算子与辅助组件位置

| 组件 | 文件 | 行号 | 说明 |
|---|---|---:|---|
| `ParallelBeamRadon2D` | `operators/ct/radon_torch.py` | 86 | 2D parallel-beam Radon `LinearOperator`，提供 `forward`、`adjoint`、`pseudo_inverse`、`subset`。 |
| `_RadonFunction` | `operators/ct/radon_torch.py` | 58 | Radon forward 的自定义 autograd function。 |
| `_RadonAdjointFunction` | `operators/ct/radon_torch.py` | 72 | Radon adjoint 的自定义 autograd function。 |
| `ASTRAOperator3D` | `operators/ct/astra_adapter.py` 或本地 fallback `operators/ct/astra_3d.py:26` | 26 | 原仓库真实 ASTRA backend 应来自 `astra_adapter.py`；本地 `astra_3d.py` 只是占位 fallback。 |
| `NoiseModel` | `operators/noise.py` | 10 | 噪声模型基类。 |
| `NoNoise` | `operators/noise.py` | 18 | 恒等噪声模型。 |
| `GaussianNoise` | `operators/noise.py` | 25 | 加性高斯噪声。 |
| `PoissonLogDomainNoise` | `operators/noise.py` | 51 | CT log-domain Poisson photon noise。 |
| `TikhonovRegularizer` | `regularizers/tikhonov.py` | 12 | 定义 `0.5 ||Lx||^2`，并提供 `L^T Lx`；默认 `L=I`。 |
| `TVRegularizer` | `regularizers/tv.py` | 11 | TV 正则项，提供 `value()` 和双变量 FGP `proximal()`。 |

## 共用 solver 校验工具

| 工具 | 文件 | 行号 | 说明 |
|---|---|---:|---|
| `expected_batch_shape` | `solvers/_utils.py` | 10 | 从 batch size 和单样本 shape 生成 batched shape。 |
| `validate_measurement` | `solvers/_utils.py` | 15 | 检查 `measurement` 是否为 `(B, *operator.range_shape)`。 |
| `initial_reconstruction` | `solvers/_utils.py` | 36 | 创建或校验 `x_init`，并迁移到 measurement 的 device/dtype。 |
| `_require_linear` | `solvers/classical.py` | 14 | 需要 adjoint 的 solver 统一检查 `LinearOperator`。 |

## 算法对应位置

下表列出每个算法在 `inv_framework` 内的实现位置。“裸函数”是实际计算入口，“Solver 类”是符合 `InverseProblemSolver` 的统一调用封装。

### Classical solvers

| 算法 | 原始参考来源 | 裸函数位置 | Solver 类位置 | 需要 `LinearOperator` | 当前说明 |
|---|---|---|---|---|---|
| FBP | TIGRE `fbp`、Skimage `iradon`、Tomopy `fbp` | `solvers/classical.py:78` `fbp()` | `solvers/classical.py:267` `FBPSolver` | Yes | 2D tensor FBP，支持多种 filter；完整 TIGRE/Tomopy 几何保真仍依赖 operator。 |
| SIRT | CIL `SIRT`，参考 TIGRE `sirt` | `solvers/classical.py:96` `sirt()` | `solvers/classical.py:287` `SIRTSolver` | Yes | 保留 row/column normalization、relaxation、clamp；CIL 通用 proximal constraint 未完整迁移。 |
| Landweber | 经典 Landweber；`LinearOperator.pseudo_inverse()` 默认方法 | `solvers/classical.py:126` `landweber()` | `solvers/classical.py:315` `LandweberSolver` | Yes | 支持 batch、`x_init`、device/dtype、min/max clamp。 |
| **CGLS** | `algorithms/cil/CGLS.py` | `solvers/classical.py:149` `cgls()` | `solvers/classical.py:334` `CGLSSolver` | Yes | 保留 CIL CGLS 的 residual、`A^T r`、`alpha/beta` 更新。 |
| **LSQR** | `algorithms/cil/LSQR.py` | `solvers/classical.py:183` `lsqr()` | `solvers/classical.py:351` `LSQRSolver` | Yes | 保留 Golub-Kahan bidiagonalization，并支持可选 `reg_alpha`。 |
| **FDK** | TIGRE `FDK` | `solvers/classical.py:244` `fdk()` | `solvers/classical.py:388` `FDKSolver` | Yes | backend-gated；要求 operator 提供 `fdk(y, **kwargs)`，无 backend 时抛 `NotImplementedError`。 |

### Subset solvers

| 算法 | 原始参考来源 | 裸函数位置 | Solver 类位置 | 需要 `LinearOperator` | 当前说明 |
|---|---|---|---|---|---|
| SART | TIGRE `SART`、Skimage `iradon_sart` | `solvers/subset.py:54` `sart()` | `solvers/subset.py:122` `SARTSolver` | Yes | 使用 `operator.subset(indices)`；支持显式 subset、ordered/random、relaxation、clamp。 |
| OS-SART | TIGRE `OS_SART` | `solvers/subset.py:93` `ossart()` | `solvers/subset.py:164` `OSSARTSolver` | Yes | 基于 SART 子集循环；支持 block size、显式 subset、ordered/random。 |

### Regularized solvers

| 算法 | 原始参考来源 | 裸函数位置 | Solver 类位置 | 需要 `LinearOperator` | 当前说明 |
|---|---|---|---|---|---|
| **Tikhonov** | 经典 Tikhonov；参考 TomoPy `tikh` 和 CIL 二次正则建模 | `solvers/regularized.py` `tikhonov()` | `solvers/regularized.py` `TikhonovSolver` | Yes | 求解 `0.5||Ax-y||^2 + 0.5 lambda ||Lx||^2`；默认 `L=I`，可传额外 `LinearOperator` 表达广义 Tikhonov。 |
| **TV-FISTA** | CIL FISTA/TotalVariation，参考 TIGRE FISTA 控制流程 | `solvers/regularized.py` `tv_fista()` | `solvers/regularized.py` `TVFISTASolver` | Yes | 外层 FISTA；内层为纯 PyTorch 2D dual FGP TV proximal；不宣称等价于 TIGRE CUDA `im3ddenoise` 或 TomoPy C `tv`。 |

### Statistical solvers

| 算法 | 原始参考来源 | 裸函数位置 | Solver 类位置 | 需要 `LinearOperator` | 当前说明 |
|---|---|---|---|---|---|
| **MLEM** | Tomopy `recon(..., algorithm="mlem")` | `solvers/statistical.py:49` `mlem()` | `solvers/statistical.py:159` `MLEMSolver` | Yes | 正初值、sensitivity、ratio correction、nonnegative clamp；默认 `initial_value=1e-6`。 |
| **OSEM** | Tomopy `recon(..., algorithm="osem")` | `solvers/statistical.py:96` `osem()` | `solvers/statistical.py:206` `OSEMSolver` | Yes | 使用 `operator.subset(indices)`；支持 `subset_indices`、block size、per-subset sensitivity。 |

## 公开导出

公开 solver 和函数在 `solvers/__init__.py` 中集中导出。当前导出的算法包括：

- Classical: `fbp`, `sirt`, `landweber`, `cgls`, `lsqr`, `fdk`
- Classical solver classes: `FBPSolver`, `SIRTSolver`, `LandweberSolver`, `CGLSSolver`, `LSQRSolver`, `FDKSolver`
- Subset: `sart`, `ossart`, `SARTSolver`, `OSSARTSolver`
- Regularized: `tikhonov`, `tv_fista`, `TikhonovSolver`, `TVFISTASolver`
- Statistical: `mlem`, `osem`, `MLEMSolver`, `OSEMSolver`

## 尚未完整保真适配的算法

以下算法在迁移矩阵或审计文档中已经记录，但当前 `inv_framework` 目录内未实现完整保真版本：

| 原始算法 | 当前状态 | 原因 |
|---|---|---|
| TIGRE `OS_SART_TV` | 未实现 | 需要 TIGRE 等价 `im3ddenoise(tviter, tvlambda)` backend。 |
| TIGRE `AwASD_POCS` | 有占位 wrapper，未实现内核 | 需要 weighted TV / `AwminTV` backend。 |
| Tomopy `art` / `bart` | 未实现 | 需要 row-wise / block-wise algebraic update。 |
| Tomopy `gridrec` | 未实现 | Fourier grid reconstruction 需要专用 backend，不能仅靠 `forward/adjoint` 保真表达。 |
| Tomopy `ospml_*` / `pml_*` | 未实现 | 需要 penalized likelihood 正则模型。 |
| Tomopy `tv` / `tikh` 的 C backend 数值等价版本 | 未实现 | 当前已有框架原生 Tikhonov 与 TV-FISTA，但 Tomopy 的几何、插值及 `reg_par` / `reg_data` 精确语义仍需专门 backend。 |

## 相关说明文档

仓库根目录下还有三份辅助文档：

- `MIGRATION_MATRIX.md`：原始算法到目标 solver 的保真迁移矩阵。
- `TOMOPY_AUDIT.md`：Tomopy `recon(..., algorithm=...)` 入口逐项审计。
- `BACKEND_BOUNDARY.md`：FBP / FDK / ASTRA backend 边界说明。

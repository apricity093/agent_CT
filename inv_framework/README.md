# inv_framework 目录说明

通用 FISTA、H1-CG 与 Haar-L1 的接口、数学条件和预算见 [使用说明](../docs/composable_reconstruction.md)。旧 `tv_fista` 与 Tikhonov-CG 保留；正式清单只新增一个 `fista` solver，不按正则组合重复注册。

本文档说明当前 `inv_framework/` 目录结构、核心接口、算子、正则项，以及已经放入框架内的各类 CT solver。文中行号基于当前工作区代码版本；如果后续继续修改源码，请用 `rg -n "^(class|def) " inv_framework` 重新刷新行号。

## 目录结构

```text
inv_framework/
├── __init__.py
├── operators/
│   ├── __init__.py
│   ├── base.py
│   ├── noise.py
│   ├── gradient.py
│   ├── wavelet.py
│   └── ct/
│       ├── __init__.py
│       ├── radon_torch.py
│       ├── astra_adapter.py      # 冻结的 ASTRA 3D 投影适配器
│       ├── astra_fdk_adapter.py  # ASTRA FP3D/BP3D/FDK_CUDA 完整后端
│       └── leap_adapter.py       # LEAP project/backproject/FBP 适配器
├── regularizers/
│   ├── __init__.py
│   ├── base.py
│   ├── tikhonov.py
│   ├── wavelet.py
│   └── tv.py
└── solvers/
    ├── __init__.py
    ├── base.py
    ├── _utils.py
    ├── classical.py
    ├── subset.py
    ├── regularized.py
    ├── fista.py
    └── statistical.py
```

## 模块职责

| 路径 | 作用 |
|---|---|
| `operators/base.py` | 定义 `ForwardOperator` 和 `LinearOperator`，是所有正向算子/线性算子的框架接口。 |
| `operators/noise.py` | 定义噪声模型，包括无噪声、高斯噪声、Poisson log-domain CT 噪声。 |
| `operators/ct/radon_torch.py` | 纯 PyTorch 2D parallel-beam Radon 算子，forward 与离散转置 adjoint 配对；solver 工具按角度重建子算子。 |
| `operators/gradient.py` | 公共单位网格前向差分 D 与转置 D*，供 TV 及后续正则项复用。 |
| `operators/ct/astra_adapter.py` | 2026-06-08 冻结的 ASTRA CUDA3D 投影适配器。 |
| `operators/ct/astra_fdk_adapter.py` | `ASTRAFDKOperator3D`；通过 ASTRA `FP3D_CUDA`、`BP3D_CUDA` 和 `FDK_CUDA` 提供可执行的完整锥束流程。 |
| `operators/ct/leap_adapter.py` | `LEAPOperator3D`；包装已配置的 LEAP model，提供 batched PyTorch forward、adjoint 和 cone FDK。 |
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

| 接口 | 文件 | 说明 |
|---|---|---|
| `ForwardOperator` | `operators/base.py` | 抽象正向模型，要求 `forward(x)`。 |
| `LinearOperator` | `operators/base.py` | 线性算子，额外要求 `adjoint(y)`；`pseudo_inverse()` 默认调用 Landweber。 |
| `InverseProblemSolver` | `solvers/base.py` | 统一 solver 基类，要求 `solve(measurement, operator, **kwargs)`。 |
| `Regularizer` | `regularizers/base.py` | 正则项抽象，要求 `value(x)`。 |
| `ProximalOperator` | `regularizers/base.py` | proximal 算子抽象，要求 `proximal(x, step_size)`。 |

## 算子与辅助组件位置

| 组件 | 文件 | 说明 |
|---|---|---|
| `ParallelBeamRadon2D` | `operators/ct/radon_torch.py` | 2D parallel-beam Radon `LinearOperator`，提供 `forward`、`adjoint`、`pseudo_inverse`；子集由 `solvers/_utils.py` 和 instrumentation 创建。 |
| `_RadonFunction` | `operators/ct/radon_torch.py` | Radon forward 的自定义 autograd function。 |
| `_RadonAdjointFunction` | `operators/ct/radon_torch.py` | Radon adjoint 的自定义 autograd function。 |
| `ASTRAOperator3D` | `operators/ct/astra_adapter.py` | 冻结的 ASTRA 3D 投影适配器；不提供 `fdk()`。 |
| `ASTRAFDKOperator3D` | `operators/ct/astra_fdk_adapter.py` | 常规 circular `cone` + cubic voxel 门控；支持原生 Ram-Lak、Parker short-scan、voxel supersampling 和 CUDA Tensor 直连。 |
| `LEAPOperator3D` | `operators/ct/leap_adapter.py` | 包装 LEAP `tomographicModels`；内部 float32，返回原 dtype/device；model 有状态，单实例不可跨线程并发。 |
| `NoiseModel` | `operators/noise.py` | 噪声模型基类。 |
| `NoNoise` | `operators/noise.py` | 恒等噪声模型。 |
| `GaussianNoise` | `operators/noise.py` | 加性高斯噪声。 |
| `PoissonLogDomainNoise` | `operators/noise.py` | CT log-domain Poisson photon noise。 |
| `TikhonovRegularizer` | `regularizers/tikhonov.py` | 定义 `0.5 ||Lx||^2`，提供 `L^T Lx`；默认 `L=I`，此时整项近端为 `v/(1+t)`。 |
| `TVRegularizer` | `regularizers/tv.py` | TV 正则项，提供 `value()` 和双变量 FGP `proximal()`。 |

## 离散伴随与梯度边界

`ParallelBeamRadon2D.forward()` 使用与输入 dtype 一致的旋转网格、`align_corners=False`、双线性插值、零填充和逐行求和。`adjoint()` 使用同一网格的四邻点权重执行转置累加，不使用逆旋转插值，也不额外执行 forward/VJP。forward 的 backward 为 A*，adjoint 的 backward 为 A；batch/channel 独立，输出保持输入 dtype/device。CPU/CUDA 的 float32/float64 路径均有专项数值测试。

公共 `gradient(x)` 返回最后两维的 `(vertical, horizontal)`，垂直末行与水平末列为零。`gradient_adjoint(vertical, horizontal)` 忽略这两个无效 dual 边界，使任意同 shape/dtype/device 的 dual 张量满足内积恒等式。TV 复用该实现；各向同性/各向异性定义、`value()`、`proximal()` 和强度约定保持原语义，FGP 更新与停止逻辑保持现有实现。

伴随关系影响 FBP、迭代重建、子集归一化、范数估计及 autograd；独立矩阵与冻结短轨迹回归位于 tests/test_p0_operators.py、tests/test_p0_short_trajectories.py。Radon 网格随输入 dtype 构造；CPU/CUDA 与 float32/float64 的数值测试分别保留。

## 共用 solver 校验工具

| 工具 | 文件 | 说明 |
|---|---|---|
| `expected_batch_shape` | `solvers/_utils.py` | 从 batch size 和单样本 shape 生成 batched shape。 |
| `validate_measurement` | `solvers/_utils.py` | 检查 `measurement` 是否为 `(B, *operator.range_shape)`。 |
| `initial_reconstruction` | `solvers/_utils.py` | 创建或校验 `x_init`，并迁移到 measurement 的 device/dtype。 |
| `_require_linear` | `solvers/classical.py` | 需要 adjoint 的 solver 统一检查 `LinearOperator`。 |

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
| **FDK** | ASTRA `FDK_CUDA`；LEAP `FBP` cone path | `solvers/classical.py:292` `fdk()` | `solvers/classical.py:359` `FDKSolver` | Yes | `ASTRAFDKOperator3D` 提供 ASTRA 完整 weighting/filter/cone-backprojection；`LEAPOperator3D` 提供 LEAP FDK。普通 operator 无 backend 时仍抛 `NotImplementedError`。 |

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
- Regularized: `tikhonov`, `tv_fista`, `fista`, `TikhonovSolver`, `TVFISTASolver`, `FISTASolver`
- Statistical: `mlem`, `osem`, `MLEMSolver`, `OSEMSolver`

CT operator 公开导出包括 `ParallelBeamRadon2D`、`ASTRAOperator3D`、
`ASTRAFDKOperator3D` 和 `LEAPOperator3D`。ASTRA 来源仓库采用 GPL-3.0，
LEAP 来源仓库采用 MIT；两者均为可选的外部数值后端。

## FDK 后端能力边界

- `ASTRAFDKOperator3D` 只接受 ASTRA regular `cone` geometry 和 cubic voxels；不把 `parallel3d` 或任意 `cone_vec` 几何伪装成已支持。
- ASTRA adapter 使用官方 `FDK_CUDA` 完成 cone weighting、可选 Parker short-scan weighting、原生 Ram-Lak filter 和 cone backprojection；按 batch 逐样本执行，并在成功或异常时释放临时 algorithm/data IDs。其他 filter 会明确拒绝，不会把 ASTRA 2.2.0 忽略的选项伪装成生效。
- `LEAPOperator3D` 要求传入已配置 geometry 的 `leapctype.tomographicModels`；只有 `get_geometry()` 返回 `CONE` 时才把 `model.FBP(..., inplace=False)` 暴露为 FDK，其他 geometry 显式拒绝。LEAP model 有状态，同一 adapter 实例不保证线程安全。
- 两个 adapter 都保持 `(B, *range_shape) -> (B, *domain_shape)`、原始 dtype/device，以及 forward autograd backward 与显式 adjoint 的一致接口。

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

## FDK 公开来源

- ASTRA Toolbox: https://github.com/astra-toolbox/astra-toolbox （GPL-3.0）
- LLNL LEAP: https://github.com/LLNL/LEAP （MIT）

## 可组合重建

ADMM/PDHG 支持 I、H1、TV、分量梯度 Huber、Haar-L1 的无约束或非负组合，
其他 box 拒绝。FISTA 支持恒等二次项的 box，TV/Haar 仅无约束。详见
[分裂算法用法](../docs/splitting_solvers.md) 和
[二次/透射 NLL 保真项](../docs/data_fidelity.md)。

辅助 D/D*、Haar 和 coefficient proximal 单独记账；H1-CG 的广义算子预算还包含
D/D*，同时报告 ct_forward_calls/ct_adjoint_calls。CUDA wall 计时等待操作完成，
A/A* 另有 event 时间；这些字段不是整个 solver 的 kernel 时间。峰值内存区分
allocator allocated/reserved 与进程高水位。

[约束 TV API 与 CLI](../docs/constrained_tv.md) 使用独立 solver ID。
ASD 的来源和变体定义见 [来源说明](../docs/constrained_tv_sources.md)。预算中断只保留
最后完整点，未确认终点量为空。独立质量示例见 [质量示例用法](../docs/quality_example.md)，
Agent 参数、预算、安装与固定版本状态见 [Agent 用法](../docs/agent_usage.md)。

`benchmarks.showcase.prepare_cases(root, ct_repo)` 为 Agent 组织展示准备一个标准案例目录：
复用版本化的 128² 稀疏角组织透射数据，另外生成同组织的匹配高斯投影与发射计数，
以及 64³ 连续三维组织体的 ASTRA CUDA 锥束观测。`tissue_volume()` 是确定性的
三维纹理/椭球 phantom，不是患者数据。约束 TV 的 ε 仅由公开 σ 和有效样本数的
99% Gaussian L2 上界确定；匹配投影器案例用于流程展示，不构成独立投影器质量证据。
数据准备与求解器隔离；truth 仅用于生成观测和运行后的独立评价。

row-action 终点校验分别检查最后两个完整点的实际更新量和终点的固定点变化量。
前者严格核对轨迹标量，后者独立核对停止阈值；两者不要求数值相等。
原生提前收敛/停滞通过独立 endpoint 后，取代 legacy 原始残差的轨迹长度推断。

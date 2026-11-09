# Richardson–Lucy

## 一、算法身份

- **名称**：Richardson–Lucy。
- **一句话简介**：通过 Poisson 似然乘法更新恢复非负图像。
- **所属任务**：反卷积。
- **方法类型**：非学习类，无需训练权重。
- **适用范围**：RL-TV 是带平滑 TV 的 Dey 更新变体，在本家族说明；需满足其分母条件。
- **代码来源**：[apricity093/agent_CT](https://github.com/apricity093/agent_CT)，包版本`0.1.0`，使用随本说明发布的源码（[版本声明](../../pyproject.toml)）。主要实现：[inv_framework/deconvolution/solvers.py](../../inv_framework/deconvolution/solvers.py)。

### 领域入口与目标

| 领域／入口 | 适用范围与实际目标 |
| --- | --- |
| 反卷积 / `richardson_lucy` | Nonnegative Poisson model; matched adjoint and boundary sensitivity sum (gain Hx+b - y log(gain Hx+b)), x>=0; emission Poisson multiplicative update. |
| 反卷积 / `rl_tv` | Periodic 2D/3D smoothed Dey multiplicative RL-TV; Poisson gain=1; positive denominator required Emission Poisson NLL + lam sum sqrt(\|\|Dx\|\|_2^2+tv_epsilon^2); Dey multiplicative update requires positive denominator. |

## 二、输入输出

### 最小示例数据

tests/data/deconvolution/cases/public_camera_levin/public.npz：counts=(256,256)，单位为光子计数；gain=40、background=0.2。psf=(19,19)，原点[9,9]像素。camera图像已抗混叠缩放为256×256、uint8除255，采用周期卷积；图像轴为y、x，不在重建时重新归一化。

示例配置或生成脚本：[configs/deconvolution/benchmarks/standard_public_camera_levin_poisson_richardson_lucy.yaml](../../configs/deconvolution/benchmarks/standard_public_camera_levin_poisson_richardson_lucy.yaml)。

**正常输出**：reconstruction.npy (256,256) 和result.json；正常输出为有限图像。

NPY用`numpy.load`读取；tensor bundle用`torch.load(..., weights_only=True)`读取。结果同时记录执行状态、实际迭代和停止原因；预算结束与数值收敛分别解释。

### 其他领域入口

- 反卷积：[deconvolution输入契约](../deconvolution/data_format.md)，说明数组轴、几何/采样、单位及预处理。

## 三、运行方式

从算法库根目录执行。Python ≥3.10；基础依赖为PyTorch、NumPy、SciPy、h5py、PyYAML、matplotlib：

```bash
python -m pip install -e ".[dev]"
```

本例使用CPU。完整安装及Windows Conda设置见[依赖说明](../../DEPENDENCIES.md)。无需Agent SDK或大模型服务。

### 最小示例命令

```bash
python -m inv_framework.runner --config configs/deconvolution/benchmarks/standard_public_camera_levin_poisson_richardson_lucy.yaml --output outputs/algorithm_examples/richardson_lucy
```

每次运行选择新的输出目录。该命令中的配置可能覆盖部分默认值；实际值以结果中的effective_params为准。

## 四、参数

下面按入口列出参数，保留领域默认值及字段名差异。默认值由本仓库配置维护，属于工程运行起点；表中来源链接给出具体配置位置。显式参数覆盖实验配置，实验配置覆盖算法默认值。

建议范围不是硬边界；先固定观测尺度、离散定义和精度，再比较权重。步长、rho、内层预算以实际数值条件选择。

### 反卷积：`richardson_lucy`

默认值来源：[本仓库配置](../../configs/deconvolution/algorithms/richardson_lucy.yaml)。

| 参数名 | 作用与单位 | 默认值 | 默认值来源 | 允许值或必要约束 | 推荐范围 |
| --- | --- | --- | --- | --- | --- |
| max_iter | Iteration budget; actual stop may be earlier；iterations | 50 | `configs/deconvolution/algorithms/richardson_lucy.yaml` | int；≥1 | [25, 50, 100] |
| tolerance | Relative numerical stopping accuracy；relative residual | 1e-06 | `configs/deconvolution/algorithms/richardson_lucy.yaml` | float；≥0 | 固定数值/采集条件，按任务显式覆盖 |
| epsilon | Residual-ball radius for constrained CT; Poisson division floor for RL；residual norm or counts, method-specific | 1e-12 | `configs/deconvolution/algorithms/richardson_lucy.yaml` | float；>0 | 固定数值/采集条件，按任务显式覆盖 |
| initial_value | Positive constant object initialization; ignored with explicit initial_estimate；image intensity | 0.5 | `configs/deconvolution/algorithms/richardson_lucy.yaml` | float；>0 | 固定数值/采集条件，按任务显式覆盖 |

保真项模式：`fixed`；可执行项：`['poisson_nll']`。RL multiplicative updates are Poisson only; Gaussian FFT/HQS/blind updates are quadratic only. L2 and unit-variance Gaussian are equivalent; arbitrary WLS not implemented.

### 反卷积：`rl_tv`

默认值来源：[本仓库配置](../../configs/deconvolution/algorithms/rl_tv.yaml)。

| 参数名 | 作用与单位 | 默认值 | 默认值来源 | 允许值或必要约束 | 推荐范围 |
| --- | --- | --- | --- | --- | --- |
| lam | Regularizer weight on the stated unnormalized sum；objective/regularizer | 0.01 | `configs/deconvolution/algorithms/rl_tv.yaml` | float；≥0 | [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0] |
| max_iter | Outer iteration cap；iterations | 50 | `configs/deconvolution/algorithms/rl_tv.yaml` | int；≥1 | [25, 50, 100] |
| tolerance | Relative stopping tolerance; zero requests exact stopping；relative residual | 1e-06 | `configs/deconvolution/algorithms/rl_tv.yaml` | float；≥0 | 固定数值/采集条件，按任务显式覆盖 |
| epsilon | Residual-ball radius for constrained CT; Poisson division floor for RL；residual norm or counts, method-specific | 1e-12 | `configs/deconvolution/algorithms/rl_tv.yaml` | float；>0 | 固定数值/采集条件，按任务显式覆盖 |
| tv_epsilon | Smoothed TV gradient magnitude floor；image intensity/pixel | 0.0001 | `configs/deconvolution/algorithms/rl_tv.yaml` | float；>0 | 固定数值/采集条件，按任务显式覆盖 |
| initial_value | Positive constant object initialization; ignored with explicit initial_estimate；image intensity | 0.5 | `configs/deconvolution/algorithms/rl_tv.yaml` | float；>0 | 固定数值/采集条件，按任务显式覆盖 |

保真项模式：`fixed`；可执行项：`['poisson_nll']`。RL multiplicative updates are Poisson only; Gaussian FFT/HQS/blind updates are quadratic only. L2 and unit-variance Gaussian are equivalent; arbitrary WLS not implemented.

### 耦合条件

Poisson均值包含gain和background；RL-TV的平滑TV分母须保持有效。

目标与数值操作见[共享数学说明](../shared_mathematics.md)，任务相关选参见[参数说明](../parameter_search.md)。

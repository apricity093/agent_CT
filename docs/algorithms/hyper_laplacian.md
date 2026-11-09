# Hyper-Laplacian 反卷积

## 一、算法身份

- **名称**：Hyper-Laplacian 反卷积。
- **一句话简介**：通过半二次分裂求解非凸梯度先验。
- **所属任务**：反卷积。
- **方法类型**：非学习类，无需训练权重。
- **适用范围**：周期已知核、二次数据项、0<p<1；有限罚参数日程得到局部数值解。
- **代码来源**：[apricity093/agent_CT](https://github.com/apricity093/agent_CT)，包版本`0.1.0`，使用随本说明发布的源码（[版本声明](../../pyproject.toml)）。主要实现：[inv_framework/deconvolution/regularized.py](../../inv_framework/deconvolution/regularized.py)。

### 领域入口与目标

| 领域／入口 | 适用范围与实际目标 |
| --- | --- |
| 反卷积 / `hyper_laplacian` | Periodic Gaussian 2D/3D anisotropic sum \|Dx\|^p; 0<p<1; finite penalty HQS; unconstrained 0.5 \|\|Hx-y\|\|^2 + lam sum \|Dx\|^p; nonconvex finite-penalty HQS, not exact global minimization. |

## 二、输入输出

### 最小示例数据

tests/data/deconvolution/cases/public_camera_levin/public.npz：gaussian=(256,256)为含噪观测。psf=(19,19)，原点[9,9]像素。camera图像已抗混叠缩放为256×256、uint8除255，采用周期卷积；图像轴为y、x，不在重建时重新归一化。

示例配置或生成脚本：[configs/deconvolution/benchmarks/standard_public_camera_levin_gaussian_hyper_laplacian.yaml](../../configs/deconvolution/benchmarks/standard_public_camera_levin_gaussian_hyper_laplacian.yaml)。

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
python -m inv_framework.runner --config configs/deconvolution/benchmarks/standard_public_camera_levin_gaussian_hyper_laplacian.yaml --output outputs/algorithm_examples/hyper_laplacian
```

每次运行选择新的输出目录。该命令中的配置可能覆盖部分默认值；实际值以结果中的effective_params为准。

## 四、参数

下面按入口列出参数，保留领域默认值及字段名差异。默认值由本仓库配置维护，属于工程运行起点；表中来源链接给出具体配置位置。显式参数覆盖实验配置，实验配置覆盖算法默认值。

建议范围不是硬边界；先固定观测尺度、离散定义和精度，再比较权重。步长、rho、内层预算以实际数值条件选择。

### 反卷积：`hyper_laplacian`

默认值来源：[本仓库配置](../../configs/deconvolution/algorithms/hyper_laplacian.yaml)。

| 参数名 | 作用与单位 | 默认值 | 默认值来源 | 允许值或必要约束 | 推荐范围 |
| --- | --- | --- | --- | --- | --- |
| lam | Regularizer weight on the stated unnormalized sum；objective/regularizer | 0.01 | `configs/deconvolution/algorithms/hyper_laplacian.yaml` | float；≥0 | [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0] |
| p | Anisotropic gradient exponent; 0<p<1；dimensionless | 0.8 | `configs/deconvolution/algorithms/hyper_laplacian.yaml` | float；>0；<1 | [0.5, 0.8] |
| beta_initial | Initial HQS splitting penalty；objective/gradient^2 | 1.0 | `configs/deconvolution/algorithms/hyper_laplacian.yaml` | float；>0 | [0.1, 1.0, 10.0] |
| beta_max | Final finite HQS penalty, >=beta_initial；objective/gradient^2 | 256.0 | `configs/deconvolution/algorithms/hyper_laplacian.yaml` | float；>0 | 固定数值/采集条件，按任务显式覆盖 |
| beta_factor | HQS penalty growth, strictly greater than 1；dimensionless | 2.0 | `configs/deconvolution/algorithms/hyper_laplacian.yaml` | float；>1 | 固定数值/采集条件，按任务显式覆盖 |
| max_iter | Outer iteration cap；iterations | 100 | `configs/deconvolution/algorithms/hyper_laplacian.yaml` | int；≥1 | [50, 100, 200] |
| tolerance | Relative stopping tolerance; zero requests exact stopping；relative residual | 1e-06 | `configs/deconvolution/algorithms/hyper_laplacian.yaml` | float；≥0 | 固定数值/采集条件，按任务显式覆盖 |

保真项模式：`fixed`；可执行项：`['l2', 'gaussian_nll']`。RL multiplicative updates are Poisson only; Gaussian FFT/HQS/blind updates are quadratic only. L2 and unit-variance Gaussian are equivalent; arbitrary WLS not implemented.

### 耦合条件

0<p<1，beta_factor>1且beta_max≥beta_initial；比较时固定精度与罚参数日程。

目标与数值操作见[共享数学说明](../shared_mathematics.md)，任务相关选参见[参数说明](../parameter_search.md)。

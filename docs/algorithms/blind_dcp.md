# 暗通道盲反卷积（Blind-DCP）

## 一、算法身份

- **名称**：暗通道盲反卷积（Blind-DCP）。
- **一句话简介**：交替估计暗通道/梯度稀疏图像和非负模糊核。
- **所属任务**：反卷积。
- **方法类型**：非学习类，无需训练权重。
- **适用范围**：二维灰度或末轴多通道、周期模型；输入核支持，不提供真核。
- **代码来源**：[apricity093/agent_CT](https://github.com/apricity093/agent_CT)，包版本`0.1.0`，使用随本说明发布的源码（[版本声明](../../pyproject.toml)）。主要实现：[inv_framework/deconvolution/blind.py](../../inv_framework/deconvolution/blind.py)。

### 领域入口与目标

| 领域／入口 | 适用范围与实际目标 |
| --- | --- |
| 反卷积 / `blind_dcp` | 2D grayscale or channel-last, shared spatially invariant PSF; periodic Gaussian model; explicit odd model.kernel_shape; final Hyper-Laplacian p=0.5 HQS Alternating dark-channel/gradient sparse latent estimation and constrained kernel ridge fitting; final p=0.5 Hyper-Laplacian recovery. |

## 二、输入输出

### 最小示例数据

tests/data/deconvolution/cases/public_camera_levin/public.npz：gaussian=(256,256)为含噪观测。盲任务仅读取观测，kernel_shape=[19,19]由配置指定。camera图像已抗混叠缩放为256×256、uint8除255，采用周期卷积；图像轴为y、x，不在重建时重新归一化。

示例配置或生成脚本：[configs/deconvolution/benchmarks/standard_public_camera_levin_gaussian_blind_dcp.yaml](../../configs/deconvolution/benchmarks/standard_public_camera_levin_gaussian_blind_dcp.yaml)。

**正常输出**：reconstruction.npy (256,256) 和result.json；另有psf.npy (19,19)。

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
python -m inv_framework.runner --config configs/deconvolution/benchmarks/standard_public_camera_levin_gaussian_blind_dcp.yaml --output outputs/algorithm_examples/blind_dcp
```

每次运行选择新的输出目录。该命令中的配置可能覆盖部分默认值；实际值以结果中的effective_params为准。

## 四、参数

下面按入口列出参数，保留领域默认值及字段名差异。默认值由本仓库配置维护，属于工程运行起点；表中来源链接给出具体配置位置。显式参数覆盖实验配置，实验配置覆盖算法默认值。

建议范围不是硬边界；先固定观测尺度、离散定义和精度，再比较权重。步长、rho、内层预算以实际数值条件选择。

### 反卷积：`blind_dcp`

默认值来源：[本仓库配置](../../configs/deconvolution/algorithms/blind_dcp.yaml)。

| 参数名 | 作用与单位 | 默认值 | 默认值来源 | 允许值或必要约束 | 推荐范围 |
| --- | --- | --- | --- | --- | --- |
| max_iter | Iteration budget; actual stop may be earlier；iterations | 5 | `configs/deconvolution/algorithms/blind_dcp.yaml` | int；≥1 | [2, 5, 10] |
| scale_factor | Image pyramid ratio; 0<value<1；dimensionless | 0.70710678 | `configs/deconvolution/algorithms/blind_dcp.yaml` | float；>0；<1 | 固定数值/采集条件，按任务显式覆盖 |
| coarse_kernel_size | Coarsest PSF support target, at least 3；pixels | 5 | `configs/deconvolution/algorithms/blind_dcp.yaml` | int；≥3 | 固定数值/采集条件，按任务显式覆盖 |
| kernel_ridge | Ridge penalty for compact nonnegative kernel fitting；objective/kernel^2 | 2.0 | `configs/deconvolution/algorithms/blind_dcp.yaml` | float；>0 | [0.02, 0.2, 2.0, 20.0, 200.0, 2000.0] |
| kernel_max_iter | Iteration budget; actual stop may be earlier；iterations | 100 | `configs/deconvolution/algorithms/blind_dcp.yaml` | int；≥1 | 固定数值/采集条件，按任务显式覆盖 |
| final_lam | Final p=0.5 Hyper-Laplacian recovery weight；objective/regularizer | 0.001 | `configs/deconvolution/algorithms/blind_dcp.yaml` | float；≥0 | [1e-05, 0.0001, 0.001, 0.01, 0.1, 1.0] |
| final_max_iter | Iteration budget; actual stop may be earlier；iterations | 100 | `configs/deconvolution/algorithms/blind_dcp.yaml` | int；≥1 | [50, 100, 200] |
| dark_weight | Dark-channel sparse latent penalty；objective/penalty | 0.004 | `configs/deconvolution/algorithms/blind_dcp.yaml` | float；≥0 | [4e-05, 0.0004, 0.004, 0.04, 0.4, 4.0] |
| gradient_weight | Gradient sparse latent penalty；objective/penalty | 0.004 | `configs/deconvolution/algorithms/blind_dcp.yaml` | float；≥0 | [4e-05, 0.0004, 0.004, 0.04, 0.4, 4.0] |
| patch_size | Odd dark-channel spatial patch width；pixels | 35 | `configs/deconvolution/algorithms/blind_dcp.yaml` | int；≥1 | 固定数值/采集条件，按任务显式覆盖 |
| penalty_initial | Initial dark/gradient latent splitting penalty；objective/variable^2 | 0.01 | `configs/deconvolution/algorithms/blind_dcp.yaml` | float；>0 | [0.001, 0.01, 0.1] |
| penalty_factor | Latent penalty growth, strictly greater than 1；dimensionless | 2.0 | `configs/deconvolution/algorithms/blind_dcp.yaml` | float；>1 | 固定数值/采集条件，按任务显式覆盖 |
| latent_iter | Iteration budget; actual stop may be earlier；iterations | 10 | `configs/deconvolution/algorithms/blind_dcp.yaml` | int；≥1 | 固定数值/采集条件，按任务显式覆盖 |
| cg_max_iter | Iteration budget; actual stop may be earlier；iterations | 100 | `configs/deconvolution/algorithms/blind_dcp.yaml` | int；≥1 | 固定数值/采集条件，按任务显式覆盖 |
| cg_tolerance | Relative numerical stopping accuracy；relative residual | 1e-06 | `configs/deconvolution/algorithms/blind_dcp.yaml` | float；>0 | 固定数值/采集条件，按任务显式覆盖 |

保真项模式：`fixed`；可执行项：`['l2', 'gaussian_nll']`。RL multiplicative updates are Poisson only; Gaussian FFT/HQS/blind updates are quadratic only. L2 and unit-variance Gaussian are equivalent; arbitrary WLS not implemented.

### 耦合条件

核支持为图像内奇数尺寸且至少3；patch_size为奇数，penalty_factor>1。

目标与数值操作见[共享数学说明](../shared_mathematics.md)，任务相关选参见[参数说明](../parameter_search.md)。

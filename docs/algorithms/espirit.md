# ESPIRiT

## 一、算法身份

- **名称**：ESPIRiT。
- **一句话简介**：从 ACS 子空间估计线圈灵敏度图。
- **所属任务**：MRI。
- **方法类型**：非学习类，无需训练权重。
- **适用范围**：这是校准任务，输出灵敏度及特征值；重建应另发显式请求。
- **代码来源**：[apricity093/agent_CT](https://github.com/apricity093/agent_CT)，包版本`0.1.0`，使用随本说明发布的源码（[版本声明](../../pyproject.toml)）。主要实现：[inv_framework/mri/calibration.py](../../inv_framework/mri/calibration.py)。

### 领域入口与目标

| 领域／入口 | 适用范围与实际目标 |
| --- | --- |
| MRI / `espirit` | 2D Cartesian fully sampled ACS; one map set; dense Hermitian eigendecomposition ACS patch SVD and image-domain calibration eigendecomposition; returns one sensitivity map set. |

## 二、输入输出

### 最小示例数据

tests/data/mri/cases/standard_mr_dev/public.npz：calibration_data=(1,4,128,32)，为连续采集ACS，输出网格128×128。数组轴为批次、线圈、y、x，采用中心正交FFT。模拟MR信号已包含组织对比度和复数噪声，运行时不再裁剪或重新归一化。

示例配置或生成脚本：[configs/mri/benchmarks/standard_standard_mr_dev_gaussian_espirit.yaml](../../configs/mri/benchmarks/standard_standard_mr_dev_gaussian_espirit.yaml)。

**正常输出**：sensitivity_maps.npy (1,4,128,128)、eigenvalues.npy和result.json，供后续重建调用。

NPY用`numpy.load`读取；tensor bundle用`torch.load(..., weights_only=True)`读取。结果同时记录执行状态、实际迭代和停止原因；预算结束与数值收敛分别解释。

### 其他领域入口

- MRI：[mri输入契约](../mri/data_format.md)，说明数组轴、几何/采样、单位及预处理。

## 三、运行方式

从算法库根目录执行。Python ≥3.10；基础依赖为PyTorch、NumPy、SciPy、h5py、PyYAML、matplotlib：

```bash
python -m pip install -e ".[dev]"
```

本例使用CPU。完整安装及Windows Conda设置见[依赖说明](../../DEPENDENCIES.md)。无需Agent SDK或大模型服务。

### 最小示例命令

```bash
python -m inv_framework.runner --config configs/mri/benchmarks/standard_standard_mr_dev_gaussian_espirit.yaml --output outputs/algorithm_examples/espirit
```

每次运行选择新的输出目录。该命令中的配置可能覆盖部分默认值；实际值以结果中的effective_params为准。

## 四、参数

下面按入口列出参数，保留领域默认值及字段名差异。默认值由本仓库配置维护，属于工程运行起点；表中来源链接给出具体配置位置。显式参数覆盖实验配置，实验配置覆盖算法默认值。

建议范围不是硬边界；先固定观测尺度、离散定义和精度，再比较权重。步长、rho、内层预算以实际数值条件选择。

### MRI：`espirit`

默认值来源：[本仓库配置](../../configs/mri/algorithms/espirit.yaml)。

| 参数名 | 作用与单位 | 默认值 | 默认值来源 | 允许值或必要约束 | 推荐范围 |
| --- | --- | --- | --- | --- | --- |
| kernel_width | Odd ACS patch width within acquired calibration grid；samples | 5 | `configs/mri/algorithms/espirit.yaml` | int；≥1 | 固定数值/采集条件，按任务显式覆盖 |
| svd_threshold | Singular-value cutoff relative to largest ACS singular value; 0<value<1；dimensionless | 0.02 | `configs/mri/algorithms/espirit.yaml` | float；>0；<1 | 固定数值/采集条件，按任务显式覆盖 |
| crop | Retain dominant calibration eigenvectors above cutoff; 0<=value<=1；dimensionless | 0.95 | `configs/mri/algorithms/espirit.yaml` | float；≥0；≤1 | 固定数值/采集条件，按任务显式覆盖 |

保真项模式：`not_applicable`；可执行项：`[]`。Complex weighted quadratic only; Gaussian uses supplied inverse-variance data_weights or identity; WLS requires explicit weights. Calibration, interpolation and learned inference have no selectable reconstruction data term.

### 耦合条件

kernel_width为奇数并适配ACS；0<svd_threshold<1，0≤crop≤1。

目标与数值操作见[共享数学说明](../shared_mathematics.md)，任务相关选参见[参数说明](../parameter_search.md)。

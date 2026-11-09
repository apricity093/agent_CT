# Zero-filled

## 一、算法身份

- **名称**：Zero-filled。
- **一句话简介**：对采样 k-space 作逆 FFT 并组合线圈图像。
- **所属任务**：MRI。
- **方法类型**：非学习类，无需训练权重。
- **适用范围**：SENSE 组合输出复数，RSS 输出幅值；参考图应匹配线圈组合约定。
- **代码来源**：[apricity093/agent_CT](https://github.com/apricity093/agent_CT)，包版本`0.1.0`，使用随本说明发布的源码（[版本声明](../../pyproject.toml)）。主要实现：[inv_framework/mri/api.py](../../inv_framework/mri/api.py)。

### 领域入口与目标

| 领域／入口 | 适用范围与实际目标 |
| --- | --- |
| MRI / `zero_filled` | Cartesian 2D/3D; explicit maps for sense combination, optional maps for magnitude RSS Centered orthonormal inverse FFT and SENSE combination or magnitude RSS; no optimization objective. |

## 二、输入输出

### 最小示例数据

tests/data/mri/cases/standard_mr_dev/public.npz：kspace与sampling_mask=(1,4,128,128)；SENSE线圈组合另读同形sensitivity_maps。数组轴为批次、线圈、y、x，采用中心正交FFT。模拟MR信号已包含组织对比度和复数噪声，运行时不再裁剪或重新归一化。

示例配置或生成脚本：[configs/mri/benchmarks/standard_standard_mr_dev_gaussian_zero_filled.yaml](../../configs/mri/benchmarks/standard_standard_mr_dev_gaussian_zero_filled.yaml)。

**正常输出**：reconstruction.npy (1,128,128) 和result.json；重建形状与输入网格一致且有限。

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
python -m inv_framework.runner --config configs/mri/benchmarks/standard_standard_mr_dev_gaussian_zero_filled.yaml --output outputs/algorithm_examples/zero_filled
```

每次运行选择新的输出目录。该命令中的配置可能覆盖部分默认值；实际值以结果中的effective_params为准。

## 四、参数

下面按入口列出参数，保留领域默认值及字段名差异。默认值由本仓库配置维护，属于工程运行起点；表中来源链接给出具体配置位置。显式参数覆盖实验配置，实验配置覆盖算法默认值。

建议范围不是硬边界；先固定观测尺度、离散定义和精度，再比较权重。步长、rho、内层预算以实际数值条件选择。

### MRI：`zero_filled`

默认值来源：[本仓库配置](../../configs/mri/algorithms/zero_filled.yaml)。

| 参数名 | 作用与单位 | 默认值 | 默认值来源 | 允许值或必要约束 | 推荐范围 |
| --- | --- | --- | --- | --- | --- |
| coil_combination | Complex sensitivity-normalized SENSE or real magnitude RSS；enum | sense | `configs/mri/algorithms/zero_filled.yaml` | str；sense / rss | ['sense', 'rss'] |

保真项模式：`not_applicable`；可执行项：`[]`。Complex weighted quadratic only; Gaussian uses supplied inverse-variance data_weights or identity; WLS requires explicit weights. Calibration, interpolation and learned inference have no selectable reconstruction data term.

### 耦合条件

SENSE需要maps，RSS参考必须采用RSS定义；不存在优化权重搜索。

目标与数值操作见[共享数学说明](../shared_mathematics.md)，任务相关选参见[参数说明](../parameter_search.md)。

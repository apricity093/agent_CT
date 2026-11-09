# VarNet

## 一、算法身份

- **名称**：VarNet。
- **一句话简介**：使用训练得到的级联网络进行多线圈 MRI 重建。
- **所属任务**：MRI。
- **方法类型**：学习类，使用训练权重。
- **适用范围**：二维 Cartesian 输入；网络结构和 ACS 设置须匹配本地权重。
- **代码来源**：[apricity093/agent_CT](https://github.com/apricity093/agent_CT)，包版本`0.1.0`，使用随本说明发布的源码（[版本声明](../../pyproject.toml)）。主要实现：[inv_framework/mri/varnet_inference.py](../../inv_framework/mri/varnet_inference.py)。

### 领域入口与目标

| 领域／入口 | 适用范围与实际目标 |
| --- | --- |
| MRI / `varnet` | 2D Cartesian multicoil; matching architecture and trained checkpoint required Checkpoint-defined learned cascades with data consistency; no selectable analytic likelihood. |

网络组件改编自 [facebookresearch/fastMRI](https://github.com/facebookresearch/fastMRI) archived main（仓库记录获取日期2026-11-06）；MIT版权声明见[第三方来源](../../THIRD_PARTY_NOTICES.md)。

## 二、输入输出

### 最小示例数据

tests/data/mri/cases/standard_mr_cartesian/public.npz：kspace与sampling_mask=(1,4,256,256)；网络从采样数据估计线圈信息。数组轴为批次、线圈、y、x，采用中心正交FFT。模拟MR信号已包含组织对比度和复数噪声，运行时不再裁剪或重新归一化。

示例配置或生成脚本：[configs/mri/benchmarks/standard_standard_mr_cartesian_gaussian_varnet.yaml](../../configs/mri/benchmarks/standard_standard_mr_cartesian_gaussian_varnet.yaml)。

**正常输出**：reconstruction.npy (1,256,256) 和result.json；重建形状与输入网格一致且有限。

NPY用`numpy.load`读取；tensor bundle用`torch.load(..., weights_only=True)`读取。结果同时记录执行状态、实际迭代和停止原因；预算结束与数值收敛分别解释。

### 其他领域入口

- MRI：[mri输入契约](../mri/data_format.md)，说明数组轴、几何/采样、单位及预处理。

## 三、运行方式

从算法库根目录执行。Python ≥3.10；基础依赖为PyTorch、NumPy、SciPy、h5py、PyYAML、matplotlib：

```bash
python -m pip install -e ".[dev]"
```

本例使用CPU。完整安装及Windows Conda设置见[依赖说明](../../DEPENDENCIES.md)。无需Agent SDK或大模型服务。

配置读取`outputs/weights/knee_leaderboard_state_dict.pt`。该模型文件独立于pip安装；从[fastMRI官方knee权重](https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/knee_leaderboard_state_dict.pt)取得后放置于此，或修改checkpoint路径。级联数、通道数、池化深度须匹配，示例ACS宽度为48。具体加载约定见[MRI模型说明](../mri/algorithms.md)。

### 最小示例命令

```bash
python -m inv_framework.runner --config configs/mri/benchmarks/standard_standard_mr_cartesian_gaussian_varnet.yaml --output outputs/algorithm_examples/varnet
```

每次运行选择新的输出目录。该命令中的配置可能覆盖部分默认值；实际值以结果中的effective_params为准。

## 四、参数

下面按入口列出参数，保留领域默认值及字段名差异。默认值由本仓库配置维护，属于工程运行起点；表中来源链接给出具体配置位置。显式参数覆盖实验配置，实验配置覆盖算法默认值。

建议范围不是硬边界；先固定观测尺度、离散定义和精度，再比较权重。步长、rho、内层预算以实际数值条件选择。

### MRI：`varnet`

默认值来源：[本仓库配置](../../configs/mri/algorithms/varnet.yaml)。

| 参数名 | 作用与单位 | 默认值 | 默认值来源 | 允许值或必要约束 | 推荐范围 |
| --- | --- | --- | --- | --- | --- |
| checkpoint | Explicit local trained weights matching declared VarNet architecture；path | None | `configs/mri/algorithms/varnet.yaml` | str；可为null | 固定数值/采集条件，按任务显式覆盖 |
| num_cascades | VarNet cascades; must match checkpoint；cascades | 12 | `configs/mri/algorithms/varnet.yaml` | int；≥1 | 固定数值/采集条件，按任务显式覆盖 |
| chans | VarNet reconstruction U-Net width; must match checkpoint；channels | 18 | `configs/mri/algorithms/varnet.yaml` | int；≥1 | 固定数值/采集条件，按任务显式覆盖 |
| pools | VarNet reconstruction U-Net depth; must match checkpoint/grid；levels | 4 | `configs/mri/algorithms/varnet.yaml` | int；≥1 | 固定数值/采集条件，按任务显式覆盖 |
| sens_chans | Sensitivity U-Net width; must match checkpoint；channels | 8 | `configs/mri/algorithms/varnet.yaml` | int；≥1 | 固定数值/采集条件，按任务显式覆盖 |
| sens_pools | Sensitivity U-Net depth; must match checkpoint/grid；levels | 4 | `configs/mri/algorithms/varnet.yaml` | int；≥1 | 固定数值/采集条件，按任务显式覆盖 |
| num_low_frequencies | Acquired central ACS width used for sensitivity estimation；k-space columns | 24 | `configs/mri/algorithms/varnet.yaml` | int；≥1 | 固定数值/采集条件，按任务显式覆盖 |

保真项模式：`not_applicable`；可执行项：`[]`。Complex weighted quadratic only; Gaussian uses supplied inverse-variance data_weights or identity; WLS requires explicit weights. Calibration, interpolation and learned inference have no selectable reconstruction data term.

### 耦合条件

网络结构与checkpoint严格匹配；num_low_frequencies对应实际采集的ACS列宽。

目标与数值操作见[共享数学说明](../shared_mathematics.md)，任务相关选参见[参数说明](../parameter_search.md)。

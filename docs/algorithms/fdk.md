# 锥束滤波反投影 FDK

## 一、算法身份

- **名称**：锥束滤波反投影 FDK。
- **一句话简介**：调用 ASTRA FDK_CUDA 重建圆轨迹锥束数据。
- **所属任务**：CT。
- **方法类型**：非学习类，无需训练权重。
- **适用范围**：要求受支持的 cone 几何、等边体素、ASTRA CUDA 与 PyTorch CUDA。
- **代码来源**：[apricity093/agent_CT](https://github.com/apricity093/agent_CT)，包版本`0.1.0`，使用随本说明发布的源码（[版本声明](../../pyproject.toml)）。主要实现：[inv_framework/ct/backends](../../inv_framework/ct/backends)。

### 领域入口与目标

| 领域／入口 | 适用范围与实际目标 |
| --- | --- |
| CT / `fdk` | Cone-beam Feldkamp-Davis-Kress reconstruction. Cone-beam FDK filtered backprojection on ASTRA CUDA; approximate analytic inversion. |

后端采用 [ASTRA Toolbox](https://github.com/astra-toolbox/astra-toolbox) 的 FDK_CUDA；ASTRA按其GPL许可独立安装。

## 二、输入输出

### 最小示例数据

tests/data/cases/cone_3d__spheres_astra_12/arrays.h5：truth/x=(1,12,12,12)，measurement/y_observed=(1,24,36,24)，投影轴为批次、探测器行、角度、探测器列。case.json 给出弧度角和归一化几何距离；无噪线积分。

示例配置或生成脚本：[configs/algorithms/fdk.yaml](../../configs/algorithms/fdk.yaml)。

**正常输出**：三维(1,12,12,12)重建；reconstruction.pt 保存tensor bundle，comparison.png可直接查看，metrics.json与diagnostics.json保存指标、参数和停止状态。

NPY用`numpy.load`读取；tensor bundle用`torch.load(..., weights_only=True)`读取。结果同时记录执行状态、实际迭代和停止原因；预算结束与数值收敛分别解释。

### 其他领域入口

- CT：[ct输入契约](../ct/data_format.md)，说明数组轴、几何/采样、单位及预处理。

## 三、运行方式

从算法库根目录执行。Python ≥3.10；基础依赖为PyTorch、NumPy、SciPy、h5py、PyYAML、matplotlib：

```bash
python -m pip install -e ".[dev]"
```

本例使用CUDA，需支持CUDA的PyTorch和ASTRA。完整安装及Windows Conda设置见[依赖说明](../../DEPENDENCIES.md)。无需Agent SDK或大模型服务。

### 最小示例命令

```bash
invct run fdk --case cone_3d/spheres_astra_12 --config configs/algorithms/fdk.yaml --out outputs/algorithm_examples/fdk --device cuda
```

每次运行选择新的输出目录。该命令中的配置可能覆盖部分默认值；实际值以结果中的effective_params为准。

## 四、参数

下面按入口列出参数，保留领域默认值及字段名差异。默认值由本仓库配置维护，属于工程运行起点；表中来源链接给出具体配置位置。显式参数覆盖实验配置，实验配置覆盖算法默认值。

建议范围不是硬边界；先固定观测尺度、离散定义和精度，再比较权重。步长、rho、内层预算以实际数值条件选择。

### CT：`fdk`

默认值来源：[本仓库配置](../../configs/algorithms/fdk.yaml)。未在YAML显式填写的值由[CT注册表](../../inv_framework/ct/registry.py)补齐。

| 参数名 | 作用与单位 | 默认值 | 默认值来源 | 允许值或必要约束 | 推荐范围 |
| --- | --- | --- | --- | --- | --- |
| filter_type | ASTRA FDK reconstruction filter；enum | ram-lak | `configs/algorithms/fdk.yaml` | str；ram-lak / ramp / ramlak | ['ram-lak', 'ramp', 'ramlak'] |
| short_scan | ASTRA Parker short-scan weighting; requires suitable acquisition coverage；boolean | False | `configs/algorithms/fdk.yaml` | bool | 固定数值/采集条件，按任务显式覆盖 |
| voxel_supersampling | ASTRA FDK subvoxel integration factor；samples/axis | 1 | `configs/algorithms/fdk.yaml` | int；≥1 | 固定数值/采集条件，按任务显式覆盖 |

保真项模式：`not_applicable`；可执行项：`[]`。Analytic inversion or residual-ball feasibility, not an optimization likelihood selector.

### 耦合条件

filter_type与short_scan须匹配扫描范围；voxel_supersampling为正整数。

目标与数值操作见[共享数学说明](../shared_mathematics.md)，任务相关选参见[参数说明](../parameter_search.md)。

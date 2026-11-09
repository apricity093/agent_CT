# 最大似然 EM（MLEM）

## 一、算法身份

- **名称**：最大似然 EM（MLEM）。
- **一句话简介**：以非负乘法更新求解发射 Poisson 计数重建。
- **所属任务**：CT。
- **方法类型**：非学习类，无需训练权重。
- **适用范围**：OSEM 是有序子集变体。发射计数与 X-ray 透射计数是不同模型，不对透射数据套用该更新。
- **代码来源**：[apricity093/agent_CT](https://github.com/apricity093/agent_CT)，包版本`0.1.0`，使用随本说明发布的源码（[版本声明](../../pyproject.toml)）。主要实现：[inv_framework/solvers/statistical.py](../../inv_framework/solvers/statistical.py)。

### 领域入口与目标

| 领域／入口 | 适用范围与实际目标 |
| --- | --- |
| CT / `mlem` | Maximum-likelihood EM for nonnegative count/intensity data. Emission Poisson NLL sum(Ax-y log Ax); multiplicative update with sensitivity normalization, not transmission NLL. |
| CT / `osem` | Ordered-subsets EM for nonnegative count/intensity data. Ordered-subset emission Poisson multiplicative updates; no full-data monotonicity guarantee. |

## 二、输入输出

### 最小示例数据

脚本生成8×8非负方块和8个[0,π)平行束角度（弧度）；以投影为均值生成固定种子Poisson计数。observation=(1,1,8,8)，轴为批次、通道、角度、探测器，单位为计数；不作负对数。

示例配置或生成脚本：[examples/emission_quickstart.py](../../examples/emission_quickstart.py)。

**正常输出**：(1,1,8,8)非负 reconstruction.npy 和 result.json；示例执行两轮。

NPY用`numpy.load`读取；tensor bundle用`torch.load(..., weights_only=True)`读取。结果同时记录执行状态、实际迭代和停止原因；预算结束与数值收敛分别解释。

### 其他领域入口

- CT：[ct输入契约](../ct/data_format.md)，说明数组轴、几何/采样、单位及预处理。

## 三、运行方式

从算法库根目录执行。Python ≥3.10；基础依赖为PyTorch、NumPy、SciPy、h5py、PyYAML、matplotlib：

```bash
python -m pip install -e ".[dev]"
```

本例使用CPU。完整安装及Windows Conda设置见[依赖说明](../../DEPENDENCIES.md)。无需Agent SDK或大模型服务。

### 最小示例命令

```bash
python examples/emission_quickstart.py --algorithm mlem --output outputs/algorithm_examples/mlem
```

每次运行选择新的输出目录。该命令中的配置可能覆盖部分默认值；实际值以结果中的effective_params为准。

将`--algorithm mlem`改为`--algorithm osem`运行两子集变体。

## 四、参数

下面按入口列出参数，保留领域默认值及字段名差异。默认值由本仓库配置维护，属于工程运行起点；表中来源链接给出具体配置位置。显式参数覆盖实验配置，实验配置覆盖算法默认值。

建议范围不是硬边界；先固定观测尺度、离散定义和精度，再比较权重。步长、rho、内层预算以实际数值条件选择。

### CT：`mlem`

默认值来源：[本仓库配置](../../configs/algorithms/mlem.yaml)。未在YAML显式填写的值由[CT注册表](../../inv_framework/ct/registry.py)补齐。

| 参数名 | 作用与单位 | 默认值 | 默认值来源 | 允许值或必要约束 | 推荐范围 |
| --- | --- | --- | --- | --- | --- |
| num_iterations | Maximum outer iterations.；iterations | 25 | `configs/algorithms/mlem.yaml` | int；≥1 | [12, 25, 50] |
| min_value | Optional lower image constraint, coupled to upper bound；image intensity | None | `configs/algorithms/mlem.yaml` | float；可为null | 固定数值/采集条件，按任务显式覆盖 |
| max_value | Optional upper image constraint, coupled to lower bound；image intensity | None | `configs/algorithms/mlem.yaml` | float；可为null | 固定数值/采集条件，按任务显式覆盖 |
| initial_value | Positive constant object initialization; ignored with explicit initial_estimate；image intensity | 1e-06 | `configs/algorithms/mlem.yaml` | float；>0.0 | 固定数值/采集条件，按任务显式覆盖 |
| eps | Positive numerical division floor；denominator units | 1e-08 | `configs/algorithms/mlem.yaml` | float；>0.0 | 固定数值/采集条件，按任务显式覆盖 |

保真项模式：`fixed`；可执行项：`['emission_poisson_nll']`。Emission multiplicative update; transmission counts/log projections are incompatible.

### CT：`osem`

默认值来源：[本仓库配置](../../configs/algorithms/osem.yaml)。未在YAML显式填写的值由[CT注册表](../../inv_framework/ct/registry.py)补齐。

| 参数名 | 作用与单位 | 默认值 | 默认值来源 | 允许值或必要约束 | 推荐范围 |
| --- | --- | --- | --- | --- | --- |
| num_iterations | Maximum outer iterations.；iterations | 25 | `configs/algorithms/osem.yaml` | int；≥1 | [12, 25, 50] |
| min_value | Optional lower image constraint, coupled to upper bound；image intensity | None | `configs/algorithms/osem.yaml` | float；可为null | 固定数值/采集条件，按任务显式覆盖 |
| max_value | Optional upper image constraint, coupled to lower bound；image intensity | None | `configs/algorithms/osem.yaml` | float；可为null | 固定数值/采集条件，按任务显式覆盖 |
| block_size | Views per row-action block; coupled to subset_count and view count；views | None | `configs/algorithms/osem.yaml` | int；可为null；≥1 | 固定数值/采集条件，按任务显式覆盖 |
| subset_count | Balanced angle subsets; <=views, coupled to block_size；subsets | 10 | `configs/algorithms/osem.yaml` | int；可为null；≥1 | 固定数值/采集条件，按任务显式覆盖 |
| order_strategy | Ordered or randomized angle traversal, coupled to seed；enum | ordered | `configs/algorithms/osem.yaml` | str；ordered / random | ['ordered', 'random'] |
| seed | Random view-order seed, fixed for paired comparisons；integer seed | None | `configs/algorithms/osem.yaml` | int；可为null；≥0 | 固定数值/采集条件，按任务显式覆盖 |
| initial_value | Positive constant object initialization; ignored with explicit initial_estimate；image intensity | 1e-06 | `configs/algorithms/osem.yaml` | float；>0.0 | 固定数值/采集条件，按任务显式覆盖 |
| eps | Positive numerical division floor；denominator units | 1e-08 | `configs/algorithms/osem.yaml` | float；>0.0 | 固定数值/采集条件，按任务显式覆盖 |

保真项模式：`fixed`；可执行项：`['emission_poisson_nll']`。Emission multiplicative update; transmission counts/log projections are incompatible.

### 耦合条件

计数非负、初值及数值除法下限为正；OSEM子集须匹配视角。

目标与数值操作见[共享数学说明](../shared_mathematics.md)，任务相关选参见[参数说明](../parameter_search.md)。

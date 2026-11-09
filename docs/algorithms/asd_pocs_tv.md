# ASD-POCS-TV

## 一、算法身份

- **名称**：ASD-POCS-TV。
- **一句话简介**：交替执行观测可行性更新和 TV 下降。
- **所属任务**：CT。
- **方法类型**：非学习类，无需训练权重。
- **适用范围**：这是独立的投影与下降更新方法；需要明确的残差球半径及来源，不是 ADMM 的 TV 配置。
- **代码来源**：[apricity093/agent_CT](https://github.com/apricity093/agent_CT)，包版本`0.1.0`，使用随本说明发布的源码（[版本声明](../../pyproject.toml)）。主要实现：[inv_framework/solvers/splitting.py](../../inv_framework/solvers/splitting.py)。

### 领域入口与目标

| 领域／入口 | 适用范围与实际目标 |
| --- | --- |
| CT / `asd_pocs_tv` | Nonnegative unwhitened residual-ball constrained TV. Nonnegative residual-ball feasibility and TV descent heuristic; feasible stagnation is not a global optimum certificate. |

## 二、输入输出

### 最小示例数据

tests/data/cases/parallel_2d__ellipses_sparse_gaussian/arrays.h5：truth/x=(1,1,64,64)，measurement/y_observed=(1,1,32,64)，投影轴为批次、通道、角度、探测器。case.json 随附弧度角及归一化距离；原图固定乘0.02。投影加入1% RMS高斯噪声，不作负对数。

示例配置或生成脚本：[configs/algorithms/asd_pocs_tv.yaml](../../configs/algorithms/asd_pocs_tv.yaml)。

**正常输出**：二维(1,1,64,64)重建；reconstruction.pt 保存tensor bundle，comparison.png可直接查看，metrics.json与diagnostics.json保存指标、参数和停止状态。

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
invct run asd_pocs_tv --case parallel_2d__ellipses_sparse_gaussian --config configs/algorithms/asd_pocs_tv.yaml --out outputs/algorithm_examples/asd_pocs_tv --device cpu
```

每次运行选择新的输出目录。该命令中的配置可能覆盖部分默认值；实际值以结果中的effective_params为准。

## 四、参数

下面按入口列出参数，保留领域默认值及字段名差异。默认值由本仓库配置维护，属于工程运行起点；表中来源链接给出具体配置位置。显式参数覆盖实验配置，实验配置覆盖算法默认值。

建议范围不是硬边界；先固定观测尺度、离散定义和精度，再比较权重。步长、rho、内层预算以实际数值条件选择。

### CT：`asd_pocs_tv`

默认值来源：[本仓库配置](../../configs/algorithms/asd_pocs_tv.yaml)。未在YAML显式填写的值由[CT注册表](../../inv_framework/ct/registry.py)补齐。

| 参数名 | 作用与单位 | 默认值 | 默认值来源 | 允许值或必要约束 | 推荐范围 |
| --- | --- | --- | --- | --- | --- |
| epsilon | Residual-ball radius for constrained CT; Poisson division floor for RL；residual norm or counts, method-specific | 0.2 | `configs/algorithms/asd_pocs_tv.yaml` | float；≥0.0 | 固定数值/采集条件，按任务显式覆盖 |
| epsilon_source | Required explicit provenance of residual-ball radius, not automatic truth tuning；text | explicit demonstration radius in line-integral units; replace for actual observations | `configs/algorithms/asd_pocs_tv.yaml` | str | 固定数值/采集条件，按任务显式覆盖 |
| residual_metric | Raw masked residual or Gaussian-whitened residual with public sigma；enum | raw | `configs/algorithms/asd_pocs_tv.yaml` | str；raw / gaussian_whitened | ['raw', 'gaussian_whitened'] |
| num_iterations | Iteration budget; actual stop may be earlier；iterations | 20 | `configs/algorithms/asd_pocs_tv.yaml` | int；≥1 | [10, 20, 40] |
| tolerance | Relative numerical stopping accuracy；relative residual | 1e-05 | `configs/algorithms/asd_pocs_tv.yaml` | float；>0.0 | 固定数值/采集条件，按任务显式覆盖 |
| feasibility_tolerance | Relative numerical stopping accuracy；relative residual | 1e-05 | `configs/algorithms/asd_pocs_tv.yaml` | float；>0.0 | 固定数值/采集条件，按任务显式覆盖 |
| tv_mode | Gradient group norm: isotropic or anisotropic where registered；enum | isotropic | `configs/algorithms/asd_pocs_tv.yaml` | str；isotropic / anisotropic | ['isotropic', 'anisotropic'] |
| min_value | Optional lower image constraint, coupled to upper bound；image intensity | 0.0 | `configs/algorithms/asd_pocs_tv.yaml` | float；0.0 | [0.0] |
| max_value | Optional upper image constraint, coupled to lower bound；image intensity | None | `configs/algorithms/asd_pocs_tv.yaml` | float；可为null | 固定数值/采集条件，按任务显式覆盖 |
| relaxation | Row-action relaxation; SART requires 0<value<=1；dimensionless | 1.0 | `configs/algorithms/asd_pocs_tv.yaml` | float；>0.0；≤1.0 | [0.25, 0.5, 1.0] |
| relaxation_decay | ASD-POCS multiplicative row-action relaxation decay；dimensionless | 0.99 | `configs/algorithms/asd_pocs_tv.yaml` | float；>0.0；<1.0 | 固定数值/采集条件，按任务显式覆盖 |
| tv_step | ASD-POCS TV descent scale；relative step | 0.2 | `configs/algorithms/asd_pocs_tv.yaml` | float；>0.0 | 固定数值/采集条件，按任务显式覆盖 |
| tv_step_decay | ASD-POCS multiplicative TV descent decay；dimensionless | 0.95 | `configs/algorithms/asd_pocs_tv.yaml` | float；>0.0；<1.0 | 固定数值/采集条件，按任务显式覆盖 |
| tv_updates | Iteration budget; actual stop may be earlier；iterations | 20 | `configs/algorithms/asd_pocs_tv.yaml` | int；≥1 | 固定数值/采集条件，按任务显式覆盖 |
| update_ratio | ASD-POCS TV versus feasibility update norm ratio；dimensionless | 0.95 | `configs/algorithms/asd_pocs_tv.yaml` | float；>0.0 | 固定数值/采集条件，按任务显式覆盖 |
| block_size | Views per row-action block; coupled to subset_count and view count；views | 1 | `configs/algorithms/asd_pocs_tv.yaml` | int；≥1 | 固定数值/采集条件，按任务显式覆盖 |

保真项模式：`not_applicable`；可执行项：`[]`。Analytic inversion or residual-ball feasibility, not an optimization likelihood selector.

### 耦合条件

epsilon与残差度量必须匹配，epsilon_source说明半径来源；TV下降和投影预算分别记录。

目标与数值操作见[共享数学说明](../shared_mathematics.md)，任务相关选参见[参数说明](../parameter_search.md)。
